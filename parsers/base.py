import json
import os
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set
from concurrent.futures import ThreadPoolExecutor

import cv2
from tqdm import tqdm


@dataclass
class ClipInfo:
    """
    비디오 클립 추출에 필요한 정보를 나타내는 데이터 클래스.

    Attributes:
        video_path (str): 원본 비디오 파일 경로.
        action_start (int): 동작이 시작되는 프레임 인덱스.
        action_end (int): 동작이 끝나는 프레임 인덱스.
        is_fall (bool): 낙상 이벤트 포함 여부.
        split (Optional[str]): 데이터셋 분할 정보 ('train' or 'val').
        action_position (Optional[tuple[int, int]]): 동작을 수행하는 Actor의 프레임 내 좌표 (x, y), 시작 프레임 기준.
    """

    video_path: str
    action_start: int
    action_end: int
    is_fall: bool
    split: Optional[str] = None
    action_position: Optional[tuple[int, int]] = None


class BaseAnnotationParser(ABC):
    def __init__(self, dataset_name: str, data_path: str, max_workers: int = 4):
        """
        파서 초기화 및 클립 출력 디렉토리 생성.

        Args:
            dataset_name (str): 데이터셋 고유 이름.
            data_path (str): 주석 파일이 위치한 디렉토리 경로.
            max_workers (int): 동시에 처리할 작업자 수.
        """
        self.dataset_name = dataset_name
        self.data_path = os.path.join(data_path, self.dataset_name)
        self.output_path = f"./data/clips/"
        self.max_workers = max_workers

        # 클립 출력 디렉토리 생성
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "fall"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "nofall"), exist_ok=True)

    def parse(self):
        """
        전체 파싱 파이프라인 실행:
        1. 주석 파일 필터링
        2. 각 파일을 ClipInfo 리스트로 파싱
        3. 클립 및 메타데이터 저장

        Raises:
            IOError: 비디오 파일을 열 수 없을 때 발생.
            RuntimeError: 프레임을 읽을 수 없을 때 발생.
        """
        asyncio.run(self._run_parsing())

    async def _run_parsing(self):
        """
        비동기적으로 파싱 작업을 실행합니다.
        """
        annotation_paths = self._filter_annotations(self.data_path)
        tasks = []

        for annotation_path in annotation_paths:
            clip_infos = self._parse_annotation(annotation_path)
            for idx, clip_info in enumerate(clip_infos):
                tasks.append(self._save_clip_async(clip_info, idx))

        # asyncio.gather를 사용하여 모든 태스크를 동시에 실행
        for completed in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await completed

    async def _save_clip_async(self, clip_info: ClipInfo, idx: int):
        """
        비동기적으로 클립을 저장합니다. 실제 I/O 작업은 별도 스레드에서 실행됩니다.

        Args:
            clip_info (ClipInfo): 클립 정보.
            idx (int): 클립 인덱스.
        """
        # ThreadPoolExecutor를 사용하여 I/O 작업을 별도 스레드에서 실행
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, self._save_clip, clip_info, idx)

    @abstractmethod
    def _parse_annotation(self, annotation_path: str) -> List[ClipInfo]:
        """
        단일 주석 파일을 읽어 영상 내 행동 정보들을 담은
        ClipInfo 객체 리스트로 파싱.

        Args:
            annotation_path (str): 주석 파일 경로.

        Returns:
            List[ClipInfo]: 생성된 ClipInfo 객체 리스트.
        """
        pass

    @abstractmethod
    def _filter_annotations(self, data_path: str) -> Set[str]:
        """
        주석 파일 경로를 식별하여 리스트로 반환.

        Args:
            data_path (str): 주석 파일 디렉토리 경로.

        Returns:
            Set[str]: 처리할 주석 파일 경로 집합.
        """
        pass

    def _save_clip(self, clip_info: ClipInfo, idx: int):
        """
        ClipInfo에 따라 비디오 클립 추출 후 메타데이터와 함께 저장.

        Args:
            clip_info (ClipInfo): 클립 정보.
            idx (int): 클립 파일 이름 중복 방지를 위한 인덱스.

        Raises:
            IOError: 비디오 파일을 열 수 없을 때 발생.
            RuntimeError: 프레임을 읽을 수 없을 때 발생.

        Returns:
            str: 저장된 비디오 경로
        """
        cap = cv2.VideoCapture(clip_info.video_path)
        if not cap.isOpened():
            raise IOError(f"비디오를 열 수 없음: {clip_info.video_path}")

        # 영상 메타데이터 추출
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        is_60fps = fps > 59 and fps < 61
        target_fps = 30 if is_60fps else fps  # 60fps인 경우 30fps로 변환

        # FPS에 따른 클립 범위 계산
        # TODO: 30FPS, 60FPS 외의 경우 추가 필요
        if is_60fps:
            half_clip_range = 100
        else:
            half_clip_range = 50

        # 비디오 크기 계산을 위해 첫 프레임 읽기
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"프레임 읽기 실패: {clip_info.video_path}")

        # 비디오 크기 계산
        height, width = frame.shape[:2]

        # 클립 중심 프레임 계산
        action_center = (clip_info.action_start + clip_info.action_end) // 2

        # 추출할 클립의 시작 프레임과 끝 프레임 계산
        start_frame = action_center - half_clip_range
        end_frame = action_center + half_clip_range

        # 예외상황 시 클립 범위 조정
        if start_frame < 0:
            start_frame = 0
        elif end_frame > total_frames:
            end_frame = total_frames

        # 클립 저장 경로 생성
        video_path = os.path.join(
            self.output_path,
            "fall" if clip_info.is_fall else "nofall",
            f"{os.path.splitext(os.path.basename(clip_info.video_path))[0]}_{idx}.mp4",
        )

        # 클립 저장 객체 생성 및 코덱 설정
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, target_fps, (width, height))

        # 시작 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 클립 추출 - 순차적으로 읽어서 처리
        frame_count = 0
        while frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # 60FPS 영상의 경우 30FPS로 저장할 수 있도록 처리
            if is_60fps:
                if frame_count % 2 == 0:  # 짝수 프레임만 저장 (2배 속도 향상)
                    writer.write(frame)
            else:
                # 30FPS 영상은 모든 프레임 저장
                writer.write(frame)

            frame_count += 1

        # 비디오 리소스 해제
        cap.release()
        writer.release()

        # 메타데이터 저장
        if clip_info.split is not None and clip_info.action_position is not None:
            json_path = os.path.join(
                self.output_path,
                "fall" if clip_info.is_fall else "nofall",
                f"{os.path.splitext(os.path.basename(video_path))[0]}.json",
            )
            json_info = {
                "split": clip_info.split,
                "action_position": clip_info.action_position,
            }
            with open(json_path, "w") as f:
                json.dump(json_info, f)

        tqdm.write(f"클립 {idx} 저장 완료: {video_path}")
        return video_path
