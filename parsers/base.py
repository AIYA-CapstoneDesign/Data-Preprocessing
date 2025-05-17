import json
import os
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set
from concurrent.futures import ThreadPoolExecutor

import cv2
from tqdm import tqdm
import subprocess


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
    def __init__(self, dataset_name: str, data_path: str, max_workers: int = None):
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

        # 클립 출력 디렉토리 생성
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "fall"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "nofall"), exist_ok=True)

        # 스레드 풀 생성
        self.executor = ThreadPoolExecutor(max_workers=max_workers or os.cpu_count())

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
        비동기적으로 파싱 작업을 실행.
        """
        annotation_paths = self._filter_annotations(self.data_path)

        tasks: List[asyncio.Task] = []
        for annotation_path in tqdm(annotation_paths, desc="주석 파일 파싱 중..."):
            for idx, info in enumerate(self._parse_annotation(annotation_path)):
                tasks.append(self._save_clip_async(info, idx))

        for f in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="클립 저장 중..."
        ):
            await f

        self.executor.shutdown(wait=True)

    async def _save_clip_async(self, clip_info: ClipInfo, idx: int):
        """
        비동기적으로 클립을 저장. 실제 I/O 작업은 별도 스레드에서 실행.

        Args:
            clip_info (ClipInfo): 클립 정보.
            idx (int): 클립 인덱스.
        """
        # ThreadPoolExecutor를 사용하여 I/O 작업을 별도 스레드에서 실행
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._save_clip, clip_info, idx
        )

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
        # 비디오 파일 존재 확인
        if not os.path.exists(clip_info.video_path):
            print(f"비디오 파일이 존재하지 않습니다: {clip_info.video_path}")
            return None

        cap = cv2.VideoCapture(clip_info.video_path)
        if not cap.isOpened():
            print(f"비디오를 열 수 없음: {clip_info.video_path}")
            return None

        # 영상 메타데이터 추출
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 프레임 범위
        center = (clip_info.action_start + clip_info.action_end) // 2
        half_range = 100 if 59 < fps < 61 else 50
        start_f = max(center - half_range, 0)
        end_f = min(center + half_range, total_frames)
        duration_sec = (end_f - start_f) / fps
        start_sec = start_f / fps

        # 출력 경로
        subdir = "fall" if clip_info.is_fall else "nofall"
        basename = os.path.splitext(os.path.basename(clip_info.video_path))[0]
        out_mp4 = os.path.join(self.output_path, subdir, f"{basename}_{idx}.mp4")

        # FFmpeg 인코더 선택
        # 우선 NVENC -> 실패하면 AMF -> 실패하면 CPU
        enc_try = [
            ("h264_nvenc", ["-qp", "28"]),
            ("h264_amf", []),
            ("libx264", ["-preset", "veryfast"]),
        ]

        # 리사이징 필터 설정
        resize_filter = []
        if width > 1920 or height > 1080:
            resize_filter = [
                "-vf",
                "scale='if(gt(iw,1920),1920,iw)':'if(gt(ih,1080),1080,ih)':force_original_aspect_ratio=decrease",
            ]

        for codec, extra in enc_try:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-ss", f"{start_sec:.3f}",
                "-t", f"{duration_sec:.3f}",
                "-i", clip_info.video_path,
                "-r", "30" if 59 < fps < 61 else str(int(round(fps))),
                *resize_filter,
                "-c:v", codec,
                *extra,
                "-y",
                out_mp4,
            ]
            ret = subprocess.call(cmd)
            if ret == 0:
                break  # success

        if ret != 0:
            print(f"[ERR] ffmpeg failed: {out_mp4}")
            return None

        # 메타데이터 JSON
        if clip_info.split is not None or clip_info.action_position is not None:
            json_out = os.path.join(self.output_path, subdir, f"{basename}_{idx}.json")
            with open(json_out, "w", encoding="utf-8") as f:
                payload = {}
                if clip_info.split is not None:
                    payload["split"] = clip_info.split
                if clip_info.action_position is not None:
                    payload["action_position"] = clip_info.action_position
                json.dump(payload, f, ensure_ascii=False, indent=4)

        return out_mp4
