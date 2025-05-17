import glob
import json
import os
import random
from typing import List

from parsers.base import BaseAnnotationParser, ClipInfo


class DatasetAParser(BaseAnnotationParser):
    def __init__(self, data_path: str, max_workers: int = 8):
        """
        DatasetA 파서 초기화

        Args:
            data_path (str): 데이터셋 위치 경로
            max_workers (int): 동시 처리 작업자 수
        """
        super().__init__("dataset_a", data_path, max_workers=max_workers)

    def _parse_annotation(self, annotation_path: str) -> List[ClipInfo]:
        try:
            with open(annotation_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"JSON 디코딩 오류: {annotation_path}")
            return []
        except Exception as e:
            print(f"파일 읽기 오류 {annotation_path}: {str(e)}")
            return []

        # 중요 키 존재 확인
        missing_keys = []
        for key in ["scene_info"]:
            if key not in data:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"경고: {annotation_path} 파일에 다음 키가 없습니다: {', '.join(missing_keys)}")
            return []
        
        # 낙상 정보 확인
        if "scene_IsFall" not in data["scene_info"]:
            print(f"경고: {annotation_path} 파일에 scene_IsFall 키가 없습니다.")
            return []
            
        # 낙상 여부 확인
        is_fall = data["scene_info"]["scene_IsFall"] == "낙상"
        
        # 비디오 경로 설정
        video_path = os.path.join(
            os.path.dirname(annotation_path).replace("annotation", "video"),
            os.path.splitext(os.path.basename(annotation_path))[0] + ".mp4",
        )
        
        # 비디오 파일 존재 확인
        if not os.path.exists(video_path):
            print(f"비디오 파일이 존재하지 않습니다: {video_path}")
            return []

        # 훈련/검증 분할 확인
        split = None
        if "Training" in video_path:
            split = "train"
        elif "Validation" in video_path:
            split = "val"

        # 데이터셋 A의 경우 한 영상에 단일 행동 데이터
        clip_infos = []
        if is_fall:
            # 낙상 데이터 필수 키 확인
            if "sensordata" not in data or "fall_start_frame" not in data["sensordata"] or "fall_end_frame" not in data["sensordata"]:
                print(f"낙상 데이터 필수 정보 누락: {annotation_path}")
                return []
                
            clip_infos.append(
                ClipInfo(
                    video_path=video_path,
                    action_start=data["sensordata"]["fall_start_frame"],
                    action_end=data["sensordata"]["fall_end_frame"],
                    is_fall=is_fall,
                    split=split,
                )
            )
        # 비낙상일 경우 클립에서 100프레임 클립을 3개 랜덤 샘플링
        else:
            # 비낙상 데이터 필수 키 확인
            if "scene_info" not in data or "scene_length" not in data["scene_info"]:
                print(f"비낙상 데이터 필수 정보 누락: {annotation_path}")
                return []
                
            scene_length = data["scene_info"]["scene_length"]
            if scene_length <= 100:
                print(f"영상 길이가 너무 짧습니다: {annotation_path}")
                return []
                
            for _ in range(3):
                action_start = random.randint(0, scene_length - 100)
                action_end = action_start + 100
                clip_infos.append(
                    ClipInfo(
                        video_path=video_path,
                        action_start=action_start,
                        action_end=action_end,
                        is_fall=is_fall,
                        split=split,
                    )
                )
        return clip_infos

    def _filter_annotations(self, data_path: str) -> set[str]:
        # 모든 주석 파일 경로 반환
        try:
            annotation_file_paths = glob.glob(os.path.join(data_path, "**", "*.json"), recursive=True)
            # data.json 파일 제외
            for path in list(annotation_file_paths):
                if path.endswith("data.json"):
                    annotation_file_paths.remove(path)
            return set(annotation_file_paths)
        except Exception as e:
            print(f"주석 파일 필터링 오류: {str(e)}")
            return set()
