import glob
import json
import os
from typing import List

from parsers.base import BaseAnnotationParser, ClipInfo


class DatasetAParser(BaseAnnotationParser):
    def __init__(self, data_path: str):
        super().__init__("dataset_a", data_path)

    def _parse_annotation(self, annotation_path: str) -> List[ClipInfo]:
        with open(annotation_path, "r") as f:
            data = json.load(f)

        is_fall = data["scene_info"]["scene_IsFall"] == "낙상"
        video_path = os.path.join(
            os.path.dirname(annotation_path).replace("label", "video"),
            os.path.splitext(os.path.basename(annotation_path))[0] + ".mp4",
        )
        if not os.path.exists(video_path):
            return []

        split = None
        if "Training" in video_path:
            split = "train"
        elif "Validation" in video_path:
            split = "val"

        # 데이터셋 A의 경우 한 영상에 단일 행동 데이터
        clip_infos = []
        if is_fall:
            clip_infos.append(
                ClipInfo(
                    video_path=video_path,
                    action_start=data["sensordata"]["fall_start_frame"],
                    action_end=data["sensordata"]["fall_end_frame"],
                    is_fall=is_fall,
                    split=split,
                )
            )
        else:
            clip_infos.append(
                ClipInfo(
                    video_path=video_path,
                    action_start=0,
                    action_end=data["scene_info"]["scene_length"],
                    is_fall=is_fall,
                    split=split,
                )
            )
        return clip_infos

    def _filter_annotations(self, data_path: str) -> set[str]:
        # 모든 주석 파일 경로 반환
        return set(glob.glob(os.path.join(data_path, "**", "*.json"), recursive=True))
