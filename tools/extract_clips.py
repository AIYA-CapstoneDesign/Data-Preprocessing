import argparse
import glob
import json
import os

import cv2

parser = argparse.ArgumentParser(
    description="Extract action clips from the raw video dataset"
)
parser.add_argument("--data-path", type=str, default="data/raw")
parser.add_argument("--output-path", type=str, default="data/clips")
parser.add_argument("--clip-length", type=int, default=100)

args = parser.parse_args()


def extract_clips(data_path: str, output_path: str, clip_length: int = 100):
    half_clip_length = clip_length // 2
    for json_file_path in glob.glob(os.path.join(data_path, "*.json")):
        with open(json_file_path, "r") as f:
            data = json.load(f)

        data = data["annotations"]

        video_path = os.path.join(data_path.replace("라벨링", "원천"), data["resource"])
        fps = float(data["fps"])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"비디오를 열 수 없음: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"프레임 읽기 실패")
        height, width = frame.shape[:2]

        for action_idx, video_info in enumerate(data["object"]):
            action_type = video_info["actionType"]

            start_frame = video_info["startFrame"]
            end_frame = video_info["endFrame"]
            if start_frame > end_frame:
                print(f"start_frame > end_frame: {start_frame} > {end_frame}")
                continue
            center_frame = (start_frame + end_frame) // 2

            start_frame = center_frame - half_clip_length
            end_frame = center_frame + half_clip_length

            if start_frame < 0:
                start_frame = 0
            elif end_frame > total_frames:
                end_frame = total_frames

            video_path = os.path.join(
                output_path,
                action_type,
                f"{os.path.basename(video_path)}_{action_idx}.mp4",
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            for frame_no in range(start_frame, end_frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = cap.read()
                if not ret:
                    print(f"경고: 프레임 {frame_no} 읽지 못함")
                    continue
                writer.write(frame)

            cap.release()
            writer.release()
            print(f"'{video_path}'에 저장 완료.")


if __name__ == "__main__":
    extract_clips(args.data_path, args.output_path, args.clip_length)
