import argparse
import glob
import os
import pickle
import tempfile
import json
from typing import List, Tuple, Union

import cv2
import numpy as np
import tqdm

from models.faster_rcnn import FasterRCNN
from models.hrnet import HRNet
from models.yolov8 import YOLOv8
from models.yolov11_pose import YOLOv11Pose
from trackers.bytetrack import ByteTrackTracker

parser = argparse.ArgumentParser()
parser.add_argument(
    "--clips-path", type=str, default="data/clips", help="영상 클립 경로"
)
parser.add_argument(
    "--output-path", type=str, default="data/keypoints", help="결과 저장 경로"
)
parser.add_argument("--det-score-thr", type=float, default=0.5, help="검출 점수 임계값")
parser.add_argument("--device", type=str, default="cuda", help="GPU 사용 여부")
parser.add_argument(
    "--det-model",
    type=str,
    default="yolov8",
    choices=["yolov8", "faster_rcnn", "yolov11-pose"],
    help="모델 선택",
)
parser.add_argument("--bbox-scale", type=float, default=1.25, help="포즈 박스 스케일")
parser.add_argument("--visualize", action="store_true", help="시각화 저장")
parser.add_argument("--mode-48", action="store_true", help="384x288 모드 사용")
parser.add_argument("--smooth-keypoints", action="store_true", help="키포인트 안정화")
parser.add_argument("--smooth-window", type=int, default=5, help="스무딩 윈도우 크기")
parser.add_argument("--batch-size", type=int, default=1, help="YOLOv11 배치 크기")
parser.add_argument("--buffer-size", type=int, default=16, help="프레임 버퍼 크기")
parser.add_argument("--static-filter", action="store_true", help="정적 물체 필터링")
parser.add_argument(
    "--static-thresh", type=float, default=5.0, help="정적 필터 임계값(px)"
)
parser.add_argument("--static-frames", type=int, default=10, help="최소 분석 프레임 수")
parser.add_argument("--log-interval", type=int, default=100, help="로그 출력 간격")
args = parser.parse_args()


def extract_frames(
    video_path: str, out_dir: str = None
) -> Tuple[List[str], List[np.ndarray]]:
    """비디오 프레임 추출"""
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"비디오를 열 수 없습니다: {video_path}")

    frame_paths = []
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

        if out_dir:
            frame_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

        frame_idx += 1

    cap.release()
    return frame_paths, frames


def process_frames_with_pose_estimation(
    hrnet: HRNet,
    detector: Union[YOLOv8, FasterRCNN, YOLOv11Pose],
    tracker: ByteTrackTracker,
    frames: List[np.ndarray],
    visualize: bool = False,
    visualize_dir: str = None,
    min_track_ratio: float = 0.8,
    filter_static: bool = False,
    static_thresh: float = 5.0,
    static_min_frames: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """포즈 추정 수행"""
    frame_results = []
    track_id_counts = {}
    track_positions = {}

    # 배치 처리 설정
    use_batch = args.batch_size > 1 and isinstance(detector, YOLOv11Pose)

    # 프레임 처리 (배치 또는 개별)
    if use_batch:
        # 배치 처리
        frame_buffer = []
        frame_indices = []
        batch_size = min(args.batch_size, args.buffer_size)

        pbar = tqdm.tqdm(range(len(frames)), desc="프레임", leave=False)

        for frame_idx in pbar:
            frame_buffer.append(frames[frame_idx])
            frame_indices.append(frame_idx)

            if len(frame_buffer) >= batch_size or frame_idx == len(frames) - 1:
                # 배치가 가득 차거나 마지막 프레임이면 처리
                try:
                    # 배치 추론
                    batch_results = detector.process_batch(frame_buffer)

                    for i, (frame_idx, frame, results) in enumerate(
                        zip(frame_indices, frame_buffer, batch_results)
                    ):
                        # 추적 및 결과 처리
                        detections = results["bboxes"]
                        detections = tracker(detections)
                        results["bboxes"] = detections

                        # 트랙 ID 카운트 및 위치 저장
                        for bbox in results["bboxes"]:
                            track_id = (
                                bbox.track_id if bbox.track_id is not None else -1
                            )
                            if track_id not in track_id_counts:
                                track_id_counts[track_id] = 0
                            track_id_counts[track_id] += 1

                            # 움직임 분석용 위치 저장
                            if filter_static and track_id != -1:
                                center_x = (bbox.x1 + bbox.x2) / 2
                                center_y = (bbox.y1 + bbox.y2) / 2

                                if track_id not in track_positions:
                                    track_positions[track_id] = []
                                track_positions[track_id].append((center_x, center_y))

                        frame_results.append(results)

                        # 시각화
                        if visualize and visualize_dir:
                            visualize_frame(
                                frames[frame_idx], results, frame_idx, visualize_dir
                            )

                except Exception:
                    # 배치 실패 시 개별 처리로 폴백
                    for frame_idx, frame in zip(frame_indices, frame_buffer):
                        try:
                            result = detector(frame)
                            detections = result["bboxes"]
                            detections = tracker(detections)
                            result["bboxes"] = detections

                            for bbox in result["bboxes"]:
                                track_id = (
                                    bbox.track_id if bbox.track_id is not None else -1
                                )
                                if track_id not in track_id_counts:
                                    track_id_counts[track_id] = 0
                                track_id_counts[track_id] += 1

                                if filter_static and track_id != -1:
                                    center_x = (bbox.x1 + bbox.x2) / 2
                                    center_y = (bbox.y1 + bbox.y2) / 2

                                    if track_id not in track_positions:
                                        track_positions[track_id] = []
                                    track_positions[track_id].append(
                                        (center_x, center_y)
                                    )

                            frame_results.append(result)

                            if visualize and visualize_dir:
                                visualize_frame(frame, result, frame_idx, visualize_dir)

                        except Exception:
                            frame_results.append(
                                {"keypoints": [], "scores": [], "bboxes": []}
                            )

                # 버퍼 초기화
                frame_buffer = []
                frame_indices = []
    else:
        # 개별 프레임 처리
        pbar = tqdm.tqdm(frames, desc="프레임", leave=False)

        for frame_idx, frame in enumerate(pbar):
            try:
                # 모델에 따른 처리
                if isinstance(detector, YOLOv11Pose):
                    # YOLOv11-Pose: 직접 포즈 추정
                    results = detector(frame)
                    detections = results["bboxes"]
                    detections = tracker(detections)
                    results["bboxes"] = detections
                else:
                    # 검출 + HRNet 조합
                    detections = detector(frame)
                    detections = tracker(detections)
                    results = hrnet(frame, detections)

                # 트랙 ID 카운트 및 위치 저장
                for bbox in results["bboxes"]:
                    track_id = bbox.track_id if bbox.track_id is not None else -1
                    if track_id not in track_id_counts:
                        track_id_counts[track_id] = 0
                    track_id_counts[track_id] += 1

                    if filter_static and track_id != -1:
                        center_x = (bbox.x1 + bbox.x2) / 2
                        center_y = (bbox.y1 + bbox.y2) / 2

                        if track_id not in track_positions:
                            track_positions[track_id] = []
                        track_positions[track_id].append((center_x, center_y))

                frame_results.append(results)

                # 시각화
                if visualize and visualize_dir:
                    visualize_frame(frame, results, frame_idx, visualize_dir)

            except Exception:
                frame_results.append({"keypoints": [], "scores": [], "bboxes": []})

    # 유효 트랙 ID 필터링 (출현 비율)
    num_frames = len(frames)
    min_frames = int(num_frames * min_track_ratio)
    valid_track_ids = [
        track_id
        for track_id, count in track_id_counts.items()
        if count >= min_frames and track_id != -1
    ]

    # 정적 객체 필터링
    if filter_static:
        moving_track_ids = []
        for track_id in valid_track_ids:
            positions = track_positions.get(track_id, [])

            if len(positions) >= static_min_frames:
                # 움직임 계산 (표준편차)
                x_positions = [pos[0] for pos in positions]
                y_positions = [pos[1] for pos in positions]
                x_std = np.std(x_positions)
                y_std = np.std(y_positions)
                position_std = np.sqrt(x_std**2 + y_std**2)

                # 임계값 이상 움직인 객체만 선택
                if position_std >= static_thresh:
                    moving_track_ids.append(track_id)
            else:
                # 프레임이 충분하지 않은 경우 일단 포함
                moving_track_ids.append(track_id)

        valid_track_ids = moving_track_ids

    # 키포인트 배열 구성
    trackid_to_idx = {track_id: idx for idx, track_id in enumerate(valid_track_ids)}
    if not trackid_to_idx:
        return np.array([]), np.array([])

    num_persons = len(trackid_to_idx)
    num_keypoints = 17  # COCO 포맷

    # 결과 배열 초기화
    keypoints = np.zeros((num_persons, num_frames, num_keypoints, 2), dtype=np.float32)
    keypoint_scores = np.zeros(
        (num_persons, num_frames, num_keypoints), dtype=np.float32
    )

    # 각 프레임의 결과를 통합
    for frame_idx, frame_result in enumerate(frame_results):
        for kps, scores, bbox in zip(
            frame_result["keypoints"],
            frame_result["scores"],
            frame_result["bboxes"],
        ):
            track_id = bbox.track_id if bbox.track_id is not None else -1
            if track_id in trackid_to_idx:
                person_idx = trackid_to_idx[track_id]
                keypoints[person_idx, frame_idx] = kps
                keypoint_scores[person_idx, frame_idx] = scores

    return keypoints, keypoint_scores


def visualize_frame(frame, results, frame_idx, visualize_dir):
    """프레임 시각화"""
    vis_frame = frame.copy()
    keypoints_list = results["keypoints"]
    scores_list = results["scores"]
    bboxes = results["bboxes"]

    # COCO 키포인트 연결
    connections = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),  # 얼굴
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),  # 팔
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),  # 몸통
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),  # 다리
    ]

    # 키포인트 색상
    colors = [
        (255, 0, 0),
        (255, 85, 0),
        (255, 170, 0),
        (255, 255, 0),
        (170, 255, 0),
        (85, 255, 0),
        (0, 255, 0),
        (0, 255, 85),
        (0, 255, 170),
        (0, 255, 255),
        (0, 170, 255),
        (0, 85, 255),
        (0, 0, 255),
        (85, 0, 255),
        (170, 0, 255),
        (255, 0, 255),
        (255, 0, 170),
    ]

    # 결과 시각화
    for keypoints, scores, bbox in zip(keypoints_list, scores_list, bboxes):
        # 바운딩 박스
        cv2.rectangle(
            vis_frame,
            (int(bbox.x1), int(bbox.y1)),
            (int(bbox.x2), int(bbox.y2)),
            (0, 255, 0),
            2,
        )

        # 트랙 ID
        track_id_text = f"ID: {bbox.track_id}" if bbox.track_id is not None else ""
        cv2.putText(
            vis_frame,
            track_id_text,
            (int(bbox.x1), int(bbox.y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        # 키포인트
        for k_idx, (kp, score) in enumerate(zip(keypoints, scores)):
            if score > 0.3:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(vis_frame, (x, y), 3, colors[k_idx], -1)

        # 스켈레톤 연결선
        for connection in connections:
            if scores[connection[0]] > 0.3 and scores[connection[1]] > 0.3:
                pt1 = tuple(map(int, keypoints[connection[0]]))
                pt2 = tuple(map(int, keypoints[connection[1]]))
                cv2.line(vis_frame, pt1, pt2, (0, 255, 255), 2)

    os.makedirs(visualize_dir, exist_ok=True)
    cv2.imwrite(os.path.join(visualize_dir, f"frame_{frame_idx:06d}.jpg"), vis_frame)


def convert_to_keypoints(clips_path: str, output_path: str):
    """비디오 클립 처리 및 키포인트 추출"""
    # 모델 초기화
    use_cuda = args.device == "cuda"

    # HRNet 모델 초기화
    hrnet = HRNet(
        onnx_path=(
            "models/onnx/hrnet_48.onnx" if args.mode_48 else "models/onnx/hrnet_32.onnx"
        ),
        cuda=use_cuda,
        mode_48=args.mode_48,
        pose_bbox_scale=args.bbox_scale,
        enable_smoothing=args.smooth_keypoints,
        smooth_window_size=args.smooth_window,
    )

    # 검출기 초기화
    detector = None
    if args.det_model == "yolov11-pose":
        person_class_id = 0
        detector = YOLOv11Pose(
            onnx_path="models/onnx/yolo11m-pose.onnx",
            cuda=use_cuda,
            person_class_id=person_class_id,
            score_threshold=args.det_score_thr,
            iou_threshold=0.45,
            batch_size=args.batch_size,
            enable_smoothing=args.smooth_keypoints,
            smooth_window_size=args.smooth_window,
        )
    elif args.det_model == "yolov8":
        person_class_id = 0
        detector = YOLOv8(
            onnx_path="models/onnx/yolov8x.onnx",
            cuda=use_cuda,
            person_class_id=person_class_id,
            score_threshold=None,
        )
    else:
        person_class_id = 1
        detector = FasterRCNN(
            onnx_path="models/onnx/faster_rcnn.onnx",
            cuda=use_cuda,
            person_class_id=person_class_id,
            score_threshold=None,
        )

    # 추적기 초기화
    tracker = ByteTrackTracker(
        high_thresh=0.6,
        low_thresh=0.2,
        target_classes=[person_class_id],
        match_iou_thresh=0.3,
        max_lost=30,
    )

    # 출력 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)

    os.makedirs(os.path.join(output_path, "train", "fall"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "train", "nofall"), exist_ok=True)

    os.makedirs(os.path.join(output_path, "val", "fall"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "val", "nofall"), exist_ok=True)

    os.makedirs(os.path.join(output_path, "unknown", "fall"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "unknown", "nofall"), exist_ok=True)

    if args.visualize:
        os.makedirs(os.path.join(output_path, "visualize", "fall"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "visualize", "nofall"), exist_ok=True)

    # 라벨링 클래스 (디렉토리 기반)
    labels = ["fall", "nofall"]

    # 처리 통계
    clip_stats = {"total": 0, "valid": 0}
    class_stats = {label: {"total": 0, "valid": 0} for label in labels}

    # 인식 인원수 통계 (1인당 1클립 데이터셋 특성 반영)
    persons_stats = {0: 0, 1: 0, 2: 0, "3+": 0}

    # 각 클래스별 처리
    for label in labels:
        class_dir = os.path.join(clips_path, label)
        if not os.path.exists(class_dir):
            print(f"경고: {class_dir} 디렉토리가 존재하지 않습니다.")
            continue

        # 클립 목록 수집
        clip_paths = glob.glob(os.path.join(class_dir, "*.mp4"))
        class_stats[label]["total"] = len(clip_paths)
        clip_stats["total"] += len(clip_paths)

        print(f"\n[{label}] 총 {len(clip_paths)}개 클립 처리 시작")

        # 클립 처리
        for clip_idx, clip_path in enumerate(
            tqdm.tqdm(clip_paths, desc=f"{label} 처리", unit="clip")
        ):
            try:
                clip_name = os.path.splitext(os.path.basename(clip_path))[0]

                # 진행 로그 (지정된 간격마다 출력)
                if clip_idx % args.log_interval == 0 or clip_idx == len(clip_paths) - 1:
                    progress = (clip_idx + 1) / len(clip_paths) * 100
                    print(
                        f"[{label}] 진행률: {progress:.1f}% ({clip_idx+1}/{len(clip_paths)})"
                    )

                # 비디오 프레임 추출
                with tempfile.TemporaryDirectory() as tmp_dir:
                    frame_paths, frames = extract_frames(clip_path, tmp_dir)

                    if len(frames) == 0:
                        continue

                    # 추적기 및 스무딩 버퍼 초기화
                    tracker.reset()
                    if args.smooth_keypoints:
                        hrnet.clear_smoothing_buffers()

                    # 시각화 디렉토리
                    temp_visualize_dir = None
                    if args.visualize:
                        temp_visualize_dir = os.path.join(tmp_dir, "visualize")
                        os.makedirs(temp_visualize_dir, exist_ok=True)

                    # 포즈 추정 수행
                    keypoints, keypoint_scores = process_frames_with_pose_estimation(
                        hrnet=hrnet,
                        detector=detector,
                        tracker=tracker,
                        frames=frames,
                        visualize=args.visualize,
                        visualize_dir=temp_visualize_dir,
                        min_track_ratio=0.8,
                        filter_static=args.static_filter,
                        static_thresh=args.static_thresh,
                        static_min_frames=args.static_frames,
                    )

                    clip_data_path = os.path.join(os.path.dirname(clip_path), f"{clip_name}.json")
                    with open(clip_data_path, "r") as f:
                        clip_data = json.load(f)

                    split = clip_data["split"]
                    if split is None:
                        split = "unknown"

                    # 결과 저장
                    if keypoints.size > 0:
                        # 인원수 통계 업데이트 (1클립 1인 원칙 확인)
                        persons_count = keypoints.shape[0]
                        if persons_count <= 2:
                            persons_stats[persons_count] += 1
                        else:
                            persons_stats["3+"] += 1

                        # 클립 정보와 키포인트 저장
                        kp_file = os.path.join(
                            output_path, split, label, f"{clip_name}_keypoint.pkl"
                        )
                        label_idx = 0 if label == "fall" else 1

                        kp_data = {
                            "keypoint": keypoints,
                            "keypoint_score": keypoint_scores,
                            "frame_dir": clip_name,
                            "total_frames": len(frames),
                            "original_shape": frames[0].shape[:2],
                            "img_shape": frames[0].shape[:2],
                            "label": label_idx,
                            "label_name": label,
                        }

                        os.makedirs(os.path.dirname(kp_file), exist_ok=True)
                        with open(kp_file, "wb") as f:
                            pickle.dump(kp_data, f)

                        # 통계 업데이트
                        class_stats[label]["valid"] += 1
                        clip_stats["valid"] += 1

                    # 시각화 비디오 생성
                    if (
                        args.visualize
                        and temp_visualize_dir
                        and os.path.exists(temp_visualize_dir)
                    ):
                        vis_video_path = os.path.join(
                            output_path, "visualize", label, f"{clip_name}.mp4"
                        )
                        frame_files = sorted(
                            glob.glob(os.path.join(temp_visualize_dir, "*.jpg"))
                        )

                        if frame_files:
                            first_frame = cv2.imread(frame_files[0])
                            h, w = first_frame.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            fps = 30.0
                            video_writer = cv2.VideoWriter(
                                vis_video_path, fourcc, fps, (w, h)
                            )

                            for frame_file in frame_files:
                                frame = cv2.imread(frame_file)
                                video_writer.write(frame)

                            video_writer.release()

            except Exception as e:
                if clip_idx % args.log_interval == 0:
                    print(f"클립 처리 오류 ({os.path.basename(clip_path)}): {str(e)}")

    # 간략 통계 출력
    print("\n===== 처리 결과 요약 =====")
    print(f"총 처리 클립: {clip_stats['total']}개")
    print(
        f"유효 클립 수: {clip_stats['valid']}개 ({clip_stats['valid']/clip_stats['total']*100:.1f}%)"
    )

    # 클래스별 통계
    for label in labels:
        stats = class_stats[label]
        if stats["total"] > 0:
            valid_ratio = stats["valid"] / stats["total"] * 100
            print(
                f"[{label}] 유효 클립: {stats['valid']}/{stats['total']} ({valid_ratio:.1f}%)"
            )

    # 인원 수 통계 (1인당 1클립 데이터셋 특성 확인)
    print("\n===== 인식된 인원 통계 =====")
    print(
        f"0명 클립: {persons_stats[0]}개 ({persons_stats[0]/clip_stats['total']*100:.1f}%)"
    )
    print(
        f"1명 클립: {persons_stats[1]}개 ({persons_stats[1]/clip_stats['total']*100:.1f}%)"
    )
    print(
        f"2명 클립: {persons_stats[2]}개 ({persons_stats[2]/clip_stats['total']*100:.1f}%)"
    )
    print(
        f"3명 이상: {persons_stats['3+']}개 ({persons_stats['3+']/clip_stats['total']*100:.1f}%)"
    )

    # 1인당 1클립 원칙 확인
    expected_ratio = (
        persons_stats[1] / clip_stats["valid"] * 100 if clip_stats["valid"] > 0 else 0
    )
    print(f"\n클립당 1명 비율: {expected_ratio:.1f}% (정상적으로 처리된 클립 중)")


if __name__ == "__main__":
    convert_to_keypoints(args.clips_path, args.output_path)
