import argparse
import glob
import os
import pickle
import tempfile
import traceback
from collections import deque
from typing import Dict, List, Tuple, Union

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
parser.add_argument(
    "--device", type=str, default="cuda", help="GPU 사용 여부 (cuda or cpu)"
)
parser.add_argument(
    "--det-model",
    type=str,
    default="yolov8",
    choices=["yolov8", "faster_rcnn", "yolov11-pose"],
    help="모델 선택 (yolov8, faster_rcnn, yolov11-pose)",
)
parser.add_argument(
    "--bbox-scale", type=float, default=1.25, help="포즈 추정용 바운딩 박스 스케일"
)
parser.add_argument("--visualize", action="store_true", help="시각화 결과 저장")
parser.add_argument("--mode-48", action="store_true", help="384x288 모드 사용 여부")
parser.add_argument(
    "--smooth-keypoints",
    action="store_true",
    help="시간적 필터링을 통한 키포인트 안정화 활성화",
)
parser.add_argument(
    "--smooth-window",
    type=int,
    default=5,
    help="키포인트 스무딩 윈도우 크기 (프레임 수)",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
    help="배치 처리 크기 (1이면 비활성화, YOLOv11-Pose만 지원)",
)
parser.add_argument(
    "--buffer-size",
    type=int,
    default=16,
    help="프레임 버퍼 크기 (배치 처리 시 사용)",
)
parser.add_argument(
    "--static-filter",
    action="store_true",
    help="정적인 물체 필터링 활성화 (인체 모형 등 움직임이 적은 물체 제거)",
)
parser.add_argument(
    "--static-thresh",
    type=float,
    default=5.0,
    help="정적 물체 필터링 임계값 (픽셀 단위, 이 값보다 작은 움직임을 가진 객체 제거)",
)
parser.add_argument(
    "--static-frames",
    type=int,
    default=10,
    help="움직임 분석에 사용할 최소 프레임 수 (이 값보다 적은 프레임에 등장한 객체는 분석하지 않음)",
)
args = parser.parse_args()


def extract_frames(
    video_path: str, out_dir: str = None
) -> Tuple[List[str], List[np.ndarray]]:
    """
    비디오에서 프레임을 추출하는 함수

    Args:
        video_path: 비디오 경로
        out_dir: 프레임 저장 경로 (None이면 저장하지 않음)

    Returns:
        frame_paths: 저장된 프레임 경로 리스트
        frames: 추출된 프레임 리스트
    """
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
    frames: List[np.ndarray] = None,
    visualize: bool = False,
    visualize_dir: str = None,
    min_track_ratio: float = 0.8,  # 최소 트랙 출현 비율 (기본 80%)
    filter_static: bool = False,   # 정적 객체 필터링 활성화
    static_thresh: float = 5.0,    # 정적 객체 필터링 임계값 (픽셀 단위)
    static_min_frames: int = 10,   # 분석에 필요한 최소 프레임 수
) -> Tuple[np.ndarray, np.ndarray]:
    """
    포즈 추정 모델을 사용하여 모든 프레임을 처리

    Args:
        hrnet: HRNet 모델
        detector: 객체 검출 모델 또는 YOLOv11Pose 모델
        tracker: 추적기
        frames: 비디오 프레임 리스트
        visualize: 시각화 여부
        visualize_dir: 시각화 결과 저장 경로
        min_track_ratio: 저장할 트랙의 최소 출현 비율 (전체 프레임 중 해당 비율 이상 나타난 트랙만 저장)
        filter_static: 정적 객체 필터링 활성화 여부
        static_thresh: 정적 객체 필터링 임계값 (픽셀 단위)
        static_min_frames: 분석에 필요한 최소 프레임 수

    Returns:
        keypoints: 포즈 키포인트 배열 (사람 수, 프레임, 17, 2)
        keypoint_scores: 키포인트 점수 배열 (사람 수, 프레임, 17)
    """
    # 프레임별 검출 및 키포인트 결과 저장
    frame_results = []

    # 각 트랙 ID별 출현 프레임 수 카운트
    track_id_counts = {}
    
    # 움직임 분석을 위한 트랙 위치 저장
    track_positions = {}  # track_id -> list of (center_x, center_y) 튜플

    # 배치 처리 여부 확인
    use_batch = args.batch_size > 1 and isinstance(detector, YOLOv11Pose)
    
    if use_batch:
        print(f"배치 처리 활성화: 배치 크기 = {args.batch_size}, 버퍼 크기 = {args.buffer_size}")
        # 배치 처리를 위한 프레임 버퍼
        frame_buffer = []
        frame_indices = []
        batch_size = min(args.batch_size, args.buffer_size)
        
        # tqdm으로 진행 상황 표시
        pbar = tqdm.tqdm(range(len(frames)), desc="프레임 처리")
        
        # 프레임 배치 처리
        for frame_idx in pbar:
            # 버퍼에 프레임 추가
            frame_buffer.append(frames[frame_idx])
            frame_indices.append(frame_idx)
            
            # 버퍼가 가득 차거나 마지막 프레임이면 배치 처리
            if len(frame_buffer) >= batch_size or frame_idx == len(frames) - 1:
                try:
                    # 배치 추론 수행
                    batch_results = detector.process_batch(frame_buffer)
                    
                    # 배치 결과 처리
                    for i, (frame_idx, frame, results) in enumerate(zip(frame_indices, frame_buffer, batch_results)):
                        # 추적 수행 (YOLOv11-Pose에서는 바운딩 박스만 추출)
                        detections = results["bboxes"]
                        detections = tracker(detections)
                        
                        # 포즈 키포인트와 점수는 그대로 유지
                        results["bboxes"] = detections
                        
                        # 트랙 ID별 출현 횟수 업데이트
                        for bbox in results["bboxes"]:
                            track_id = bbox.track_id if bbox.track_id is not None else -1
                            if track_id not in track_id_counts:
                                track_id_counts[track_id] = 0
                            track_id_counts[track_id] += 1
                            
                            # 움직임 분석을 위한 위치 저장
                            if filter_static and track_id != -1:
                                # 바운딩 박스 중심점 계산
                                center_x = (bbox.x1 + bbox.x2) / 2
                                center_y = (bbox.y1 + bbox.y2) / 2
                                
                                if track_id not in track_positions:
                                    track_positions[track_id] = []
                                track_positions[track_id].append((center_x, center_y))
                        
                        # 결과 저장
                        frame_results.append(results)
                        
                        # 시각화 (필요한 경우)
                        if visualize and visualize_dir:
                            visualize_frame(frames[frame_idx], results, frame_idx, visualize_dir)
                    
                    # 현재 진행 상황 업데이트 (추적 중인 객체 수 표시)
                    pbar.set_postfix({"추적 객체 수": len(track_id_counts)})
                    
                except Exception as e:
                    print(f"배치 처리 오류: {traceback.format_exc()}")
                    # 오류가 발생하면 프레임별로 처리 (배치 처리 포기)
                    for frame_idx, frame in zip(frame_indices, frame_buffer):
                        # 개별 처리
                        try:
                            # 개별 추론
                            result = detector(frame)
                            detections = result["bboxes"]
                            detections = tracker(detections)
                            result["bboxes"] = detections
                            
                            # 트랙 ID별 출현 횟수 업데이트
                            for bbox in result["bboxes"]:
                                track_id = bbox.track_id if bbox.track_id is not None else -1
                                if track_id not in track_id_counts:
                                    track_id_counts[track_id] = 0
                                track_id_counts[track_id] += 1
                                
                                # 움직임 분석을 위한 위치 저장
                                if filter_static and track_id != -1:
                                    # 바운딩 박스 중심점 계산
                                    center_x = (bbox.x1 + bbox.x2) / 2
                                    center_y = (bbox.y1 + bbox.y2) / 2
                                    
                                    if track_id not in track_positions:
                                        track_positions[track_id] = []
                                    track_positions[track_id].append((center_x, center_y))
                            
                            # 결과 저장
                            frame_results.append(result)
                            
                            # 시각화 (필요한 경우)
                            if visualize and visualize_dir:
                                visualize_frame(frame, result, frame_idx, visualize_dir)
                                
                        except Exception as e:
                            print(f"프레임 {frame_idx} 처리 오류: {e}")
                            # 빈 결과 생성
                            frame_results.append({"keypoints": [], "scores": [], "bboxes": []})
                
                # 버퍼 초기화
                frame_buffer = []
                frame_indices = []
    else:
        # 기존 코드: 프레임별 처리
        # tqdm으로 진행 상황 표시
        pbar = tqdm.tqdm(frames, desc="프레임 처리")

        # 각 프레임 처리
        for frame_idx, frame in enumerate(pbar):
            try:
                # YOLOv11-Pose 직접 추론
                if isinstance(detector, YOLOv11Pose):
                    # YOLOv11-Pose로 직접 추론
                    results = detector(frame)
                    # 추적 처리
                    detections = results["bboxes"]
                    detections = tracker(detections)
                    results["bboxes"] = detections
                else:
                    # 기존 방식: 검출 + HRNet
                    # 검출 수행
                    detections = detector(frame)
                    # 추적 수행
                    detections = tracker(detections)
                    # 포즈 추정 (검출 결과를 이용)
                    results = hrnet(frame, detections)

                # 트랙 ID별 출현 횟수 업데이트
                for bbox in results["bboxes"]:
                    track_id = bbox.track_id if bbox.track_id is not None else -1
                    if track_id not in track_id_counts:
                        track_id_counts[track_id] = 0
                    track_id_counts[track_id] += 1
                    
                    # 움직임 분석을 위한 위치 저장
                    if filter_static and track_id != -1:
                        # 바운딩 박스 중심점 계산
                        center_x = (bbox.x1 + bbox.x2) / 2
                        center_y = (bbox.y1 + bbox.y2) / 2
                        
                        if track_id not in track_positions:
                            track_positions[track_id] = []
                        track_positions[track_id].append((center_x, center_y))

                # 현재 진행 상황 업데이트 (추적 중인 객체 수 표시)
                pbar.set_postfix({"추적 객체 수": len(track_id_counts)})

                # 결과 저장
                frame_results.append(results)

                # 시각화 (필요한 경우)
                if visualize and visualize_dir:
                    visualize_frame(frame, results, frame_idx, visualize_dir)
                    
            except Exception as e:
                print(f"프레임 {frame_idx} 처리 오류: {e}")
                # 빈 결과 생성
                frame_results.append({"keypoints": [], "scores": [], "bboxes": []})

    # 최소 출현 프레임 수 계산
    num_frames = len(frames)
    min_frames = int(num_frames * min_track_ratio)

    # 해당 기준을 만족하는 트랙 ID 필터링
    valid_track_ids = [
        track_id
        for track_id, count in track_id_counts.items()
        if count >= min_frames and track_id != -1  # -1은 유효한 트랙 ID가 아님
    ]
    
    # 정적 객체 필터링 (옵션이 활성화된 경우)
    if filter_static:
        static_track_ids = []
        moving_track_ids = []
        
        for track_id in valid_track_ids:
            positions = track_positions.get(track_id, [])
            
            # 충분한 프레임이 있는 경우에만 분석
            if len(positions) >= static_min_frames:
                # 움직임 분석: 위치 변화의 표준 편차 계산
                x_positions = [pos[0] for pos in positions]
                y_positions = [pos[1] for pos in positions]
                
                # 변화량 계산 방법 1: 표준 편차
                x_std = np.std(x_positions)
                y_std = np.std(y_positions)
                position_std = np.sqrt(x_std**2 + y_std**2)
                
                # 변화량 계산 방법 2: 최대 이동 거리
                max_distance = 0
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        dx = positions[i][0] - positions[j][0]
                        dy = positions[i][1] - positions[j][1]
                        distance = np.sqrt(dx**2 + dy**2)
                        max_distance = max(max_distance, distance)
                
                # 디버깅 정보 출력
                print(f"트랙 ID {track_id}: 표준편차={position_std:.2f}px, 최대거리={max_distance:.2f}px")
                
                # 임계값보다 작은 움직임을 가진 객체는 정적으로 간주
                if position_std < static_thresh:
                    static_track_ids.append(track_id)
                else:
                    moving_track_ids.append(track_id)
            else:
                # 프레임이 충분하지 않은 경우 일단 포함
                moving_track_ids.append(track_id)
        
        # 필터링 결과 출력
        if static_track_ids:
            print(f"정적 객체 필터링: {len(static_track_ids)}개 객체 제외 (움직임 < {static_thresh}px)")
            for track_id in static_track_ids:
                positions = track_positions.get(track_id, [])
                if positions:
                    x_std = np.std([pos[0] for pos in positions])
                    y_std = np.std([pos[1] for pos in positions])
                    position_std = np.sqrt(x_std**2 + y_std**2)
                    print(f"  - ID {track_id}: 움직임={position_std:.2f}px (프레임 수={len(positions)})")
        
        # 필터링된 결과로 유효한 트랙 ID 업데이트
        valid_track_ids = moving_track_ids
    
    print(
        f"총 {len(track_id_counts)} 객체 중 {len(valid_track_ids)}개 객체가 최종 선택됨 (출현 비율 {min_track_ratio*100:.0f}% 이상 & 충분한 움직임)"
    )

    # 추적 ID를 기준으로 결과 정렬
    trackid_to_idx = {track_id: idx for idx, track_id in enumerate(valid_track_ids)}

    # 유효한 트랙 ID가 없으면 빈 결과 반환
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
        for i, (kps, scores, bbox) in enumerate(
            zip(
                frame_result["keypoints"],
                frame_result["scores"],
                frame_result["bboxes"],
            )
        ):
            track_id = bbox.track_id if bbox.track_id is not None else -1

            # 유효한 트랙 ID만 처리
            if track_id in trackid_to_idx:
                person_idx = trackid_to_idx[track_id]
                keypoints[person_idx, frame_idx] = kps
                keypoint_scores[person_idx, frame_idx] = scores

    return keypoints, keypoint_scores


def visualize_frame(frame, results, frame_idx, visualize_dir):
    """
    프레임에 포즈 추정 결과를 시각화

    Args:
        frame: 원본 프레임
        results: 포즈 추정 결과
        frame_idx: 프레임 인덱스
        visualize_dir: 시각화 결과 저장 경로
    """
    vis_frame = frame.copy()
    keypoints_list = results["keypoints"]
    scores_list = results["scores"]
    bboxes = results["bboxes"]

    # COCO 키포인트 연결 (시각화용)
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

    # 키포인트 색상 (시각화용)
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
    for i, (keypoints, scores, bbox) in enumerate(
        zip(keypoints_list, scores_list, bboxes)
    ):
        # 바운딩 박스 그리기
        cv2.rectangle(
            vis_frame,
            (int(bbox.x1), int(bbox.y1)),
            (int(bbox.x2), int(bbox.y2)),
            (0, 255, 0),
            2,
        )

        # ID 표시
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

        # 키포인트 그리기
        for k_idx, (kp, score) in enumerate(zip(keypoints, scores)):
            if score > 0.3:  # 신뢰도 임계값
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(vis_frame, (x, y), 3, colors[k_idx], -1)

        # 스켈레톤 연결선 그리기
        for connection in connections:
            if scores[connection[0]] > 0.3 and scores[connection[1]] > 0.3:
                pt1 = tuple(map(int, keypoints[connection[0]]))
                pt2 = tuple(map(int, keypoints[connection[1]]))
                cv2.line(vis_frame, pt1, pt2, (0, 255, 255), 2)

    # 프레임 저장
    os.makedirs(visualize_dir, exist_ok=True)
    cv2.imwrite(os.path.join(visualize_dir, f"frame_{frame_idx:06d}.jpg"), vis_frame)


def convert_to_keypoints(clips_path: str, output_path: str):
    """
    비디오 클립을 처리하여 키포인트를 추출하고 mmaction2 형식으로 저장
    항상 top-down 방식을 사용

    Args:
        clips_path: 클립 경로
        output_path: 결과 저장 경로
    """
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
        enable_smoothing=args.smooth_keypoints,  # 스무딩 활성화 여부
        smooth_window_size=args.smooth_window,  # 스무딩 윈도우 크기
    )

    # YOLOv11-Pose 모델 초기화
    yolov11_pose = None
    detector = None
    
    # 모델 선택에 따라 적절한 모델 로드
    if args.det_model == "yolov11-pose":
        yolov11_pose = YOLOv11Pose(
            onnx_path="models/onnx/yolo11m-pose.onnx",
            cuda=use_cuda,
            person_class_id=0,
            score_threshold=args.det_score_thr,
            iou_threshold=0.45,
            batch_size=args.batch_size,
            enable_smoothing=args.smooth_keypoints,
            smooth_window_size=args.smooth_window,
        )
        detector = yolov11_pose  # 이후 코드에서 detector 변수를 사용하므로 통일
        person_class_id = 0
    # Faster R-CNN 또는 YOLOv8 초기화
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
        target_classes=[person_class_id],  # 사람 클래스만
        match_iou_thresh=0.3,
        max_lost=30,
    )

    # 출력 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "fall"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "nofall"), exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(output_path, "visualize", "fall"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "visualize", "nofall"), exist_ok=True)

    # 라벨링 클래스 (디렉토리 기반)
    labels = ["fall", "nofall"]

    # 각 클래스별 처리를 위한 루프
    clip_stats = {}
    total_clips = 0

    for label in labels:
        # 해당 라벨 디렉터리의 모든 클립 처리
        class_dir = os.path.join(clips_path, label)
        if not os.path.exists(class_dir):
            print(f"경고: {class_dir} 디렉토리가 존재하지 않습니다.")
            continue

        # 라벨 디렉토리 내 모든 클립 경로 수집
        clip_paths = glob.glob(os.path.join(class_dir, "*.mp4"))
        print(f"{label} 클래스: {len(clip_paths)}개의 클립 발견")
        total_clips += len(clip_paths)

        # tqdm으로 클립 진행상황 표시
        for clip_idx, clip_path in enumerate(
            tqdm.tqdm(clip_paths, desc=f"{label} 클립 처리")
        ):
            try:
                clip_name = os.path.splitext(os.path.basename(clip_path))[0]

                # 현재 처리 중인 클립 정보 표시
                print(f"\n[{clip_idx+1}/{len(clip_paths)}] {label}/{clip_name} 처리 중")

                # 비디오 프레임 추출
                with tempfile.TemporaryDirectory() as tmp_dir:
                    frame_paths, frames = extract_frames(clip_path, tmp_dir)

                    if len(frames) == 0:
                        print(f"경고: {clip_path}에서 프레임을 추출할 수 없습니다.")
                        continue

                    # 추적기 초기화 (새 클립마다)
                    tracker.reset()
                    # 스무딩 버퍼 초기화 (새 클립마다)
                    if args.smooth_keypoints:
                        hrnet.clear_smoothing_buffers()

                    # 시각화를 위한 임시 디렉토리 생성 (필요한 경우)
                    temp_visualize_dir = None
                    if args.visualize:
                        temp_visualize_dir = os.path.join(tmp_dir, "visualize")
                        os.makedirs(temp_visualize_dir, exist_ok=True)

                    # 포즈 추정 (top-down 방식)
                    keypoints, keypoint_scores = process_frames_with_pose_estimation(
                        hrnet=hrnet,
                        detector=detector,
                        tracker=tracker,
                        frames=frames,
                        visualize=args.visualize,
                        visualize_dir=temp_visualize_dir,  # 임시 디렉토리 사용
                        min_track_ratio=0.8,  # 전체 프레임의 80% 이상 나타난 객체만 처리
                        filter_static=args.static_filter,  # 정적 객체 필터링
                        static_thresh=args.static_thresh,  # 정적 객체 필터링 임계값
                        static_min_frames=args.static_frames,  # 분석에 필요한 최소 프레임 수
                    )

                    # 적어도 하나의 객체가 있는 경우에만 저장
                    if keypoints.size > 0:
                        # 라벨별로 디렉토리 구분하여 저장
                        kp_file = os.path.join(
                            output_path, label, f"{clip_name}_keypoint.pkl"
                        )

                        # 라벨 정보 추가
                        label_idx = 0 if label == "fall" else 1  # fall: 0, nofall: 1

                        kp_data = {
                            "keypoint": keypoints,
                            "keypoint_score": keypoint_scores,
                            "frame_dir": clip_name,
                            "total_frames": len(frames),
                            "original_shape": frames[0].shape[:2],
                            "img_shape": frames[0].shape[:2],
                            "label": label_idx,  # 라벨 정보 추가
                            "label_name": label,  # 라벨 이름도 추가
                        }

                        os.makedirs(os.path.dirname(kp_file), exist_ok=True)
                        with open(kp_file, "wb") as f:
                            pickle.dump(kp_data, f)

                        # 클립 통계 저장
                        clip_stats[f"{label}/{clip_name}"] = {
                            "persons": keypoints.shape[0],
                            "frames": len(frames),
                            "label": label,
                        }

                        print(
                            f"{label}/{clip_name} 처리 완료 - {keypoints.shape[0]}명 저장됨 (전체 {len(frames)}프레임)"
                        )
                    else:
                        print(
                            f"{label}/{clip_name} 처리 완료 - 저장된 객체 없음 (기준: 전체 프레임의 80% 이상 출현)"
                        )
                        # 클립 통계 저장
                        clip_stats[f"{label}/{clip_name}"] = {
                            "persons": 0,
                            "frames": len(frames),
                            "label": label,
                        }

                    # 시각화를 비디오로 변환 (시각화가 켜져 있는 경우)
                    if (
                        args.visualize
                        and temp_visualize_dir
                        and os.path.exists(temp_visualize_dir)
                    ):
                        vis_video_path = os.path.join(
                            output_path, "visualize", label, f"{clip_name}.mp4"
                        )

                        # 프레임을 비디오로 변환
                        frame_files = sorted(
                            glob.glob(os.path.join(temp_visualize_dir, "*.jpg"))
                        )
                        if frame_files:
                            first_frame = cv2.imread(frame_files[0])
                            h, w = first_frame.shape[:2]

                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            fps = 30.0  # 일반적인 FPS

                            video_writer = cv2.VideoWriter(
                                vis_video_path, fourcc, fps, (w, h)
                            )

                            for frame_file in frame_files:
                                frame = cv2.imread(frame_file)
                                video_writer.write(frame)

                            video_writer.release()
                            print(f"시각화 비디오 저장: {vis_video_path}")

                    # 임시 디렉토리는 with 블록이 끝나면 자동으로 삭제됨

            except Exception as e:
                print(f"클립 처리 오류 ({clip_path}): {e}")
                # 오류 발생 시에도 통계에 기록
                clip_name = os.path.splitext(os.path.basename(clip_path))[0]
                clip_stats[f"{label}/{clip_name}"] = {
                    "persons": 0,
                    "frames": 0,
                    "error": str(e),
                    "label": label,
                }

    # 전체 처리 결과 통계 출력
    total_persons = sum(stats["persons"] for stats in clip_stats.values())
    valid_clips = sum(1 for stats in clip_stats.values() if stats["persons"] > 0)

    # 각 클래스별 통계
    class_stats = {}
    for label in labels:
        class_clips = sum(
            1 for key, stats in clip_stats.items() if stats["label"] == label
        )
        class_valid = sum(
            1
            for key, stats in clip_stats.items()
            if stats["label"] == label and stats["persons"] > 0
        )
        class_persons = sum(
            stats["persons"]
            for key, stats in clip_stats.items()
            if stats["label"] == label
        )

        class_stats[label] = {
            "total_clips": class_clips,
            "valid_clips": class_valid,
            "total_persons": class_persons,
            "avg_persons": class_persons / class_valid if class_valid > 0 else 0,
        }

    print("\n===== 처리 결과 요약 =====")
    print(f"총 처리 클립: {total_clips}개")
    print(f"유효 클립 수: {valid_clips}개 (사람이 1명 이상 감지된 클립)")
    print(f"총 저장된 사람 수: {total_persons}명")
    print(f"클립당 평균 사람 수: {total_persons/valid_clips:.2f}명 (유효 클립 기준)")

    # 클래스별 통계 출력
    for label in labels:
        stats = class_stats[label]
        print(f"\n=== {label} 클래스 통계 ===")
        print(f"총 클립 수: {stats['total_clips']}개")
        print(f"유효 클립 수: {stats['valid_clips']}개 (사람이 1명 이상 감지된 클립)")
        print(f"총 저장된 사람 수: {stats['total_persons']}명")
        print(f"클립당 평균 사람 수: {stats['avg_persons']:.2f}명 (유효 클립 기준)")

    print("=========================")

    # 클립별 상세 정보
    print("\n클립별 상세 정보:")
    for clip_name, stats in clip_stats.items():
        if "error" in stats:
            status = "❌ 오류"
        elif stats["persons"] == 0:
            status = "⚠️ 사람 없음"
        else:
            status = f"✅ {stats['persons']}명"

        print(f"  - {clip_name}: {status} ({stats['frames']}프레임)")


if __name__ == "__main__":
    convert_to_keypoints(args.clips_path, args.output_path)
