import argparse
import os
import time
from collections import deque
import traceback

import cv2

from models import YOLOv11Pose, FasterRCNN, HRNet, YOLOv8
from trackers import ByteTrackTracker

parser = argparse.ArgumentParser()
parser.add_argument("--video-path", type=str, default="assets/demo_1.mp4")
parser.add_argument(
    "--keypoint-thresh", type=float, default=0.3, help="키포인트 표시 임계값"
)
parser.add_argument(
    "--bbox-scale", type=float, default=1.25, help="바운딩 박스 확대 비율"
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="비디오 저장 경로 (지정 안하면 화면에 표시)",
)
parser.add_argument(
    "--det-model",
    type=str,
    default="yolov8",
    choices=["yolov8", "faster_rcnn", "yolov11-pose"],
    help="모델 선택 (yolov8, faster_rcnn 또는 yolov11-pose)",
)
parser.add_argument(
    "--det-score-thresh",
    type=float,
    default=0.5,
    help="객체 검출 신뢰도 임계값",
)
parser.add_argument(
    "--min-box-area", type=int, default=1000, help="최소 바운딩 박스 면적"
)
parser.add_argument("--mode-48", action="store_true", help="384x288 모드 사용 여부")
parser.add_argument(
    "--pose-mode",
    type=str,
    default="top-down",
    choices=["top-down", "bottom-up"],
    help="포즈 추정 모드 (top-down 또는 bottom-up)",
)
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
    default=4,
    help="프레임 버퍼 크기 (배치 처리 시 사용)",
)
args = parser.parse_args()


try:
    print("모델 로딩 중...")

    # HRNet 포즈 추정 모델 초기화
    hrnet = HRNet(
        onnx_path=(
            "models/onnx/hrnet_48.onnx" if args.mode_48 else "models/onnx/hrnet_32.onnx"
        ),
        cuda=True,
        mode_48=args.mode_48,
        pose_bbox_scale=args.bbox_scale,
        enable_smoothing=args.smooth_keypoints,
        smooth_window_size=args.smooth_window,
    )

    # YOLOv11-Pose 모델 초기화
    yolov11_pose = None
    if args.det_model == "yolov11-pose":
        yolov11_pose = YOLOv11Pose(
            onnx_path="models/onnx/yolo11m-pose.onnx",
            cuda=True,
            person_class_id=0,
            score_threshold=None,
            batch_size=args.batch_size,
            enable_smoothing=args.smooth_keypoints,
            smooth_window_size=args.smooth_window,
        )

    # Top-Down 모드에서만 검출기와 추적기 초기화
    detector = None
    tracker = None
    if args.pose_mode == "top-down" and args.det_model != "yolov11-pose":
        # Faster R-CNN 초기화

        if args.det_model == "yolov8":
            person_class_id = 0

            detector = YOLOv8(
                onnx_path="models/onnx/yolov8x.onnx",
                cuda=True,
                person_class_id=person_class_id,
                score_threshold=None,
            )
        elif args.det_model == "faster_rcnn":
            person_class_id = 1

            detector = FasterRCNN(
                onnx_path="models/onnx/faster_rcnn.onnx",
                cuda=True,
                person_class_id=person_class_id,
                score_threshold=None,
            )
        else:
            raise ValueError(f"Invalid detection model: {args.det_model}")

        # 추적기 초기화 (사람 클래스만 추적)
        tracker = ByteTrackTracker(
            high_thresh=0.6,
            low_thresh=0.2,
            match_iou_thresh=0.3,
            max_lost=30,
            target_classes=[person_class_id],  # 사람 클래스만 추적 (모델에 따라 다름)
        )

    print(f"모델 로딩 완료 ({args.pose_mode} 모드, {args.det_model} 모델)")
except Exception as e:
    print(f"모델 로딩 오류: {e}")
    exit(1)

DISPLAY_MAX_W = 1280

# COCO 키포인트 연결 정의
SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # 얼굴 및 어깨
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # 팔
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),  # 몸통
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # 다리
]

# 키포인트 색상 정의
KEYPOINT_COLORS = [
    (255, 0, 0),  # 코
    (255, 85, 0),  # 왼쪽 눈
    (255, 170, 0),  # 오른쪽 눈
    (255, 255, 0),  # 왼쪽 귀
    (170, 255, 0),  # 오른쪽 귀
    (85, 255, 0),  # 왼쪽 어깨
    (0, 255, 0),  # 오른쪽 어깨
    (0, 255, 85),  # 왼쪽 팔꿈치
    (0, 255, 170),  # 오른쪽 팔꿈치
    (0, 255, 255),  # 왼쪽 손목
    (0, 170, 255),  # 오른쪽 손목
    (0, 85, 255),  # 왼쪽 골반
    (0, 0, 255),  # 오른쪽 골반
    (85, 0, 255),  # 왼쪽 무릎
    (170, 0, 255),  # 오른쪽 무릎
    (255, 0, 255),  # 왼쪽 발목
    (255, 0, 170),  # 오른쪽 발목
]

# 스켈레톤 라인 색상
LIMB_COLORS = [
    (255, 51, 51),
    (255, 51, 153),
    (255, 51, 255),
    (153, 51, 255),
    (51, 51, 255),
    (51, 153, 255),
    (51, 255, 255),
    (51, 255, 153),
    (51, 255, 51),
    (153, 255, 51),
    (255, 255, 51),
    (255, 153, 51),
    (255, 102, 102),
    (192, 102, 255),
    (102, 102, 255),
    (102, 192, 255),
]


def draw_keypoints(image, keypoints, scores, threshold=0.3):
    """
    이미지에 키포인트와 스켈레톤 시각화

    Args:
        image: 시각화할 이미지
        keypoints: 키포인트 좌표 (17, 2)
        scores: 키포인트 신뢰도 점수 (17,)
        threshold: 키포인트 표시 임계값
    """
    try:
        if keypoints is None or scores is None:
            return

        # 키포인트 그리기
        for i, (x, y) in enumerate(keypoints):
            if scores[i] > threshold:
                cv2.circle(image, (int(x), int(y)), 4, KEYPOINT_COLORS[i], -1)

        # 스켈레톤 그리기
        for i, (p1_idx, p2_idx) in enumerate(SKELETON):
            if p1_idx >= len(keypoints) or p2_idx >= len(keypoints):
                continue

            x1, y1 = keypoints[p1_idx]
            x2, y2 = keypoints[p2_idx]

            # 두 키포인트가 모두 임계값 이상인 경우에만 선 그리기
            if scores[p1_idx] > threshold and scores[p2_idx] > threshold:
                cv2.line(
                    image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    LIMB_COLORS[i % len(LIMB_COLORS)],
                    2,
                )
    except Exception as e:
        print(f"키포인트 그리기 오류: {e}")


def visualize_inference(video_path: str, output_path: str = None):
    if not os.path.exists(video_path):
        print(f"비디오 파일이 존재하지 않습니다: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: 비디오 파일을 열 수 없습니다: {video_path}")
        return

    # 비디오 속성 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 출력 비디오 설정
    video_writer = None
    if output_path:
        # 출력 디렉토리 확인
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # VideoWriter 초기화
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, (frame_width, frame_height)
        )

        print(f"비디오를 저장합니다: {output_path}")
        print(f"총 프레임 수: {total_frames}, FPS: {fps}")

    # 프레임 카운터
    frame_count = 0
    start_time = time.time()
    processing_fps = 0

    # 키포인트의 이전 위치를 저장하는 버퍼
    keypoints_buffers = {}  # track_id를 키로 사용하는 딕셔너리
    
    # 배치 처리용 프레임 버퍼
    frame_buffer = []
    frame_positions = []  # 프레임 위치 저장
    
    # 배치 처리 사용 여부
    use_batch = args.batch_size > 1 and args.det_model == "yolov11-pose"
    if use_batch:
        print(f"배치 추론 활성화: 배치 크기 = {args.batch_size}, 버퍼 크기 = {args.buffer_size}")

    while cap.isOpened():
        # 프레임 처리 시작 시간
        frame_start_time = time.time()
        
        # 배치 처리를 위한 프레임 버퍼 채우기
        if use_batch:
            # 다음 배치만큼 프레임 읽기
            while len(frame_buffer) < args.buffer_size:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                frame_buffer.append(frame)
                frame_positions.append(frame_count)
            
            # 버퍼가 비었으면 종료
            if not frame_buffer:
                break
            
            # 배치 추론 수행
            try:
                batch_results = yolov11_pose.process_batch(frame_buffer)
                
                # 결과 처리
                for i, (frame, position, results) in enumerate(zip(frame_buffer, frame_positions, batch_results)):
                    orig_h, orig_w = frame.shape[:2]
                    
                    # 결과 시각화
                    keypoints_list = results["keypoints"]
                    scores_list = results["scores"]
                    bboxes = results["bboxes"]
                    
                    # 결과 필터링 (너무 작은 박스, 종횡비 등)
                    filtered_results = {"keypoints": [], "scores": [], "bboxes": []}

                    for j, (keypoints, scores, bbox) in enumerate(
                        zip(keypoints_list, scores_list, bboxes)
                    ):
                        try:
                            # 신뢰도 너무 낮은 것은 제외
                            if bbox.score < args.det_score_thresh:
                                continue

                            # 박스 크기가 너무 작은 것은 제외 (픽셀 기준)
                            box_w = max(0, bbox.x2 - bbox.x1)
                            box_h = max(0, bbox.y2 - bbox.y1)
                            if box_w * box_h < args.min_box_area:  # 최소 박스 면적
                                continue

                            # 종횡비가 비정상적인 것 제외
                            aspect_ratio = box_w / box_h if box_h > 0 else 0
                            if aspect_ratio < 0.2:
                                continue

                            filtered_results["keypoints"].append(keypoints)
                            filtered_results["scores"].append(scores)
                            filtered_results["bboxes"].append(bbox)
                        except Exception as e:
                            print(f"바운딩 박스 필터링 오류: {e}")
                            continue

                    # 필터링된 결과로 업데이트
                    keypoints_list = filtered_results["keypoints"]
                    scores_list = filtered_results["scores"]
                    bboxes = filtered_results["bboxes"]
                    
                    # 결과 시각화
                    for j, (keypoints, scores, bbox) in enumerate(
                        zip(keypoints_list, scores_list, bboxes)
                    ):
                        try:
                            # 바운딩 박스 그리기
                            x1 = max(0, int(bbox.x1))
                            y1 = max(0, int(bbox.y1))
                            x2 = min(orig_w - 1, int(bbox.x2))
                            y2 = min(orig_h - 1, int(bbox.y2))

                            # 유효한 박스 확인
                            if x2 <= x1 or y2 <= y1:
                                continue

                            # 박스 색상: 신뢰도에 따라 색상 변경 (높을수록 빨간색에 가까움)
                            color_scale = min(1.0, bbox.score)
                            color = (0, int(255 * (1 - color_scale)), int(255 * color_scale))

                            # 바운딩 박스 그리기
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            # 트랙 ID와 신뢰도 텍스트 표시
                            text = (
                                f"ID:{bbox.track_id} {bbox.score:.2f}"
                                if bbox.track_id is not None
                                else f"{bbox.score:.2f}"
                            )
                            cv2.putText(
                                frame,
                                text,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
                            )

                            if args.smooth_keypoints and bbox.track_id is not None:
                                # 각 객체별로 버퍼 관리
                                track_id = bbox.track_id
                                if track_id not in keypoints_buffers:
                                    keypoints_buffers[track_id] = deque(
                                        maxlen=args.smooth_window
                                    )

                                # 키포인트 스무딩 적용
                                smoothed_keypoints = keypoints.copy()
                                buffer = keypoints_buffers[track_id]

                                if buffer:  # 버퍼에 이전 프레임 데이터가 있으면
                                    for k in range(len(keypoints)):
                                        if scores[k] < 0.3:  # 낮은 신뢰도는 필터링 안함
                                            continue

                                        weight_sum = scores[k]
                                        weighted_pos = keypoints[k] * scores[k]

                                        # 이전 프레임 키포인트 가중 평균
                                        for prev_keypoints, prev_scores in buffer:
                                            if prev_scores[k] > 0.2:
                                                weight = prev_scores[k]
                                                weighted_pos += prev_keypoints[k] * weight
                                                weight_sum += weight

                                        if weight_sum > 0:
                                            smoothed_keypoints[k] = weighted_pos / weight_sum

                                # 현재 키포인트 저장
                                buffer.append((keypoints.copy(), scores.copy()))

                                # 스무딩된 키포인트 그리기
                                draw_keypoints(
                                    frame, smoothed_keypoints, scores, args.keypoint_thresh
                                )
                            else:
                                # 스무딩 없이 키포인트 그리기
                                draw_keypoints(frame, keypoints, scores, args.keypoint_thresh)

                        except Exception as e:
                            print(f"사람 처리 오류: {e}")
                            continue
                    
                    # 현재 프레임 정보 표시
                    cv2.putText(
                        frame,
                        f"Frame: {position}/{total_frames} | FPS: {processing_fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )

                    # 경과 시간 표시
                    elapsed_time = time.time() - start_time
                    smooth_text = "Smooth ON" if args.smooth_keypoints else "Smooth OFF"
                    batch_text = f"Batch {args.batch_size}" if use_batch else "Single"
                    cv2.putText(
                        frame,
                        f"Time: {elapsed_time:.1f}s | Mode: {args.pose_mode} | {smooth_text} | {batch_text}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                    
                    # 화면 또는 비디오에 출력
                    if video_writer:
                        video_writer.write(frame)
                    else:
                        # 디스플레이용 프레임 크기 조정
                        if orig_w > DISPLAY_MAX_W:
                            scale = DISPLAY_MAX_W / orig_w
                            new_size = (int(orig_w * scale), int(orig_h * scale))
                            frame_disp = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
                        else:
                            frame_disp = frame

                        cv2.imshow(f"Pose Estimation ({args.pose_mode})", frame_disp)

                        # ESC 또는 q 키로 종료
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:  # ESC
                            break
                
                # 진행 상황 표시
                if video_writer and (frame_count % 10 == 0 or frame_count >= total_frames):
                    progress = (frame_count / total_frames) * 100
                    print(
                        f"\r진행 상황: {progress:.1f}% ({frame_count}/{total_frames}) | FPS: {processing_fps:.1f}",
                        end="",
                    )
                
                # 배치 처리 시간 계산
                batch_time = time.time() - frame_start_time
                processing_fps = len(frame_buffer) / batch_time if batch_time > 0 else 0
                
                # 처리된 프레임 버퍼 비우기
                frame_buffer = []
                frame_positions = []
                
            except Exception as e:
                print(f"배치 처리 오류: {traceback.format_exc()}")
                # 오류 발생 시 배치 처리를 중단하고 단일 프레임 처리로 전환
                frame_buffer = []
                frame_positions = []
                use_batch = False
                
        else:
            # 단일 프레임 처리 (기존 코드)
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            orig_h, orig_w = frame.shape[:2]

            try:
                # YOLOv11-Pose 모델 사용
                if args.det_model == "yolov11-pose" and yolov11_pose is not None:
                    # YOLOv11-Pose로 직접 추론
                    results = yolov11_pose(frame)
                # Top-Down 모드
                elif args.pose_mode == "top-down" and detector is not None:
                    # 객체 검출 (Faster R-CNN 또는 YOLOv8)
                    detections = detector(frame)

                    # 추적기가 있으면 추적 수행
                    if tracker is not None:
                        detections = tracker(detections)

                    # 포즈 추정 (HRNet with detections)
                    results = hrnet(frame, detections)
                # Bottom-Up 모드
                else:
                    # 포즈 추정 (HRNet without detections)
                    results = hrnet(frame)

                keypoints_list = results["keypoints"]
                scores_list = results["scores"]
                bboxes = results["bboxes"]

                # 결과 필터링 (너무 작은 박스, 종횡비 등)
                filtered_results = {"keypoints": [], "scores": [], "bboxes": []}

                for i, (keypoints, scores, bbox) in enumerate(
                    zip(keypoints_list, scores_list, bboxes)
                ):
                    try:
                        # 신뢰도 너무 낮은 것은 제외
                        if bbox.score < args.det_score_thresh:
                            continue

                        # 박스 크기가 너무 작은 것은 제외 (픽셀 기준)
                        box_w = max(0, bbox.x2 - bbox.x1)
                        box_h = max(0, bbox.y2 - bbox.y1)
                        if box_w * box_h < args.min_box_area:  # 최소 박스 면적
                            continue

                        # 종횡비가 비정상적인 것 제외
                        aspect_ratio = box_w / box_h if box_h > 0 else 0
                        if aspect_ratio < 0.2:
                            continue

                        filtered_results["keypoints"].append(keypoints)
                        filtered_results["scores"].append(scores)
                        filtered_results["bboxes"].append(bbox)
                    except Exception as e:
                        print(f"바운딩 박스 필터링 오류: {e}")
                        continue

                # 필터링된 결과로 업데이트
                keypoints_list = filtered_results["keypoints"]
                scores_list = filtered_results["scores"]
                bboxes = filtered_results["bboxes"]

                # 결과 시각화
                for i, (keypoints, scores, bbox) in enumerate(
                    zip(keypoints_list, scores_list, bboxes)
                ):
                    try:
                        # 바운딩 박스 그리기
                        x1 = max(0, int(bbox.x1))
                        y1 = max(0, int(bbox.y1))
                        x2 = min(orig_w - 1, int(bbox.x2))
                        y2 = min(orig_h - 1, int(bbox.y2))

                        # 유효한 박스 확인
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # 박스 색상: 신뢰도에 따라 색상 변경 (높을수록 빨간색에 가까움)
                        color_scale = min(1.0, bbox.score)
                        color = (0, int(255 * (1 - color_scale)), int(255 * color_scale))

                        # 바운딩 박스 그리기
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # 트랙 ID와 신뢰도 텍스트 표시
                        text = (
                            f"ID:{bbox.track_id} {bbox.score:.2f}"
                            if bbox.track_id is not None
                            else f"{bbox.score:.2f}"
                        )
                        cv2.putText(
                            frame,
                            text,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                        if args.smooth_keypoints and bbox.track_id is not None:
                            # 각 객체별로 버퍼 관리
                            track_id = bbox.track_id
                            if track_id not in keypoints_buffers:
                                keypoints_buffers[track_id] = deque(
                                    maxlen=args.smooth_window
                                )

                            # 키포인트 스무딩 적용
                            smoothed_keypoints = keypoints.copy()
                            buffer = keypoints_buffers[track_id]

                            if buffer:  # 버퍼에 이전 프레임 데이터가 있으면
                                for j in range(len(keypoints)):
                                    if scores[j] < 0.3:  # 낮은 신뢰도는 필터링 안함
                                        continue

                                    weight_sum = scores[j]
                                    weighted_pos = keypoints[j] * scores[j]

                                    # 이전 프레임 키포인트 가중 평균
                                    for prev_keypoints, prev_scores in buffer:
                                        if prev_scores[j] > 0.2:
                                            weight = prev_scores[j]
                                            weighted_pos += prev_keypoints[j] * weight
                                            weight_sum += weight

                                    if weight_sum > 0:
                                        smoothed_keypoints[j] = weighted_pos / weight_sum

                            # 현재 키포인트 저장
                            buffer.append((keypoints.copy(), scores.copy()))

                            # 스무딩된 키포인트 그리기
                            draw_keypoints(
                                frame, smoothed_keypoints, scores, args.keypoint_thresh
                            )
                        else:
                            # 스무딩 없이 키포인트 그리기
                            draw_keypoints(frame, keypoints, scores, args.keypoint_thresh)

                    except Exception as e:
                        print(f"사람 처리 오류: {e}")
                        continue

                # 처리 FPS 계산
                frame_time = time.time() - frame_start_time
                processing_fps = 1.0 / frame_time if frame_time > 0 else 0

            except Exception as e:
                print(f"프레임 {frame_count} 처리 오류: {traceback.format_exc()}")

            # 현재 프레임 정보 표시
            cv2.putText(
                frame,
                f"Frame: {frame_count}/{total_frames} | FPS: {processing_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            # 경과 시간 표시
            elapsed_time = time.time() - start_time
            smooth_text = "Smooth ON" if args.smooth_keypoints else "Smooth OFF"
            cv2.putText(
                frame,
                f"Time: {elapsed_time:.1f}s | Mode: {args.pose_mode} | {smooth_text}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            # 화면 또는 비디오에 출력
            if video_writer:
                video_writer.write(frame)

                # 진행 상황 표시
                if frame_count % 10 == 0 or frame_count == total_frames:
                    progress = (frame_count / total_frames) * 100
                    print(
                        f"\r진행 상황: {progress:.1f}% ({frame_count}/{total_frames}) | FPS: {processing_fps:.1f}",
                        end="",
                    )
            else:
                # 디스플레이용 프레임 크기 조정
                if orig_w > DISPLAY_MAX_W:
                    scale = DISPLAY_MAX_W / orig_w
                    new_size = (int(orig_w * scale), int(orig_h * scale))
                    frame_disp = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
                else:
                    frame_disp = frame

                cv2.imshow(f"Pose Estimation ({args.pose_mode})", frame_disp)

                # ESC 또는 q 키로 종료
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # ESC
                    break

    # 자원 해제
    cap.release()
    if video_writer:
        video_writer.release()
        print("\n처리 완료!")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_inference(args.video_path, args.output)
