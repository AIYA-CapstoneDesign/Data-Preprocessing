from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime
from collections import deque

from utils import BBox


class HRNet:
    """
    HRNet 포즈 추정 모델
    """

    def __init__(
        self,
        onnx_path: str,
        cuda: bool = False,
        mode_48: bool = False,
        pose_bbox_scale: float = 1.25,
        enable_smoothing: bool = False,
        smooth_window_size: int = 5,
    ):
        """
        HRNet 모델 초기화

        Args:
            onnx_path: ONNX 모델 경로
            cuda: GPU 사용 여부
            mode_48: 384x288 모드 사용 여부
            pose_bbox_scale: 포즈 추정을 위한 바운딩 박스 확장 비율 (top-down 모드에서 사용)
            enable_smoothing: 키포인트 시간적 스무딩 활성화 여부
            smooth_window_size: 스무딩 윈도우 크기
        """
        self.onnx_path = onnx_path
        self.use_cuda = cuda
        self.input_size = (
            (384, 288) if mode_48 else (256, 192)
        )  # (height, width) for HRNet
        self.num_keypoints = 17  # COCO 키포인트 수
        self.pose_bbox_scale = pose_bbox_scale

        # 시간적 스무딩 관련 설정
        self.enable_smoothing = enable_smoothing
        self.smooth_window_size = smooth_window_size
        self.keypoints_buffers = {}  # track_id를 키로 사용하는 딕셔너리

        self.load_model()

    def __call__(
        self, image: np.ndarray, detections: Optional[List[BBox]] = None
    ) -> Dict:
        """
        이미지에서 포즈 추정 수행
        detections이 제공되면 top-down 방식으로, 아니면 bottom-up 방식으로 작동

        Args:
            image: 입력 이미지, BGR 포맷
            detections: 검출된 사람 바운딩 박스 목록 (top-down 모드에서 사용)

        Returns:
            Dict: 키포인트 좌표 및 점수, 바운딩 박스 정보
        """
        if image is None or image.size == 0:
            return {"keypoints": np.array([]), "scores": np.array([]), "bboxes": []}

        height, width = image.shape[:2]

        # Top-down mode (검출 결과가 제공된 경우)
        if detections is not None:
            keypoints_list = []
            scores_list = []
            valid_bboxes = []

            # 각 검출에 대해 포즈 추정
            for bbox in detections:
                # 바운딩 박스 확장 (포즈 추정을 위해)
                x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

                # 중심점 계산
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # 박스 크기 계산
                box_width = (x2 - x1) * self.pose_bbox_scale
                box_height = (y2 - y1) * self.pose_bbox_scale

                # 새 바운딩 박스 좌표 계산 (확장)
                x1_new = max(0, int(center_x - box_width / 2))
                y1_new = max(0, int(center_y - box_height / 2))
                x2_new = min(width - 1, int(center_x + box_width / 2))
                y2_new = min(height - 1, int(center_y + box_height / 2))

                # 바운딩 박스가 유효한지 확인
                if x2_new <= x1_new or y2_new <= y1_new:
                    continue

                # 사람 영역 크롭
                person_img = image[y1_new:y2_new, x1_new:x2_new].copy()
                if (
                    person_img.size == 0
                    or person_img.shape[0] < 10
                    or person_img.shape[1] < 10
                ):
                    continue

                # 포즈 추정
                try:
                    # 원본 이미지 크기 저장
                    orig_h, orig_w = person_img.shape[:2]

                    # 전처리
                    img = self.preprocess(person_img)

                    # 예측
                    heatmaps = self.predict(img)

                    # 히트맵에서 키포인트 추출
                    keypoints, scores = self.postprocess(heatmaps, orig_h, orig_w)

                    # 좌표 변환 (크롭 영역 → 원본 이미지)
                    keypoints[:, 0] += x1_new
                    keypoints[:, 1] += y1_new

                    # 시간적 스무딩 적용 (활성화된 경우)
                    if self.enable_smoothing and bbox.track_id is not None:
                        keypoints = self.apply_temporal_smoothing(
                            keypoints, scores, bbox.track_id
                        )

                    keypoints_list.append(keypoints)
                    scores_list.append(scores)
                    valid_bboxes.append(bbox)
                except Exception as e:
                    print(f"포즈 추정 오류 (top-down): {e}")
                    continue

            return {
                "keypoints": (
                    np.array(keypoints_list) if keypoints_list else np.array([])
                ),
                "scores": np.array(scores_list) if scores_list else np.array([]),
                "bboxes": valid_bboxes,
            }

        # Bottom-up mode (검출 결과가 없는 경우)
        else:
            try:
                # 원본 이미지 크기 저장
                orig_height, orig_width = height, width

                # 이미지가 너무 크면 리사이즈 (효율성을 위해)
                resized_image = image.copy()
                scale_x, scale_y = 1.0, 1.0

                max_size = 1280
                if max(height, width) > max_size:
                    new_height, new_width = resized_image.shape[:2]

                    # 스케일 계산 (원본->리사이즈)
                    scale_x = new_width / float(orig_width)
                    scale_y = new_height / float(orig_height)

                    height, width = new_height, new_width

                # 전체 이미지에 대해 포즈 추정
                img = self.preprocess(resized_image)
                heatmaps = self.predict(img)
                keypoints, scores = self.postprocess(heatmaps, height, width)

                # 키포인트 유효성 확인 (최소 3개 이상의 키포인트가 신뢰도 0.3 이상)
                valid_points = np.sum(scores > 0.3)
                if valid_points < 3:
                    return {
                        "keypoints": np.array([]),
                        "scores": np.array([]),
                        "bboxes": [],
                    }

                # HRNet 입력 크기와 감지된 리사이즈된 이미지 크기 간의 스케일 조정이 필요함
                # 원본 이미지 좌표로 변환
                if scale_x != 1.0 or scale_y != 1.0:
                    keypoints[:, 0] /= scale_x
                    keypoints[:, 1] /= scale_y

                # 키포인트로부터 바운딩 박스 계산 (원본 이미지 좌표 기준)
                valid_indices = np.where(scores > 0.3)[0]
                if len(valid_indices) == 0:
                    return {
                        "keypoints": np.array([]),
                        "scores": np.array([]),
                        "bboxes": [],
                    }

                valid_keypoints = keypoints[valid_indices]
                x_min = np.min(valid_keypoints[:, 0])
                y_min = np.min(valid_keypoints[:, 1])
                x_max = np.max(valid_keypoints[:, 0])
                y_max = np.max(valid_keypoints[:, 1])

                # 바운딩 박스 생성 (원본 이미지 좌표 기준)
                bbox = BBox(
                    x1=max(0, int(x_min)),
                    y1=max(0, int(y_min)),
                    x2=min(orig_width - 1, int(x_max)),
                    y2=min(orig_height - 1, int(y_max)),
                    score=np.mean(scores),
                    class_id=1,  # 사람
                    track_id=None,
                )

                return {
                    "keypoints": np.array([keypoints]),
                    "scores": np.array([scores]),
                    "bboxes": [bbox],
                }
            except Exception as e:
                print(f"포즈 추정 오류 (bottom-up): {e}")
                return {"keypoints": np.array([]), "scores": np.array([]), "bboxes": []}

    def load_model(self):
        """
        ONNX 모델 로드
        """
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_cuda
            else ["CPUExecutionProvider"]
        )
        self.session = onnxruntime.InferenceSession(self.onnx_path, providers=providers)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        HRNet ONNX 입력에 맞춰 전처리

        Args:
            image: OpenCV로 로드한 BGR 이미지

        Returns:
            전처리된 이미지
        """
        # BGR에서 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 이미지 크기 조정
        image = cv2.resize(image, (self.input_size[1], self.input_size[0]))

        # (H, W, 3)에서 (3, H, W)로 변환
        image = image.transpose(2, 0, 1)

        # [0, 1] 범위로 정규화 및 명시적으로 float32 타입 지정
        image = image.astype(np.float32) / 255.0

        # ImageNet 정규화 (float32 타입 명시)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((3, 1, 1))
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((3, 1, 1))
        image = (image - mean) / std

        # 배치 차원 추가
        image = np.expand_dims(image, axis=0)

        # 명시적으로 float32 타입 확인
        image = image.astype(np.float32)

        return image

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        HRNet 모델로 히트맵 예측

        Args:
            image: 전처리된 이미지

        Returns:
            예측된 히트맵
        """
        try:
            # 모델 입력 이름 가져오기
            input_name = self.session.get_inputs()[0].name

            # 모델 추론
            outputs = self.session.run(None, {input_name: image})

            # 첫 번째 출력은 일반적으로 히트맵 (B, K, H, W)
            heatmaps = outputs[0]

            return heatmaps
        except Exception as e:
            print(f"HRNet 예측 오류: {e}")
            raise

    def postprocess(
        self, heatmaps: np.ndarray, orig_h: int, orig_w: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        히트맵에서 키포인트 좌표 추출

        Args:
            heatmaps: 예측된 히트맵 (B, K, H, W)
            orig_h: 원본 이미지 높이
            orig_w: 원본 이미지 너비

        Returns:
            keypoints: 키포인트 좌표 (17, 2)
            scores: 키포인트 신뢰도 점수 (17,)
        """
        # 배치 차원 제거
        heatmaps = heatmaps[0]  # (K, H, W)

        # 히트맵의 크기
        num_keypoints, h, w = heatmaps.shape

        # 키포인트 좌표와 점수 초기화
        keypoints = np.zeros((num_keypoints, 2), dtype=np.float32)
        scores = np.zeros(num_keypoints, dtype=np.float32)

        # 각 키포인트에 대해 최대값 찾기
        for i in range(num_keypoints):
            heatmap = heatmaps[i]

            # 최대값과 인덱스 찾기
            score = np.max(heatmap)
            idx = np.argmax(heatmap)
            y, x = np.unravel_index(idx, (h, w))

            # 좌표 변환에 대한 서브픽셀 정밀도 개선
            # 최대값 주변의 3x3 윈도우에서 가중 평균 계산
            if 0 < x < w - 1 and 0 < y < h - 1:
                dx = (
                    0.5
                    * (heatmap[y, x + 1] - heatmap[y, x - 1])
                    / (2 * heatmap[y, x] + heatmap[y, x + 1] + heatmap[y, x - 1] + 1e-6)
                )
                dy = (
                    0.5
                    * (heatmap[y + 1, x] - heatmap[y - 1, x])
                    / (2 * heatmap[y, x] + heatmap[y + 1, x] + heatmap[y - 1, x] + 1e-6)
                )
                x += dx
                y += dy

            # 좌표 정규화 및 원본 이미지 크기로 조정
            x_norm = x / float(w - 1) if w > 1 else 0
            y_norm = y / float(h - 1) if h > 1 else 0

            # 원본 이미지 크기로 키포인트 조정
            keypoints[i, 0] = x_norm * orig_w
            keypoints[i, 1] = y_norm * orig_h
            scores[i] = score

        return keypoints, scores

    def get_max_preds(self, heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        히트맵에서 최대값 위치 찾기

        Args:
            heatmaps: 히트맵 (B, K, H, W)

        Returns:
            preds: 키포인트 위치
            maxvals: 키포인트 신뢰도
        """
        # N: 배치 크기, K: 키포인트 수, H: 높이, W: 너비
        N, K, H, W = heatmaps.shape

        # 히트맵을 2D로 변환
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))

        # 최대값과 인덱스 찾기
        idx = np.argmax(heatmaps_reshaped, axis=2)
        maxvals = np.max(heatmaps_reshaped, axis=2)

        # 인덱스를 (y, x) 좌표로 변환
        preds = np.zeros((N, K, 2))
        preds[:, :, 0] = idx % W  # x 좌표
        preds[:, :, 1] = idx // W  # y 좌표

        return preds, maxvals

    def apply_temporal_smoothing(self, keypoints, scores, track_id):
        """
        키포인트에 시간적 스무딩 적용

        Args:
            keypoints: 현재 프레임의 키포인트 좌표 (17, 2)
            scores: 키포인트 신뢰도 점수 (17,)
            track_id: 추적 ID

        Returns:
            smoothed_keypoints: 스무딩된 키포인트 좌표
        """
        # 해당 ID의 버퍼가 없으면 새로 생성
        if track_id not in self.keypoints_buffers:
            self.keypoints_buffers[track_id] = deque(maxlen=self.smooth_window_size)

        buffer = self.keypoints_buffers[track_id]
        smoothed_keypoints = keypoints.copy()

        if buffer:  # 버퍼에 이전 프레임 데이터가 있으면
            for i in range(len(keypoints)):
                if scores[i] < 0.3:  # 낮은 신뢰도는 필터링 안함
                    continue

                weight_sum = scores[i]
                weighted_pos = keypoints[i] * scores[i]

                # 이전 프레임 키포인트 가중 평균
                for prev_keypoints, prev_scores in buffer:
                    if prev_scores[i] > 0.2:
                        weight = prev_scores[i]
                        weighted_pos += prev_keypoints[i] * weight
                        weight_sum += weight

                if weight_sum > 0:
                    smoothed_keypoints[i] = weighted_pos / weight_sum

        # 현재 키포인트 버퍼에 저장
        buffer.append((keypoints.copy(), scores.copy()))

        return smoothed_keypoints

    def clear_smoothing_buffers(self):
        """
        모든 스무딩 버퍼를 초기화
        (새로운 시퀀스 처리 시 호출)
        """
        self.keypoints_buffers.clear()
