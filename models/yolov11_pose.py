import math
from typing import Optional, Tuple, List

import cv2
import numpy as np
import onnxruntime
from collections import deque

from utils import BBox


class YOLOv11Pose:
    """
    YOLOv11-Pose 모델
    """

    def __init__(
        self,
        onnx_path: str,
        cuda: bool = False,
        person_class_id: int = 0,
        score_threshold: Optional[float] = 0.5,
        iou_threshold: Optional[float] = 0.5,
        batch_size: int = 1,
        enable_smoothing: bool = False,
        smooth_window_size: int = 5,
    ):
        self.onnx_path = onnx_path
        self.use_cuda = cuda
        self.person_class_id = person_class_id
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.num_keypoints = 17  # COCO 키포인트 수
        self.batch_size = batch_size
        
        # 시간적 스무딩 관련 설정
        self.enable_smoothing = enable_smoothing
        self.smooth_window_size = smooth_window_size
        self.keypoints_buffers = {}  # track_id를 키로 사용하는 딕셔너리
        
        self.load_model()

    def __call__(self, image: np.ndarray) -> dict:
        original_H, original_W = image.shape[:2]

        image, top, left = self.preprocess(image)
        resized_H, resized_W = image.shape[2], image.shape[3]

        outputs = self.predict(image)

        results = self.postprocess(
            outputs, original_H, original_W, resized_H, resized_W, top, left
        )

        return results

    def load_model(self):
        """
        ONNX 모델 로드
        """
        sess_opts = onnxruntime.SessionOptions()
        sess_opts.log_severity_level = 3
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_cuda
            else ["CPUExecutionProvider"]
        )
        self.session = onnxruntime.InferenceSession(
            self.onnx_path, sess_options=sess_opts, providers=providers
        )

    def letterbox(
        self, image: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        패딩을 추가하여, 이미지를 640x640 크기로 리사이즈.

        Args:
            image (np.ndarray): 입력 이미지
            new_shape (Tuple[int, int]): 목표 크기 (height, width)

        Returns:
            (np.ndarray): 리사이즈되고 패딩이 추가된 이미지
            (Tuple[int, int]): 패딩 값 (top, left)
        """
        shape = image.shape[:2]  # current shape [height, width]

        # 비율 계산
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (
            new_shape[0] - new_unpad[1]
        ) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return image, (top, left)

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        YOLOv11-Pose ONNX 입력에 맞춰 전처리
        YOLOv11-Pose ONNX 입력은 (1, 3, 640, 640) 형태

        Args:
            image (np.ndarray): OpenCV로 로드한 BGR 이미지
        Returns:
            image (np.ndarray): 전처리된 이미지 (1, 3, 640, 640)
            top (int): 상단 패딩 값
            left (int): 좌측 패딩 값
        """
        # RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 이미지 크기 조정 (가로 세로 종횡비 유지)
        image, (top, left) = self.letterbox(image, (640, 640))

        # (H, W, 3) -> (3, H, W)
        image = image.transpose(2, 0, 1)

        # int -> float32로 변환
        image = image.astype(np.float32)

        # 정규화
        image = image / 255.0

        # 배치 차원 추가
        image = np.expand_dims(image, axis=0)

        return image, top, left

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        YOLOv11-Pose 모델 추론
        
        Returns:
            np.ndarray: 출력 텐서 형태
        """
        input_name = self.session.get_inputs()[0].name
        input_data = {input_name: image}
        outputs = self.session.run(None, input_data)
        return outputs[0]
    
    def postprocess(
        self, outputs, original_H, original_W, resized_H, resized_W, pad_h, pad_w
    ) -> dict:
        """
        YOLOv11-Pose 모델 후처리
        
        Args:
            outputs: 모델 출력 결과 (1, 56, 8400)
            original_H: 원본 이미지 높이
            original_W: 원본 이미지 너비
            resized_H: 리사이즈된 이미지 높이
            resized_W: 리사이즈된 이미지 너비
            pad_h: 상단 패딩 값
            pad_w: 좌측 패딩 값
            
        Returns:
            dict: {"bboxes": List[BBox], "keypoints": np.ndarray, "scores": np.ndarray}
        """
        # 배치 차원 제거 및 형상 조정
        # 출력이 (1, 56, 8400)이 아닐 경우 다른 방식으로 처리
        if len(outputs.shape) == 3 and outputs.shape[1] == 56:
            # 예측 결과 형상 조정 (1, 56, 8400) -> (8400, 56)
            predictions = np.transpose(outputs, (0, 2, 1))[0]
        else:
            # 다른 출력 형태일 경우 적절히 처리
            if len(outputs.shape) == 2:  # (N, 56) 형태로 이미 변환된 경우
                predictions = outputs
            elif len(outputs.shape) > 3:  # 추가 차원이 있는 경우
                predictions = outputs.reshape(-1, 56)
            else:
                # 기본 처리 시도
                predictions = np.transpose(outputs, (2, 1, 0)).reshape(-1, 56)
            
        # 바운딩 박스, 객체성, 키포인트 분리
        # 4(bbox) + 1(obj) + 17*3(kpt) = 56
        box_predictions = predictions[:, :4]  # x, y, w, h (중심점 + 크기)
        obj_predictions = predictions[:, 4]  # 객체성 점수
        kpt_predictions = predictions[:, 5:].reshape(-1, 17, 3)  # 17개 키포인트 x, y, conf
        
        if len(box_predictions) == 0:
            return {"bboxes": [], "keypoints": np.array([]), "scores": np.array([])}
        
        # 패딩 제외한 실제 이미지 크기 계산
        unpadded_H = resized_H - 2 * pad_h
        unpadded_W = resized_W - 2 * pad_w
        
        # 원본 이미지 스케일 계산
        scale_x = original_W / unpadded_W
        scale_y = original_H / unpadded_H
        
        # 바운딩 박스를 XYWH(중심점, 너비, 높이)에서 XYXY(좌상단, 우하단)로 변환
        boxes = []
        valid_indices = []  # 유효한 박스의 인덱스를 저장
        
        for i, box in enumerate(box_predictions):
            x, y, w, h = box
            
            # 패딩 제거
            x = x - pad_w
            y = y - pad_h
            
            # 패딩을 제외한 실제 이미지 영역 내에서의 비율로 변환 후 원본 이미지 크기로 스케일링
            left = max(0, int((x - w / 2) * scale_x))
            top = max(0, int((y - h / 2) * scale_y))
            right = min(original_W, int((x + w / 2) * scale_x))
            bottom = min(original_H, int((y + h / 2) * scale_y))
            
            # 유효한 좌표인지 확인
            if left >= original_W or top >= original_H or right <= 0 or bottom <= 0 or right <= left or bottom <= top:
                continue
                
            boxes.append([left, top, right, bottom])
            valid_indices.append(i)  # 유효한 박스의 인덱스 저장
        
        if not boxes:
            return {"bboxes": [], "keypoints": np.array([]), "scores": np.array([])}
        
        # 유효한 박스에 대응하는 점수만 선택
        valid_scores = obj_predictions[valid_indices]
        valid_keypoints = kpt_predictions[valid_indices]
        
        # NMS 적용
        boxes_np = np.array(boxes)
        scores_np = np.array(valid_scores)
        
        # 크기 일치 확인
        if len(boxes_np) != len(scores_np):
            print(f"경고: 바운딩 박스({len(boxes_np)})와 점수({len(scores_np)})의 크기가 일치하지 않습니다.")
            return {"bboxes": [], "keypoints": np.array([]), "scores": np.array([])}
        
        indices = cv2.dnn.NMSBoxes(
            boxes_np.tolist(),
            scores_np.tolist(),
            self.score_threshold if self.score_threshold is not None else 0.0,
            self.iou_threshold
        )
        
        if len(indices) == 0:
            return {"bboxes": [], "keypoints": np.array([]), "scores": np.array([])}
            
        if not isinstance(indices, list):
            indices = indices.flatten()
        
        # 최종 결과 생성
        bboxes = []
        final_keypoints = []
        final_kpt_scores = []
        
        for i in indices:
            # 바운딩 박스 생성
            idx = int(i)  # 인덱스 정수 확인
            left, top, right, bottom = boxes_np[idx]
            score = scores_np[idx]
            bboxes.append(BBox(left, top, right, bottom, score, self.person_class_id))
            
            # 키포인트 변환
            keypoints = valid_keypoints[idx]
            keypoints_xy = keypoints[:, :2].copy()
            
            # 패딩 제거
            keypoints_xy[:, 0] -= pad_w
            keypoints_xy[:, 1] -= pad_h
            
            # 원본 이미지 크기로 스케일링
            keypoints_xy[:, 0] *= scale_x
            keypoints_xy[:, 1] *= scale_y
            
            # 키포인트 점수
            keypoints_conf = keypoints[:, 2]
            
            # 시간적 스무딩 적용 (활성화된 경우)
            if self.enable_smoothing and bboxes[-1].track_id is not None:
                keypoints_xy = self.apply_temporal_smoothing(
                    keypoints_xy, keypoints_conf, bboxes[-1].track_id
                )
                
            final_keypoints.append(keypoints_xy)
            final_kpt_scores.append(keypoints_conf)
        
        # 최종 결과를 상위 10개로 제한 (선택적)
        if len(bboxes) > 10:
            # 점수 기준으로 상위 10개만 선택
            scores_array = np.array([bbox.score for bbox in bboxes])
            top_indices = np.argsort(scores_array)[-10:]
            
            bboxes = [bboxes[i] for i in top_indices]
            final_keypoints = [final_keypoints[i] for i in top_indices]
            final_kpt_scores = [final_kpt_scores[i] for i in top_indices]
        
        return {
            "bboxes": bboxes,
            "keypoints": np.array(final_keypoints),
            "scores": np.array(final_kpt_scores)
        }

    def process_batch(self, images: List[np.ndarray]) -> List[dict]:
        """
        배치 이미지 처리 (속도 향상)
        
        Args:
            images: 이미지 리스트
            
        Returns:
            List[dict]: 각 이미지에 대한 추론 결과 리스트
        """
        if not images:
            return []
        
        batch_size = min(len(images), self.batch_size)
        results = []
        
        # 배치 단위로 처리
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            # 각 이미지 정보 저장
            orig_shapes = []
            processed_batch = []
            padding_info = []
            
            # 배치 내 이미지 전처리
            for img in batch_images:
                original_H, original_W = img.shape[:2]
                orig_shapes.append((original_H, original_W))
                
                processed_img, top, left = self.preprocess(img)
                processed_batch.append(processed_img)
                padding_info.append((top, left))
            
            # 배치 이미지 준비
            if len(processed_batch) == 1:
                # 단일 이미지인 경우
                batch_input = processed_batch[0]
            else:
                # 여러 이미지 배치로 합치기
                batch_input = np.concatenate(processed_batch, axis=0)
            
            # 배치 추론
            batch_outputs = self.predict_batch(batch_input)
            
            # 결과 후처리
            for j in range(len(batch_images)):
                original_H, original_W = orig_shapes[j]
                pad_h, pad_w = padding_info[j]
                
                # 배치 출력에서 해당 이미지 결과 가져오기
                if len(processed_batch) == 1:
                    output = batch_outputs
                else:
                    output = batch_outputs[j:j+1]
                
                # 후처리
                resized_H, resized_W = processed_batch[0].shape[2], processed_batch[0].shape[3]
                result = self.postprocess(
                    output, original_H, original_W, resized_H, resized_W, pad_h, pad_w
                )
                results.append(result)
        
        return results

    def predict_batch(self, batch_input: np.ndarray) -> np.ndarray:
        """
        배치 입력에 대한 모델 추론
        
        Args:
            batch_input: 배치 처리된 이미지 입력 (B, 3, 640, 640)
            
        Returns:
            np.ndarray: 출력 텐서
        """
        input_name = self.session.get_inputs()[0].name
        input_data = {input_name: batch_input}
        outputs = self.session.run(None, input_data)
        return outputs[0]

    def clear_smoothing_buffers(self):
        """
        모든 스무딩 버퍼를 초기화
        (새로운 시퀀스 처리 시 호출)
        """
        self.keypoints_buffers.clear()

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

