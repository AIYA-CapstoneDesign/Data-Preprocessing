import math
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime

from utils import BBox


class YOLOv8:
    """
    YOLOv8 model
    """

    def __init__(
        self,
        onnx_path: str,
        cuda: bool = False,
        person_class_id: int = 0,
        score_threshold: Optional[float] = 0.65,
        iou_threshold: Optional[float] = 0.5,
    ):
        self.onnx_path = onnx_path
        self.use_cuda = cuda
        self.person_class_id = person_class_id
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        self.load_model()

    def __call__(self, image: np.ndarray) -> list[BBox]:
        original_H, original_W = image.shape[:2]

        image, top, left = self.preprocess(image)
        resized_H, resized_W = image.shape[2], image.shape[3]

        outputs = self.predict(image)

        bboxes = self.postprocess(
            outputs, original_H, original_W, resized_H, resized_W, top, left
        )

        return bboxes

    def load_model(self):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_cuda
            else ["CPUExecutionProvider"]
        )
        self.session = onnxruntime.InferenceSession(self.onnx_path, providers=providers)

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
        YOLOv8 ONNX 입력에 맞춰 전처리
        YOLOv8 ONNX 입력은 (1, 3, 640, 640) 형태

        Args:
            image (np.ndarray): OpenCV로 로드한 BGR 이미지 (1, 3, 640, 640)
        Returns:
            image (np.ndarray): 전처리된 BGR 이미지 (1, 3, 640, 640)
            top (int): 상단 패딩 값
            left (int): 좌측 패딩 값
        """
        # RGB로 변환환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 이미지 크기 조정
        # 가로 세로 종횡비 유지
        image, (top, left) = self.letterbox(image, (640, 640))

        # (H, W, 3) -> (3, H, W)
        image = image.transpose(2, 0, 1)

        # int -> float32로 변환
        image = image.astype(np.float32)

        # YOLOv8 정규화
        image = image / 255.0

        # 배치 차원 추가
        image = np.expand_dims(image, axis=0)

        return image, top, left

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        YOLOv8 모델 추론
        """

        input_data = {self.session.get_inputs()[0].name: image}

        outputs = self.session.run(None, input_data)

        return outputs

    def postprocess(
        self, outputs, original_H, original_W, resized_H, resized_W, pad_h, pad_w
    ):
        """
        YOLOv8 모델 후처리
        Non-max Suppression은 ONNX 모델 내에서 수행되므로 별도로 수행하지 않음

        Args:
            outputs: 모델 출력 결과
            original_H: 원본 이미지 높이
            original_W: 원본 이미지 너비
            resized_H: 리사이즈된 이미지 높이
            resized_W: 리사이즈된 이미지 너비
            pad_h: 상단 패딩 값
            pad_w: 좌측 패딩 값
        Returns:
            bboxes (list[BBox]): 후처리된 박스 좌표
        """
        # 배치 차원 제거
        outputs = np.transpose(np.squeeze(outputs[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        # 패딩 제외한 실제 이미지 크기 계산
        unpadded_H = resized_H - 2 * pad_h
        unpadded_W = resized_W - 2 * pad_w

        # 스케일 계산
        scale_x = original_W / unpadded_W
        scale_y = original_H / unpadded_H

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Get the class ID with the highest score and its score
            class_id = np.argmax(classes_scores)
            max_score = classes_scores[class_id]

            # 점수가 너무 낮은 경우 무시
            if max_score < 0.25:
                continue

            # person_class_id와 일치하는 클래스만 처리
            if class_id != self.person_class_id:
                continue

            # Extract the bounding box coordinates from the current row (XYWH 형식)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # 패딩 제거
            x = x - pad_w
            y = y - pad_h

            # 패딩을 제외한 실제 이미지 영역 내에서의 비율로 변환
            # 원본 이미지 크기로 스케일링
            left = max(0, int((x - w / 2) * scale_x))
            top = max(0, int((y - h / 2) * scale_y))
            width = min(original_W, int(w * scale_x))
            height = min(original_H, int(h * scale_y))

            # 유효한 좌표인지 확인
            if left >= original_W or top >= original_H or width <= 0 or height <= 0:
                continue

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        if not boxes:  # 유효한 박스가 없으면 빈 리스트 반환
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.score_threshold if self.score_threshold is not None else 0.0,
            self.iou_threshold,
        )

        if not isinstance(indices, list) and len(indices) > 0:
            indices = indices.flatten()

        bboxes = []
        for i in indices:
            left = boxes[i][0]
            top = boxes[i][1]
            width = boxes[i][2]
            height = boxes[i][3]
            score = scores[i]
            class_id = class_ids[i]
            bboxes.append(BBox(left, top, left + width, top + height, score, class_id))

        return bboxes
