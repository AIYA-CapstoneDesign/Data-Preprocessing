import math
from typing import Optional

import cv2
import numpy as np
import onnxruntime

from utils import BBox, resize_shorter_side


class FasterRCNN:
    """
    Faster R-CNN model
    """

    def __init__(
        self,
        onnx_path: str,
        cuda: bool = False,
        person_class_id: int = 1,
        score_threshold: Optional[float] = 0.5,
    ):
        self.onnx_path = onnx_path
        self.use_cuda = cuda
        self.person_class_id = person_class_id
        self.score_threshold = score_threshold

        self.load_model()

    def __call__(self, image: np.ndarray) -> list[BBox]:
        original_H, original_W = image.shape[:2]

        image = self.preprocess(image)
        resized_H, resized_W = image.shape[1:]

        boxes, labels, scores = self.predict(image)

        bboxes = self.postprocess(
            boxes, labels, scores, original_H, original_W, resized_H, resized_W
        )

        return bboxes

    def load_model(self):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_cuda
            else ["CPUExecutionProvider"]
        )
        self.session = onnxruntime.InferenceSession(self.onnx_path, providers=providers)

    def preprocess(self, image: np.ndarray):
        """
        Faster R-CNN ONNX 입력에 맞춰 전처리
        Faster R-CNN ONNX 입력은 배치 차원 없이 (3, H, W) 형태

        Args:
            image (np.ndarray): OpenCV로 로드한 BGR 이미지 (3, H, W)
        Returns:
            image (np.ndarray): 전처리된 BGR 이미지 (3, H, W)
        """

        # 이미지 크기 조정
        # 가로 세로 종횡비 유지
        image = resize_shorter_side(image, 800)

        # (H, W, 3) -> (3, H, W)
        image = image.transpose(2, 0, 1)

        # int -> float32로 변환
        image = image.astype(np.float32)

        # Faster R-CNN 정규화
        mean_vec = np.array([102.9801, 115.9465, 122.7717])
        for i in range(image.shape[0]):
            image[i, :, :] = image[i, :, :] - mean_vec[i]

        # 32의 배수로 패딩
        padded_h = int(math.ceil(image.shape[1] / 32) * 32)
        padded_w = int(math.ceil(image.shape[2] / 32) * 32)

        padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
        padded_image[:, : image.shape[1], : image.shape[2]] = image
        image = padded_image

        return image

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Faster R-CNN 모델 추론
        """

        input_data = {self.session.get_inputs()[0].name: image}

        boxes, labels, scores = self.session.run(None, input_data)

        return boxes, labels, scores

    def postprocess(
        self, boxes, labels, scores, original_H, original_W, resized_H, resized_W
    ):
        """
        Faster R-CNN 모델 후처리
        Non-max Suppression은 ONNX 모델 내에서 수행되므로 별도로 수행하지 않음

        Args:
            boxes (np.ndarray): 추론 결과 박스 좌표
            labels (np.ndarray): 추론 결과 레이블
            scores (np.ndarray): 추론 결과 점수
        Returns:
            bboxes (list[BBox]): 후처리된 박스 좌표
        """
        bboxes = []
        for box, label, score in zip(boxes, labels, scores):

            if label == self.person_class_id and (
                self.score_threshold is None or score >= self.score_threshold
            ):
                x_ratio = original_W / resized_W
                y_ratio = original_H / resized_H
                bboxes.append(
                    BBox(
                        box[0] * x_ratio,
                        box[1] * y_ratio,
                        box[2] * x_ratio,
                        box[3] * y_ratio,
                        score=score,
                        track_id=None,
                        class_id=label,
                    )
                )

        return bboxes
