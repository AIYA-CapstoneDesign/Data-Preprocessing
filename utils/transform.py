import cv2
import numpy as np


def resize_image(
    image: np.ndarray, width: int, height: int, keep_aspect_ratio: bool = True, center: bool = True
) -> np.ndarray:
    """
    이미지를 원하는 너비와 높이로 리사이즈함.

    Args:
        image (np.ndarray): 입력 이미지 
        width (int): 원하는 너비
        height (int): 원하는 높이
        keep_aspect_ratio (bool): 비율 유지 여부
        center (bool): 중앙 정렬 여부. False일 경우 좌상단 정렬

    Returns:
        np.ndarray: 리사이즈된 이미지
    """
    h, w = image.shape[:2]
    if keep_aspect_ratio:
        ratio = min(width / w, height / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
    else:
        new_w, new_h = width, height

    resized = cv2.resize(image, (new_w, new_h))

    # 패딩 계산
    if center:
        # 중앙 정렬을 위한 패딩 계산
        pad_x = (width - new_w) // 2
        pad_y = (height - new_h) // 2
    else:
        # 좌상단 정렬
        pad_x = 0
        pad_y = 0

    # 패딩 적용
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return image
    


def min_max_resize(
    image: np.ndarray, min_size: int, max_size: int, keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    이미지의 짧은 쪽은 최소 min_size, 긴 쪽은 최대 max_size가 되도록 리사이즈.
    keep_aspect_ratio=True: 비율 유지 + 패딩
    keep_aspect_ratio=False: 비율 무시, 각 차원 clamp해서 스트레칭
    """
    h, w = image.shape[:2]

    if not keep_aspect_ratio:
        # 각 차원별로 clamp 후 스트레칭 리사이즈
        new_h = int(min(max(h, min_size), max_size))
        new_w = int(min(max(w, min_size), max_size))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 1단계: 짧은 쪽 → min_size
    scale = min_size / min(h, w)
    # 2단계: 긴 쪽이 max_size 초과하면 스케일 재계산
    if max(h, w) * scale > max_size:
        scale = max_size / max(h, w)

    # 실제 리사이즈
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 패딩할 최종 크기 계산
    target_h = max(min_size, new_h)
    target_w = max(min_size, new_w)

    # 패딩 크기 (위/아래, 좌/우)
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # 검은색(0)으로 패딩
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],  # 컬러 이미지는 3채널, 그레이면 자동으로 0
    )
    return padded


def resize_shorter_side(image: np.ndarray, min_size: int) -> np.ndarray:
    """
    이미지의 짧은 쪽을 최소 min_size로 리사이즈.
    """
    h, w = image.shape[:2]
    scale = min_size / min(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return resized
