from dataclasses import dataclass
from typing import Optional


@dataclass
class BBox:
    """
    박스 좌표 및 점수
    """

    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    class_id: int
    track_id: Optional[int] = None
