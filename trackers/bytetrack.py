from collections import deque
from typing import Dict, List, Set

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils import BBox


# 칼만 필터를 이용한 추적 객체 클래스
class Track:
    def __init__(
        self,
        initial_det: BBox,
        track_id: int,
        max_lost: int = 30,
        max_history: int = 10,
    ):
        """
        새로운 추적 객체 초기화

        Args:
            initial_det: 첫 검출 BBox
            track_id: 할당할 추적 ID
            max_lost: 최대 손실 허용 프레임 수
            max_history: 저장할 이전 상태 히스토리 크기
        """
        self.id = track_id
        self.class_id = initial_det.class_id
        self.max_lost = max_lost
        self.time_since_update = 0

        # 바운딩 박스 히스토리 저장용 큐
        self.history = deque(maxlen=max_history)

        # 칼만 필터 초기화 (상태 7차원, 측정 4차원)
        self.kf = cv2.KalmanFilter(7, 4)

        # 상태 전이 행렬 설정 (상수 속도 모델)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],  # x -> x + vx
                [0, 1, 0, 0, 0, 1, 0],  # y -> y + vy
                [0, 0, 1, 0, 0, 0, 1],  # a -> a + va
                [0, 0, 0, 1, 0, 0, 0],  # r -> r (종횡비는 유지)
                [0, 0, 0, 0, 1, 0, 0],  # vx -> vx
                [0, 0, 0, 0, 0, 1, 0],  # vy -> vy
                [0, 0, 0, 0, 0, 0, 1],  # va -> va
            ],
            dtype=np.float32,
        )

        # 측정 행렬 설정
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],  # x
                [0, 1, 0, 0, 0, 0, 0],  # y
                [0, 0, 1, 0, 0, 0, 0],  # a (area)
                [0, 0, 0, 1, 0, 0, 0],  # r (aspect ratio)
            ],
            dtype=np.float32,
        )

        # 초기 상태 추정치 설정
        x, y, w, h = self._bbox_to_xyah(initial_det)

        # 칼만 필터 상태 초기화
        self.kf.statePost = np.array(
            [[x], [y], [w * h], [w / float(h) if h > 0 else 1.0], [0], [0], [0]],
            dtype=np.float32,
        )

        # 초기 추정 공분산 행렬 설정
        self.kf.errorCovPost = np.diag([10, 10, 10, 10, 100, 100, 100]).astype(
            np.float32
        )

        # 프로세스 잡음 공분산 설정
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32)
        # 속도 항목 잡음 조정 (움직임 모델에 따라 설정)
        self.kf.processNoiseCov[4:, 4:] *= 0.01
        # 종횡비 변화율에 대한 잡음 감소
        self.kf.processNoiseCov[6, 6] *= 0.01

        # 측정 잡음 공분산 설정
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32)
        # 면적, 종횡비는 상대적으로 불확실성 높음
        self.kf.measurementNoiseCov[2:, 2:] *= 10.0

        # 최근 측정값 저장
        self.last_detection = initial_det

        # 종횡비 제약 범위 (비정상적인 종횡비 제한)
        self.min_aspect_ratio = 0.2
        self.max_aspect_ratio = 5.0

    def _bbox_to_xyah(self, bbox: BBox):
        """
        바운딩 박스를 중심 좌표, 종횡비, 높이로 변환

        Args:
            bbox: 입력 바운딩 박스

        Returns:
            x, y: 중심 좌표
            w, h: 너비, 높이
        """
        w = bbox.x2 - bbox.x1
        h = bbox.y2 - bbox.y1
        x = bbox.x1 + w / 2.0
        y = bbox.y1 + h / 2.0
        return x, y, w, h

    def _xyah_to_bbox(self, x, y, area, aspect_ratio):
        """
        중심 좌표, 면적, 종횡비를 바운딩 박스로 변환

        Args:
            x, y: 중심 좌표
            area: 박스 면적
            aspect_ratio: 박스 종횡비 (w/h)

        Returns:
            x1, y1, x2, y2: 바운딩 박스 좌표
        """
        # 종횡비 제약 적용
        aspect_ratio = max(
            self.min_aspect_ratio, min(self.max_aspect_ratio, aspect_ratio)
        )

        # 비정상적인 면적 처리
        if area <= 0:
            # 이전 검출 정보 활용
            w, h = self._bbox_to_xyah(self.last_detection)[2:4]
            area = w * h

        # 너비, 높이 계산
        w = np.sqrt(area * aspect_ratio)
        h = area / w

        # 바운딩 박스 좌표 계산
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0

        return x1, y1, x2, y2

    def predict(self):
        """
        칼만 필터로 현재 상태 예측

        Returns:
            BBox: 예측된 바운딩박스
        """
        # 칼만 필터 예측 단계
        predicted_state = self.kf.predict()

        # 업데이트 안 된 상태 카운트 증가
        self.time_since_update += 1

        # 상태 벡터에서 값 추출
        cx, cy, area, aspect_ratio = (
            predicted_state[0, 0],
            predicted_state[1, 0],
            predicted_state[2, 0],
            predicted_state[3, 0],
        )

        # 바운딩 박스 변환
        x1, y1, x2, y2 = self._xyah_to_bbox(cx, cy, area, aspect_ratio)

        # 예측 결과 바운딩 박스 생성
        return BBox(x1, y1, x2, y2, score=1.0, track_id=self.id, class_id=self.class_id)

    def update(self, detection: BBox):
        """
        새로운 검출로 칼만 필터 상태 업데이트

        Args:
            detection: 매칭된 검출 BBox
        """
        # 검출 바운딩 박스를 중심좌표, 면적, 종횡비로 변환
        x, y, w, h = self._bbox_to_xyah(detection)
        area = w * h
        aspect_ratio = w / float(h) if h > 0 else 1.0

        # 종횡비 제약 적용
        aspect_ratio = max(
            self.min_aspect_ratio, min(self.max_aspect_ratio, aspect_ratio)
        )

        # 칼만 필터 측정 벡터 생성
        measurement = np.array([[x], [y], [area], [aspect_ratio]], dtype=np.float32)

        # 칼만 필터 보정
        self.kf.correct(measurement)

        # 시간 초기화 및 히스토리 저장
        self.time_since_update = 0
        self.last_detection = detection
        self.history.append(detection)


# ByteTrack 추적기 클래스
class ByteTrackTracker:
    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.2,
        match_iou_thresh: float = 0.3,
        max_lost: int = 30,
        max_track_id: int = 10000,
        target_classes: List[int] = None,
    ):
        """
        ByteTrack 추적기 초기화

        Args:
            high_thresh: 검출 신뢰도 상한 임계값
            low_thresh: 검출 신뢰도 하한 임계값
            match_iou_thresh: 트랙-검출 매칭 IoU 임계값
            max_lost: 트랙 손실 허용 최대 프레임 수
            max_track_id: 트랙 ID 최대값 (순환용)
            target_classes: 추적할 클래스 ID 리스트 (None이면 모든 클래스)
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_iou_thresh = match_iou_thresh
        self.max_lost = max_lost
        self.max_track_id = max_track_id
        self.target_classes = target_classes

        self.tracks = []
        self._next_track_id = 0
        self.frame_count = 0

    def __call__(self, detections: List[BBox]) -> List[BBox]:
        """
        객체 추적 수행

        Args:
            detections: 검출된 BBox 리스트

        Returns:
            track_id가 할당된 BBox 리스트
        """
        self.frame_count += 1

        # 타겟 클래스 필터링
        if self.target_classes is not None:
            detections = [
                det for det in detections if det.class_id in self.target_classes
            ]

        # 검출 결과를 신뢰도 기준으로 분류
        high_dets: List[BBox] = []
        low_dets: List[BBox] = []

        for det in detections:
            score = det.score
            if score < self.low_thresh:
                continue  # 하한 임계값 미만은 버림

            if score >= self.high_thresh:
                high_dets.append(det)
            else:
                low_dets.append(det)

        # 모든 트랙에 대해 칼만 필터 예측
        for track in self.tracks:
            track.predict()

        # 1차 연관: 높은 신뢰도 검출과 기존 트랙 매칭
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_high_dets = list(range(len(high_dets)))
        matches_high = []

        # 클래스별 매칭 수행
        tracks_by_class = self._group_by_class(self.tracks, unmatched_tracks)
        high_dets_by_class = self._group_by_class_dets(high_dets, unmatched_high_dets)

        # 각 클래스마다 별도 매칭
        for class_id, track_indices in tracks_by_class.items():
            if class_id not in high_dets_by_class:
                continue

            det_indices = high_dets_by_class[class_id]
            iou_matrix = self._compute_iou_matrix(
                [self.tracks[i] for i in track_indices],
                [high_dets[j] for j in det_indices],
            )

            # 매칭 수행
            if iou_matrix.size > 0:
                cost_matrix = 1.0 - iou_matrix
                row_idx, col_idx = linear_sum_assignment(cost_matrix)

                for r, c in zip(row_idx, col_idx):
                    if iou_matrix[r, c] >= self.match_iou_thresh:
                        matches_high.append((track_indices[r], det_indices[c]))

        # 매칭된 트랙/검출 인덱스 구하기
        matched_track_indices = {t for t, _ in matches_high}
        matched_high_indices = {d for _, d in matches_high}

        # 미매칭 목록 업데이트
        unmatched_tracks = [
            i for i in unmatched_tracks if i not in matched_track_indices
        ]
        unmatched_high_dets = [
            i for i in unmatched_high_dets if i not in matched_high_indices
        ]

        # 매칭된 트랙 업데이트
        for track_idx, det_idx in matches_high:
            track = self.tracks[track_idx]
            det = high_dets[det_idx]
            track.update(det)
            det.track_id = track.id
            det.class_id = track.class_id

        # 2차 연관: 남은 트랙과 낮은 신뢰도 검출 매칭
        matches_low = []
        unmatched_low_dets = list(range(len(low_dets)))

        # 클래스별 매칭
        tracks_by_class = self._group_by_class(self.tracks, unmatched_tracks)
        low_dets_by_class = self._group_by_class_dets(low_dets, unmatched_low_dets)

        for class_id, track_indices in tracks_by_class.items():
            if class_id not in low_dets_by_class:
                continue

            det_indices = low_dets_by_class[class_id]
            iou_matrix = self._compute_iou_matrix(
                [self.tracks[i] for i in track_indices],
                [low_dets[j] for j in det_indices],
            )

            if iou_matrix.size > 0:
                cost_matrix = 1.0 - iou_matrix
                row_idx, col_idx = linear_sum_assignment(cost_matrix)

                for r, c in zip(row_idx, col_idx):
                    if iou_matrix[r, c] >= self.match_iou_thresh:
                        matches_low.append((track_indices[r], det_indices[c]))

        # 매칭된 트랙/저신뢰 검출 인덱스 구하기
        matched_track_indices_low = {t for t, _ in matches_low}
        matched_low_indices = {d for _, d in matches_low}

        # 미매칭 목록 업데이트
        unmatched_tracks = [
            i for i in unmatched_tracks if i not in matched_track_indices_low
        ]
        unmatched_low_dets = [
            i for i in unmatched_low_dets if i not in matched_low_indices
        ]

        # 저신뢰 매칭된 트랙 업데이트
        for track_idx, det_idx in matches_low:
            track = self.tracks[track_idx]
            det = low_dets[det_idx]
            track.update(det)
            det.track_id = track.id
            det.class_id = track.class_id

        # 오래된 트랙 제거 (안전하게 역순으로)
        keep_indices = []
        for i, track_idx in enumerate(unmatched_tracks):
            track = self.tracks[track_idx]
            if track.time_since_update <= self.max_lost:
                keep_indices.append(track_idx)

        # 삭제할 트랙 필터링
        remove_indices = set(unmatched_tracks) - set(keep_indices)

        # 트랙 리스트를 안전하게 업데이트 (새 리스트 구성)
        self.tracks = [
            track for i, track in enumerate(self.tracks) if i not in remove_indices
        ]

        # 새로운 트랙 생성
        for det_idx in unmatched_high_dets:
            det = high_dets[det_idx]

            # ID 순환 처리
            if self._next_track_id >= self.max_track_id:
                self._next_track_id = 0

            new_track = Track(det, track_id=self._next_track_id, max_lost=self.max_lost)
            self.tracks.append(new_track)
            det.track_id = new_track.id
            det.class_id = new_track.class_id
            self._next_track_id += 1

        # 출력 결과 준비
        output_bboxes = []

        # 매칭된 고신뢰 검출
        for det in high_dets:
            if det.track_id is not None:
                output_bboxes.append(det)

        # 매칭된 저신뢰 검출
        for det in low_dets:
            if det.track_id is not None:
                output_bboxes.append(det)

        return output_bboxes

    def _group_by_class(self, tracks, track_indices):
        """
        트랙을 클래스별로 그룹화

        Args:
            tracks: 트랙 리스트
            track_indices: 사용할 트랙 인덱스

        Returns:
            클래스별 트랙 인덱스 딕셔너리
        """
        tracks_by_class = {}
        for idx in track_indices:
            class_id = tracks[idx].class_id
            if class_id not in tracks_by_class:
                tracks_by_class[class_id] = []
            tracks_by_class[class_id].append(idx)
        return tracks_by_class

    def _group_by_class_dets(self, detections, detection_indices):
        """
        검출을 클래스별로 그룹화

        Args:
            detections: 검출 리스트
            detection_indices: 사용할 검출 인덱스

        Returns:
            클래스별 검출 인덱스 딕셔너리
        """
        dets_by_class = {}
        for idx in detection_indices:
            class_id = detections[idx].class_id
            if class_id not in dets_by_class:
                dets_by_class[class_id] = []
            dets_by_class[class_id].append(idx)
        return dets_by_class

    def _compute_iou_matrix(self, tracks, detections):
        """
        트랙과 검출 간 IoU 행렬 계산

        Args:
            tracks: 트랙 리스트
            detections: 검출 리스트

        Returns:
            IoU 행렬 (T x D)
        """
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        # 트랙 박스 추출
        track_bboxes = []
        for track in tracks:
            state = track.kf.statePost
            cx, cy, area, aspect = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
            x1, y1, x2, y2 = track._xyah_to_bbox(cx, cy, area, aspect)
            track_bboxes.append([x1, y1, x2, y2])

        track_bboxes = np.array(track_bboxes)

        # 검출 박스 추출
        det_bboxes = np.array([[det.x1, det.y1, det.x2, det.y2] for det in detections])

        # IoU 계산
        xx1 = np.maximum(track_bboxes[:, None, 0], det_bboxes[None, :, 0])
        yy1 = np.maximum(track_bboxes[:, None, 1], det_bboxes[None, :, 1])
        xx2 = np.minimum(track_bboxes[:, None, 2], det_bboxes[None, :, 2])
        yy2 = np.minimum(track_bboxes[:, None, 3], det_bboxes[None, :, 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter_area = w * h
        track_area = (track_bboxes[:, 2] - track_bboxes[:, 0]) * (
            track_bboxes[:, 3] - track_bboxes[:, 1]
        )
        det_area = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (
            det_bboxes[:, 3] - det_bboxes[:, 1]
        )

        union_area = track_area[:, None] + det_area[None, :] - inter_area
        iou_matrix = np.where(union_area > 0, inter_area / union_area, 0.0).astype(
            np.float32
        )

        return iou_matrix

    def reset(self):
        """
        추적기 초기화 (다음 시퀀스 처리용)
        """
        self.tracks = []
        self._next_track_id = 0
        self.frame_count = 0
