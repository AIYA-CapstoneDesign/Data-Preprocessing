# 시니어 이상행동 영상 전처리

## 환경 설정

### 요구 사항

- Python (3.8 ~ 3.10)
- Anaconda3 (권장, Miniconda 사용 가능)
- CUDA 환경 (골격 데이터 추출 가속)

## 단계

### 1. 클립 추출

원본 비디오에서 타임스탬프 기반으로 행동 클립을 추출한다. 

### 2. 골격 데이터 추출

MMPose의 HRNet 기반으로 Top down Pose Estimation을 수행하여 COCO-Keypoint 형식 17 키포인트를 추출한다. 결과물은 각 클립에 대한 `.pkl` 파일이다.

### 3. 최종 데이터 생성

각 클립 `.pkl`을 스플릿(train or val) 단위로 묶어 최종 데이터셋을 생성한다.

## 참고: 데이터 포맷

### 시니어 이상행동 영상 라벨 (JSON)

```json
{
	"annotations": {
		"duration": "00:05:00",
		"resourceId": "JFID_002330967",
		"resource": "FD_In_H11H22H31_0001_20201016_20.mp4",
		"resourcePath": ".//PID_000001857/",
		"fps": 29.97,
		"totFrame": 0,
		"resourceSize": 381141978,
		"object": [
			{
				"startPosition": {
					"x": "1968.2953307735665",
					"y": "377.0414201183432",
					"keyFrame": 7801.0
				},
				"endPosition": {
					"x": "2083.3255773772166",
					"y": "939.4082840236686",
					"keyFrame": 7860.0
				},
				"startFrame": 7801.0,
				"endFrame": 7860.0,
				"actionType": "ABNOR_H",
				"actionName": "H11H22H31"
			}
		]
	}
}
```

- `duration`: 영상 총 길이
- `fps`: 영상 FPS
- `object`: 이상행동 타임스탬프
  - `startPosition`: 행동 시작 시 위치 (가슴 중앙)
  - `endPosition`: 행동 종료 시 위치 (가슴 중앙)
  - `startFrame`: 행동 시작 시 프레임
  - `endFrame`: 행동 종료 시 프레임
  - `actionType`: 이상행동 타입
    - `ABNOR_H`: 낙상
    - `ABNOR_W`: 배회
    - `ACT_I_TYPE`: 일상생활

### 골격 데이터 포맷 (Pickle)

```python
{
    "split":
        {
            "train": ["S001C001P001R001A001", ...],
            "val": ["S001C001P003R001A001", ...],
            ...
        }
    "annotations":
        [
            {
                {
                    "frame_dir": "S001C001P001R001A001",
                    "label": 0,
                    "img_shape": (1080, 1920),
                    "original_shape": (1080, 1920),
                    "total_frames": 103,
                    "keypoint": array([[[[1032. ,  334.8], ...]]])
                    "keypoint_score": array([[[0.934 , 0.9766, ...]]])
                },
                {
                    "frame_dir": "S001C001P003R001A001",
                    ...
                },
                ...
            }
        ]
}
```

- `split`: Dictionary, 키는 스플릿 이름, 값은 해당하는 스플릿에 대한 클립 고유 이름을 모은 리스트
- `annotations`: List of Dictionary, 각 딕셔너리는 클립별 행동 정보를 포함
  - `frame_dir`: 클립의 고유 이름
  - `label`: 정수형 클래스
  - `img_shape`: 영상의 크기, `(Height, Width)` 형식
  - `original_shape`: `img_shape`와 같음
  - `total_frames`: 클립 총 길이
  - `keypoint`: List of Tuple, 키포인트 정보
    - 키포인트의 좌표, `(X, Y)` 형식
    - `(사람 수, 프레임, 17, 2)`
  - `keypoint_score`: 키포인트 점수
    - `(사람 수, 프레임, 17)`

## 주의
- 영상(`.mp4`, `.avi`), 골격 데이터(`.pkl`)를 직접 업로드하지 말 것.
  - `git add .`, `git add *` 사용 지양