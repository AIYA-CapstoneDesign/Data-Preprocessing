# 데이터셋 A: 낙상사고 위험동작 영상-센서 쌍 데이터

- **링크:** https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71641
- **어노테이션 형식:** JSON
- **영상 포맷:** MP4
- **영상 FPS:** 60

## Raw 데이터 디렉토리 구조도

```plaintext
.
└── dataset_a/
    ├── 원천데이터/
    │   └── 영상/
    │       ├── N(비낙상)/
    │       │   └── N(비낙상)/
    │       │       ├── 00047_H_A_N_C1/
    │       │       │   └── 00047_H_A_N_C1.mp4
    │       │       ├── 00047_H_A_N_C2/
    │       │       │   └── 00047_H_A_N_C2.mp4
    │       │       └── ...
    │       └── Y(낙상)/
    │           ├── BY(후면낙상)/
    │           │   ├── 00151_H_A_BY_C1/
    │           │   │   └── 00151_H_A_BY_C1.mp4
    │           │   ├── 00151_H_A_BY_C2/
    │           │   │   └── 00151_H_A_BY_C2.mp4
    │           │   └── ...
    │           ├── FY(전면낙상)/
    │           │   └── ...
    │           └── SY(측면낙상)/
    │               └── ...
    └── 라벨링데이터/
        └── 영상/
            ├── N(비낙상)/
            │   └── N(비낙상)/
            │       ├── 00047_H_A_N_C1/
            │       │   └── 00047_H_A_N_C1.json
            │       ├── 00047_H_A_N_C2/
            │       │   └── 00047_H_A_N_C2.json
            │       └── ...
            └── Y(낙상)/
                ├── BY(후면낙상)/
                │   ├── 00151_H_A_BY_C1/
                │   │   └── 00151_H_A_BY_C1.json
                │   ├── 00151_H_A_BY_C2/
                │   │   └── 00151_H_A_BY_C2.json
                │   └── ...
                ├── FY(전면낙상)/
                │   └── ...
                └── SY(측면낙상)/
                    └── ...
```

## 어노테이션 형식

### JSON

```json
{
  "metadata": {
    "description": "낙상사고 위험동작 이미지 데이터",
    "scene_id": "00015_H_A_SY_C2",
    "scene_format": "MP4",
    "scene_res": "3840 X 2160",
    "creator": "순천향대학교 산학협력단",
    "distributor": "NIA",
    "date": "2023-09-05"
  },
  "scene_info": {
    "scene_loc": "병원",
    "scene_pos": "병실",
    "scene_method": "none",
    "scene_IsFall": "낙상",
    "scene_cat_name": "측면낙상",
    "fall_type": "중심을 잃고 넘어짐",
    "scene_length": 600,
    "cam_num": 2
  },
  "actor_info": {
    "actor_id": "F",
    "actor_age": "adult1(청소년청년)",
    "actor_sex": "m"
  },
  "sensordata": {
    "fall_start_frame": 281,
    "fall_end_frame": 341
  },
  "scene_path": {
    "scene_path": "낙상/Y/SY/00015_H_A_SY_C2"
  }
}
```

### 주요 키

- `scene_isFall`: 낙상, 비낙상 여부
- `fall_start_frame`: 낙상 이벤트 시작 프레임
- `fall_end_frame`: 낙상 이벤트 종료 프레임
  
### 비고

- 영상 데이터에 등장하는 배우가 1명으로, 골격 추출 시 별도의 Action Localization을 할 필요가 없어 보임.
- 전면 낙상, 후면 낙상, 측면 낙상을 전부 낙상 클래스로 통합.
  - 낙상 : 비낙상 데이터 비율이 3 : 1로 클래스 불균형이 발생할 가능성이 큼.
