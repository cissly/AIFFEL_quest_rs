# config.py
import os

# 프로젝트 기본 경로
PROJECT_PATH = r"C:\Users\tkdwl\AIFFEL_quest_rs\GoingDeeper\GD08"

# 원본 JSON / 이미지 / 모델 / pt_records 경로
IMAGE_PATH = os.path.join(PROJECT_PATH, 'data', 'images')
MODEL_PATH = os.path.join(PROJECT_PATH, 'data', 'models')
PT_RECORD_PATH = os.path.join(PROJECT_PATH, 'data', 'pt_records')

TRAIN_JSON = os.path.join(PROJECT_PATH, 'data', 'train.json')
VALID_JSON = os.path.join(PROJECT_PATH, 'data', 'validation.json')

# (노트북 하단에 있던 것들)
IMAGE_DIR = os.path.join(PROJECT_PATH, 'images')        # 필요 시 사용
VAL_JSON = os.path.join(PROJECT_PATH, 'data', 'validation.json')
WEIGHTS_PATH = os.path.join(
    PROJECT_PATH, 'data', 'models', 'model-epoch-2-loss-1.2050.pt'
)

# 이미지 / 히트맵 관련 상수
IMAGE_SHAPE = (256, 256, 3)
HEATMAP_SIZE = (64, 64)

# 조인트 인덱스 상수들
R_ANKLE = 0
R_KNEE = 1
R_HIP = 2
L_HIP = 3
L_KNEE = 4
L_ANKLE = 5
PELVIS = 6
THORAX = 7
UPPER_NECK = 8
HEAD_TOP = 9
R_WRIST = 10
R_ELBOW = 11
R_SHOULDER = 12
L_SHOULDER = 13
L_ELBOW = 14
L_WRIST = 15

# 뼈대 연결 관계
MPII_BONES = [
    [R_ANKLE, R_KNEE],
    [R_KNEE, R_HIP],
    [L_ANKLE, L_KNEE],
    [L_KNEE, L_HIP],
    [PELVIS, R_HIP],
    [PELVIS, L_HIP],
    [PELVIS, THORAX],
    [THORAX, UPPER_NECK],
    [UPPER_NECK, HEAD_TOP],
    [R_WRIST, R_ELBOW],
    [R_ELBOW, R_SHOULDER],
    [THORAX, R_SHOULDER],
    [THORAX, L_SHOULDER],
    [L_SHOULDER, L_ELBOW],
    [L_ELBOW, L_WRIST],
]