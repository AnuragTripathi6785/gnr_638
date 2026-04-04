import os

# Auto-detect environment
if os.path.exists('/content'):
    DATA_ROOT = os.environ.get('YOLO_DATA_ROOT', '/content/yolov1/data')
elif os.path.exists('/kaggle/working'):
    DATA_ROOT = os.environ.get('YOLO_DATA_ROOT', '/kaggle/working/data')
else:
    DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TRAIN_IMG_DIR   = os.path.join(DATA_ROOT, "train", "images")
TRAIN_LABEL_DIR = os.path.join(DATA_ROOT, "train", "labels")
VALID_IMG_DIR   = os.path.join(DATA_ROOT, "valid", "images")
VALID_LABEL_DIR = os.path.join(DATA_ROOT, "valid", "labels")
TEST_IMG_DIR    = os.path.join(DATA_ROOT, "test",  "images")
TEST_LABEL_DIR  = os.path.join(DATA_ROOT, "test",  "labels")

CLASS_NAMES = ["bike", "car", "cat", "cellphone", "dog", "person"]
NUM_CLASSES = len(CLASS_NAMES)  # 6

S = 7    # grid size
B = 2    # boxes per cell
CELL_SIZE = B * 5 + NUM_CLASSES  # 16

IMAGE_SIZE = 448

BATCH_SIZE = 64
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 135

LR_WARMUP_EPOCHS = 5
LR_INIT = 1e-3
LR_MAIN = 1e-2
LR_SCHEDULE = [
    (LR_WARMUP_EPOCHS, LR_INIT),
    (LR_WARMUP_EPOCHS + 75, LR_MAIN),
    (LR_WARMUP_EPOCHS + 75 + 30, 1e-3),
    (LR_WARMUP_EPOCHS + 75 + 30 + 30, 1e-4),
]

DROPOUT_RATE = 0.5

LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5

AUG_SCALE_RANGE = (0.8, 1.2)
AUG_TRANSLATE = 0.2
AUG_HSV_EXPOSURE = 1.5
AUG_HSV_SATURATION = 1.5
AUG_FLIP_PROB = 0.5

CONF_THRESHOLD = 0.20
NMS_THRESHOLD = 0.5

if os.path.exists('/content'):
    CHECKPOINT_DIR = '/content/checkpoints'
    LOG_DIR        = '/content/logs'
elif os.path.exists('/kaggle/working'):
    CHECKPOINT_DIR = '/kaggle/working/checkpoints'
    LOG_DIR        = '/kaggle/working/logs'
else:
    CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    LOG_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

DEVICE = "cuda"
