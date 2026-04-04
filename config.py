"""
Configuration for YOLOv1 implementation.
All hyperparameters follow the original CVPR 2016 paper by Redmon et al.
"""

import os

# ── Dataset ──────────────────────────────────────────────────────────────────
# Auto-detect environment: Colab / Kaggle / Local
if os.path.exists('/content'):
    # Google Colab — project extracted from Drive zip to /content/yolov1
    DATA_ROOT = os.environ.get('YOLO_DATA_ROOT', '/content/yolov1/data')
elif os.path.exists('/kaggle/working'):
    # Kaggle
    DATA_ROOT = os.environ.get('YOLO_DATA_ROOT', '/kaggle/working/data')
else:
    # Local
    DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TRAIN_IMG_DIR   = os.path.join(DATA_ROOT, "train", "images")
TRAIN_LABEL_DIR = os.path.join(DATA_ROOT, "train", "labels")
VALID_IMG_DIR   = os.path.join(DATA_ROOT, "valid", "images")
VALID_LABEL_DIR = os.path.join(DATA_ROOT, "valid", "labels")
TEST_IMG_DIR    = os.path.join(DATA_ROOT, "test",  "images")
TEST_LABEL_DIR  = os.path.join(DATA_ROOT, "test",  "labels")

# Hocmay dataset classes
CLASS_NAMES = ["bike", "car", "cat", "cellphone", "dog", "person"]
NUM_CLASSES = len(CLASS_NAMES)  # C = 6

# ── YOLO Grid ────────────────────────────────────────────────────────────────
S = 7           # Grid size  (S x S)
B = 2           # Bounding boxes per cell
# Each cell predicts: B * 5 (x, y, w, h, conf) + C class probs
CELL_SIZE = B * 5 + NUM_CLASSES  # 14

# ── Image ────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 448  # Input resolution as per the paper

# ── Training (Section 2.2 of the paper) ──────────────────────────────────────
BATCH_SIZE = 64
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 3

# Learning rate schedule (paper Section 2.2):
#   First epochs: slowly raise from 1e-3 to 1e-2
#   75 epochs at 1e-2, 30 epochs at 1e-3, 30 epochs at 1e-4
LR_WARMUP_EPOCHS = 5
LR_INIT = 1e-3
LR_MAIN = 1e-2
LR_SCHEDULE = [
    # (end_epoch, lr)
    (LR_WARMUP_EPOCHS, LR_INIT),   # warm-up: linearly ramp 1e-3 → 1e-2
    (LR_WARMUP_EPOCHS + 75, LR_MAIN),   # 75 epochs at 1e-2
    (LR_WARMUP_EPOCHS + 75 + 30, 1e-3), # 30 epochs at 1e-3
    (LR_WARMUP_EPOCHS + 75 + 30 + 30, 1e-4),  # 30 epochs at 1e-4  (total ~140 but we cap at 135)
]

# Dropout rate after first FC layer
DROPOUT_RATE = 0.5

# ── Loss weights (Section 2.2) ───────────────────────────────────────────────
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5

# ── Data augmentation (Section 2.2) ──────────────────────────────────────────
AUG_SCALE_RANGE = (0.8, 1.2)   # random scaling up to 20%
AUG_TRANSLATE = 0.2             # random translations up to 20% of image size
AUG_HSV_EXPOSURE = 1.5          # exposure adjustment factor
AUG_HSV_SATURATION = 1.5        # saturation adjustment factor
AUG_FLIP_PROB = 0.5             # horizontal flip probability

# ── Inference ────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.20
NMS_THRESHOLD = 0.5

# ── Paths ────────────────────────────────────────────────────────────────────
if os.path.exists('/content'):
    CHECKPOINT_DIR = '/content/checkpoints'
    LOG_DIR        = '/content/logs'
elif os.path.exists('/kaggle/working'):
    CHECKPOINT_DIR = '/kaggle/working/checkpoints'
    LOG_DIR        = '/kaggle/working/logs'
else:
    CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    LOG_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Device
DEVICE = "cuda"  # will fall back to cpu in code if not available
