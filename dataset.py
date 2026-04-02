"""
YOLOv1 Dataset — loads images and YOLO-format labels, applies the data
augmentation described in Section 2.2 of the paper:
  - Random scaling and translations of up to 20% of the original image size
  - Randomly adjust exposure and saturation by up to a factor of 1.5 in HSV
  - Random horizontal flip

Labels are in YOLO format:  class_id  cx  cy  w  h  (all normalized 0-1)
We encode them into the (S, S, B*5 + C) target tensor that the loss expects.
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from config import (
    S, B, NUM_CLASSES, IMAGE_SIZE,
    AUG_SCALE_RANGE, AUG_TRANSLATE, AUG_HSV_EXPOSURE, AUG_HSV_SATURATION,
    AUG_FLIP_PROB,
)


class YOLODataset(Dataset):
    """
    Args:
        img_dir:   path to the images/ folder
        label_dir: path to the labels/ folder
        augment:   whether to apply data augmentation (True for training)
    """

    def __init__(self, img_dir, label_dir, augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.augment = augment
        self.S = S
        self.B = B
        self.C = NUM_CLASSES
        self.img_size = IMAGE_SIZE

        # Collect image files
        self.img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # ── Load image ───────────────────────────────────────────────────────
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # ── Load labels ──────────────────────────────────────────────────────
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = [float(x) for x in parts[1:]]
                        boxes.append([cls_id, cx, cy, w, h])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)

        # ── Data Augmentation (Section 2.2) ──────────────────────────────────
        if self.augment:
            img, boxes = self._augment(img, boxes)

        # ── Resize to 448×448 ────────────────────────────────────────────────
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # ── Convert to tensor ────────────────────────────────────────────────
        img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)

        # ── Encode target tensor ─────────────────────────────────────────────
        target = self._encode_target(boxes)

        return img_tensor, target

    def _augment(self, img, boxes):
        """Apply the data augmentations described in the paper."""
        img_np = np.array(img, dtype=np.uint8)
        h, w = img_np.shape[:2]

        # ── Random horizontal flip ───────────────────────────────────────────
        if random.random() < AUG_FLIP_PROB:
            img_np = np.fliplr(img_np).copy()
            if len(boxes) > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1]  # flip cx

        # ── Random scale and translate ───────────────────────────────────────
        scale = random.uniform(AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1])
        max_tx = AUG_TRANSLATE
        max_ty = AUG_TRANSLATE
        tx = random.uniform(-max_tx, max_tx)
        ty = random.uniform(-max_ty, max_ty)

        # Apply affine: scale around center, then translate
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_pil = Image.fromarray(img_np)
        img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)

        # Create canvas and paste
        canvas = Image.new("RGB", (w, h), (128, 128, 128))
        paste_x = int(tx * w) + (w - new_w) // 2
        paste_y = int(ty * h) + (h - new_h) // 2
        canvas.paste(img_pil, (paste_x, paste_y))

        # Adjust boxes
        if len(boxes) > 0:
            # Scale and translate box centers
            boxes[:, 1] = boxes[:, 1] * scale + (paste_x + new_w / 2 - w / 2) / w + (1 - scale) / 2
            boxes[:, 2] = boxes[:, 2] * scale + (paste_y + new_h / 2 - h / 2) / h + (1 - scale) / 2
            # Simpler: just apply the transform
            # Actually let's redo this properly:
            # The box center in pixel coords was: cx_px = boxes[:,1] * w,  cy_px = boxes[:,2] * h
            # After scaling and pasting:
            #   new_cx_px = boxes[:,1] * new_w + paste_x
            #   new_cy_px = boxes[:,2] * new_h + paste_y
            # Normalized: new_cx = new_cx_px / w
            new_cx = (boxes[:, 1] * new_w + paste_x) / w
            new_cy = (boxes[:, 2] * new_h + paste_y) / h
            new_bw = boxes[:, 3] * scale
            new_bh = boxes[:, 4] * scale

            boxes[:, 1] = new_cx
            boxes[:, 2] = new_cy
            boxes[:, 3] = new_bw
            boxes[:, 4] = new_bh

            # Clip and remove out-of-bounds boxes
            valid = (boxes[:, 1] > 0.01) & (boxes[:, 1] < 0.99) & \
                    (boxes[:, 2] > 0.01) & (boxes[:, 2] < 0.99) & \
                    (boxes[:, 3] > 0.005) & (boxes[:, 4] > 0.005)
            boxes = boxes[valid]

            # Clip w, h
            if len(boxes) > 0:
                boxes[:, 3] = np.clip(boxes[:, 3], 0, 1)
                boxes[:, 4] = np.clip(boxes[:, 4], 0, 1)

        img_np = np.array(canvas, dtype=np.uint8)

        # ── Random HSV jitter (exposure & saturation) ────────────────────────
        img_hsv = Image.fromarray(img_np).convert("HSV")
        img_hsv_np = np.array(img_hsv, dtype=np.float32)

        # Saturation channel (index 1)
        sat_factor = random.uniform(1.0 / AUG_HSV_SATURATION, AUG_HSV_SATURATION)
        img_hsv_np[:, :, 1] = np.clip(img_hsv_np[:, :, 1] * sat_factor, 0, 255)

        # Value/exposure channel (index 2)
        exp_factor = random.uniform(1.0 / AUG_HSV_EXPOSURE, AUG_HSV_EXPOSURE)
        img_hsv_np[:, :, 2] = np.clip(img_hsv_np[:, :, 2] * exp_factor, 0, 255)

        img_hsv = Image.fromarray(img_hsv_np.astype(np.uint8), "HSV")
        img = img_hsv.convert("RGB")

        return img, boxes

    def _encode_target(self, boxes):
        """
        Encode ground-truth boxes into the (S, S, B*5 + C) target tensor.

        For each object, we determine which grid cell its center falls into.
        We fill in the first box slot with (x_offset, y_offset, w, h, 1)
        and set the class probabilities to a one-hot vector.

        If multiple objects fall in the same cell, only the last one is kept
        (a known limitation of YOLOv1).
        """
        target = torch.zeros(self.S, self.S, self.B * 5 + self.C)

        for box in boxes:
            cls_id = int(box[0])
            cx, cy, bw, bh = box[1], box[2], box[3], box[4]

            # Which grid cell?
            gi = int(cx * self.S)  # column
            gj = int(cy * self.S)  # row
            gi = min(gi, self.S - 1)
            gj = min(gj, self.S - 1)

            # Offset within the cell (0..1)
            x_offset = cx * self.S - gi
            y_offset = cy * self.S - gj

            # Fill first box slot: (x, y, w, h, obj_flag)
            target[gj, gi, 0] = x_offset
            target[gj, gi, 1] = y_offset
            target[gj, gi, 2] = bw
            target[gj, gi, 3] = bh
            target[gj, gi, 4] = 1.0  # object present

            # Second box slot (indices 5-9) left as zeros in target
            # (the loss will handle responsible predictor selection)

            # Class probabilities (one-hot)
            class_start = self.B * 5
            target[gj, gi, class_start:class_start + self.C] = 0  # reset
            if cls_id < self.C:
                target[gj, gi, class_start + cls_id] = 1.0

        return target


if __name__ == "__main__":
    from config import TRAIN_IMG_DIR, TRAIN_LABEL_DIR
    ds = YOLODataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, augment=True)
    print(f"Dataset size: {len(ds)}")
    img, target = ds[0]
    print(f"Image shape: {img.shape}")
    print(f"Target shape: {target.shape}")
    n_obj = (target[:, :, 4] > 0).sum().item()
    print(f"Objects in target: {n_obj}")
