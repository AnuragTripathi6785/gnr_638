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
    def __init__(self, img_dir, label_dir, augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.augment = augment
        self.S = S
        self.B = B
        self.C = NUM_CLASSES
        self.img_size = IMAGE_SIZE

        self.img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

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

        if self.augment:
            img, boxes = self._augment(img, boxes)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        target = self._encode_target(boxes)
        return img_tensor, target

    def _augment(self, img, boxes):
        img_np = np.array(img, dtype=np.uint8)
        h, w = img_np.shape[:2]

        # Horizontal flip
        if random.random() < AUG_FLIP_PROB:
            img_np = np.fliplr(img_np).copy()
            if len(boxes) > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1]

        # Scale and translate
        scale = random.uniform(AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1])
        tx = random.uniform(-AUG_TRANSLATE, AUG_TRANSLATE)
        ty = random.uniform(-AUG_TRANSLATE, AUG_TRANSLATE)

        new_w = int(w * scale)
        new_h = int(h * scale)
        img_pil = Image.fromarray(img_np)
        img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (w, h), (128, 128, 128))
        paste_x = int(tx * w) + (w - new_w) // 2
        paste_y = int(ty * h) + (h - new_h) // 2
        canvas.paste(img_pil, (paste_x, paste_y))

        if len(boxes) > 0:
            new_cx = (boxes[:, 1] * new_w + paste_x) / w
            new_cy = (boxes[:, 2] * new_h + paste_y) / h
            new_bw = boxes[:, 3] * scale
            new_bh = boxes[:, 4] * scale

            boxes[:, 1] = new_cx
            boxes[:, 2] = new_cy
            boxes[:, 3] = new_bw
            boxes[:, 4] = new_bh

            valid = (boxes[:, 1] > 0.01) & (boxes[:, 1] < 0.99) & \
                    (boxes[:, 2] > 0.01) & (boxes[:, 2] < 0.99) & \
                    (boxes[:, 3] > 0.005) & (boxes[:, 4] > 0.005)
            boxes = boxes[valid]

            if len(boxes) > 0:
                boxes[:, 3] = np.clip(boxes[:, 3], 0, 1)
                boxes[:, 4] = np.clip(boxes[:, 4], 0, 1)

        img_np = np.array(canvas, dtype=np.uint8)

        # HSV jitter
        import matplotlib.colors as mcolors
        img_float = img_np.astype(np.float32) / 255.0
        img_hsv = mcolors.rgb_to_hsv(img_float)

        sat_factor = random.uniform(1.0 / AUG_HSV_SATURATION, AUG_HSV_SATURATION)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * sat_factor, 0, 1)

        exp_factor = random.uniform(1.0 / AUG_HSV_EXPOSURE, AUG_HSV_EXPOSURE)
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * exp_factor, 0, 1)

        img = Image.fromarray((mcolors.hsv_to_rgb(img_hsv) * 255).astype(np.uint8))

        return img, boxes

    def _encode_target(self, boxes):
        """Encode boxes into (S, S, B*5+C) target tensor."""
        target = torch.zeros(self.S, self.S, self.B * 5 + self.C)

        for box in boxes:
            cls_id = int(box[0])
            cx, cy, bw, bh = box[1], box[2], box[3], box[4]

            gi = min(int(cx * self.S), self.S - 1)
            gj = min(int(cy * self.S), self.S - 1)

            x_offset = cx * self.S - gi
            y_offset = cy * self.S - gj

            target[gj, gi, 0] = float(x_offset)
            target[gj, gi, 1] = float(y_offset)
            target[gj, gi, 2] = float(bw)
            target[gj, gi, 3] = float(bh)
            target[gj, gi, 4] = 1.0

            class_start = self.B * 5
            target[gj, gi, class_start:class_start + self.C] = 0
            if cls_id < self.C:
                target[gj, gi, class_start + cls_id] = 1.0

        return target


if __name__ == "__main__":
    from config import TRAIN_IMG_DIR, TRAIN_LABEL_DIR
    ds = YOLODataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, augment=True)
    print(f"Dataset size: {len(ds)}")
    img, target = ds[0]
    print(f"Image: {img.shape}  Target: {target.shape}")
    print(f"Objects: {(target[:, :, 4] > 0).sum().item()}")
