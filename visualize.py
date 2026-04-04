import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import S, B, NUM_CLASSES, CLASS_NAMES, IMAGE_SIZE


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (161, 56, 255),
    (56, 255, 255),
]


def draw_grid(image, S=7):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    cell_w = w / S
    cell_h = h / S

    for i in range(1, S):
        x = int(i * cell_w)
        draw.line([(x, 0), (x, h)], fill=(200, 200, 200), width=1)
        y = int(i * cell_h)
        draw.line([(0, y), (w, y)], fill=(200, 200, 200), width=1)

    return image


def draw_target_boxes(image, target, S=7):
    draw = ImageDraw.Draw(image)
    w, h = image.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for j in range(S):
        for i in range(S):
            if target[j, i, 4] > 0.5:
                x_off = target[j, i, 0].item()
                y_off = target[j, i, 1].item()
                bw = target[j, i, 2].item()
                bh = target[j, i, 3].item()

                cx = (i + x_off) / S * w
                cy = (j + y_off) / S * h
                box_w = bw * w
                box_h = bh * h

                x1 = cx - box_w / 2
                y1 = cy - box_h / 2
                x2 = cx + box_w / 2
                y2 = cy + box_h / 2

                class_start = B * 5
                cls_probs = target[j, i, class_start:class_start + NUM_CLASSES]
                cls_id = cls_probs.argmax().item()
                color = COLORS[cls_id % len(COLORS)]

                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1, y1 - 14), CLASS_NAMES[cls_id], fill=color, font=font)

    return image


def visualize_sample(dataset, idx, output_path="sample_vis.jpg"):
    img_tensor, target = dataset[idx]
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img = draw_grid(img)
    img = draw_target_boxes(img, target)
    img.save(output_path)
    print(f"Saved to {output_path}")
    return img


if __name__ == "__main__":
    from dataset import YOLODataset
    from config import TRAIN_IMG_DIR, TRAIN_LABEL_DIR

    ds = YOLODataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, augment=False)
    for i in range(min(5, len(ds))):
        visualize_sample(ds, i, f"sample_vis_{i}.jpg")
