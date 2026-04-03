"""
YOLOv1 Inference / Detection Script.

Loads a trained model and runs detection on images.
Implements the full inference pipeline:
  1. Resize input to 448×448
  2. Forward pass through the network
  3. Decode the S×S×(B*5+C) output tensor
  4. Apply Non-Maximum Suppression
  5. Draw bounding boxes and save results
"""

import os
import sys
import argparse

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    IMAGE_SIZE, S, B, NUM_CLASSES, CLASS_NAMES,
    CONF_THRESHOLD, NMS_THRESHOLD, CHECKPOINT_DIR, DEVICE,
    TEST_IMG_DIR,
)
from model import YOLOv1
from utils import decode_predictions, nms


# Colors for each class (RGBA-ish for visibility)
COLORS = [
    (255, 0, 0),       # bike - red
    (0, 255, 0),       # car - green
    (0, 0, 255),       # person - blue
    (255, 255, 0),     # tree - yellow
]


def load_model(checkpoint_path, device):
    """Load a trained YOLOv1 model from checkpoint."""
    model = YOLOv1(num_classes=NUM_CLASSES, num_boxes=B, grid_size=S).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    if 'mAP' in checkpoint:
        print(f"  Checkpoint mAP: {checkpoint['mAP']:.4f}")
    return model


def preprocess_image(img_path):
    """Load and preprocess a single image for inference."""
    img = Image.open(img_path).convert("RGB")
    orig_size = img.size  # (w, h)

    # Resize to 448×448
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    # Convert to tensor
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    return img_tensor, img, orig_size


def draw_detections(image, detections, orig_size):
    """
    Draw bounding boxes on the original image.

    Args:
        image: PIL Image (original size)
        detections: tensor (num_dets, 7) after NMS
                    [x1, y1, x2, y2, obj_conf, class_conf, class_id]
        orig_size: (w, h) of original image
    """
    draw = ImageDraw.Draw(image)
    w, h = orig_size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    if detections.numel() == 0:
        return image

    for det in detections:
        x1 = det[0].item() * w
        y1 = det[1].item() * h
        x2 = det[2].item() * w
        y2 = det[3].item() * h
        obj_conf = det[4].item()
        class_conf = det[5].item()
        class_id = int(det[6].item())

        color = COLORS[class_id % len(COLORS)]
        label = f"{CLASS_NAMES[class_id]} {class_conf:.2f}"

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 16), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 16), label, fill=(255, 255, 255), font=font)

    return image


def detect_single(model, img_path, device, conf_thresh=CONF_THRESHOLD,
                  nms_thresh=NMS_THRESHOLD):
    """
    Run detection on a single image.

    Returns:
        detections: tensor (num_dets, 7) after NMS
        image: PIL Image with drawn detections
    """
    img_tensor, orig_img, orig_size = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    # Decode predictions
    dets = decode_predictions(predictions.cpu(), conf_thresh=conf_thresh)
    dets = nms(dets[0], iou_threshold=nms_thresh)

    # Draw on original image
    result_img = draw_detections(orig_img.copy(), dets, orig_size)

    return dets, result_img


def detect_folder(model, img_dir, output_dir, device,
                  conf_thresh=CONF_THRESHOLD, nms_thresh=NMS_THRESHOLD):
    """Run detection on all images in a folder."""
    os.makedirs(output_dir, exist_ok=True)

    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"Running detection on {len(img_files)} images...")

    for i, img_name in enumerate(img_files):
        img_path = os.path.join(img_dir, img_name)
        dets, result_img = detect_single(model, img_path, device,
                                          conf_thresh, nms_thresh)

        out_path = os.path.join(output_dir, img_name)
        result_img.save(out_path)

        n_dets = dets.size(0) if dets.numel() > 0 else 0
        if (i + 1) % 10 == 0 or (i + 1) == len(img_files):
            print(f"  [{i+1}/{len(img_files)}] {img_name} — {n_dets} detections")


def main():
    parser = argparse.ArgumentParser(description="YOLOv1 Detection")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(CHECKPOINT_DIR, "yolov1_best.pth"),
                        help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image")
    parser.add_argument("--img-dir", type=str, default=TEST_IMG_DIR,
                        help="Directory of images to process")
    parser.add_argument("--output-dir", type=str, default="detections",
                        help="Directory to save detection results")
    parser.add_argument("--conf-thresh", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--nms-thresh", type=float, default=NMS_THRESHOLD)
    args = parser.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)

    if args.image:
        dets, result_img = detect_single(model, args.image, device,
                                          args.conf_thresh, args.nms_thresh)
        out_path = args.image.replace('.', '_det.')
        result_img.save(out_path)
        n_dets = dets.size(0) if dets.numel() > 0 else 0
        print(f"Detected {n_dets} objects → saved to {out_path}")
    else:
        detect_folder(model, args.img_dir, args.output_dir, device,
                      args.conf_thresh, args.nms_thresh)


if __name__ == "__main__":
    main()
