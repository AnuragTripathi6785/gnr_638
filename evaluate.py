import os
import argparse

import torch
from torch.utils.data import DataLoader

from config import (
    VALID_IMG_DIR, VALID_LABEL_DIR,
    TEST_IMG_DIR, TEST_LABEL_DIR,
    S, B, NUM_CLASSES, CLASS_NAMES,
    CONF_THRESHOLD, NMS_THRESHOLD,
    CHECKPOINT_DIR, DEVICE, BATCH_SIZE,
)
from model import YOLOv1
from dataset import YOLODataset
from loss import YOLOv1Loss
from utils import decode_predictions, nms, compute_map, get_gt_boxes_for_eval


def evaluate(model, dataloader, criterion, device,
             conf_thresh=CONF_THRESHOLD, nms_thresh=NMS_THRESHOLD):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_pred_boxes = []
    all_gt_boxes = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1

            pred_dets = decode_predictions(predictions.cpu(), conf_thresh=conf_thresh)
            gt_dets = get_gt_boxes_for_eval(targets.cpu())

            for i in range(len(pred_dets)):
                pred_dets[i] = nms(pred_dets[i], iou_threshold=nms_thresh)

            all_pred_boxes.extend(pred_dets)
            all_gt_boxes.extend(gt_dets)

    avg_loss = total_loss / max(num_batches, 1)
    mAP, per_class_ap = compute_map(all_pred_boxes, all_gt_boxes)

    return avg_loss, mAP, per_class_ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(CHECKPOINT_DIR, "yolov1_best.pth"))
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--conf-thresh", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--nms-thresh", type=float, default=NMS_THRESHOLD)
    args = parser.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    if args.split == "valid":
        img_dir, label_dir = VALID_IMG_DIR, VALID_LABEL_DIR
    else:
        img_dir, label_dir = TEST_IMG_DIR, TEST_LABEL_DIR

    dataset = YOLODataset(img_dir, label_dir, augment=False)
    loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)),
                        shuffle=False, num_workers=2)

    print(f"Evaluating on {args.split} set ({len(dataset)} images)")

    model = YOLOv1(num_classes=NUM_CLASSES, num_boxes=B, grid_size=S).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    criterion = YOLOv1Loss()

    avg_loss, mAP, per_class_ap = evaluate(
        model, loader, criterion, device, args.conf_thresh, args.nms_thresh
    )

    print(f"\nResults on {args.split} set:")
    print(f"  Loss:    {avg_loss:.4f}")
    print(f"  mAP@0.5: {mAP:.4f}")
    print(f"\nPer-class AP:")
    for cls_name, ap in per_class_ap.items():
        print(f"  {cls_name:15s}: {ap:.4f}")


if __name__ == "__main__":
    main()
