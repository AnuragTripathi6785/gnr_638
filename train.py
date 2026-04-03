"""
YOLOv1 Training Script — follows Section 2.2 of the CVPR 2016 paper.

Training details:
  - SGD with momentum 0.9, weight decay 0.0005
  - Batch size 64
  - LR schedule: warm up 1e-3 → 1e-2 over first few epochs,
    then 1e-2 for 75 epochs, 1e-3 for 30, 1e-4 for 30
  - Dropout 0.5 after first FC layer
  - ~135 epochs total on PASCAL VOC train+val 2007 & 2012
"""

import os
import sys
import time

import torch
from torch.utils.data import DataLoader

from config import (
    TRAIN_IMG_DIR, TRAIN_LABEL_DIR,
    VALID_IMG_DIR, VALID_LABEL_DIR,
    BATCH_SIZE, MOMENTUM, WEIGHT_DECAY, NUM_EPOCHS,
    LR_WARMUP_EPOCHS, LR_INIT, LR_MAIN, LR_SCHEDULE,
    CHECKPOINT_DIR, DEVICE, IMAGE_SIZE, S, B, NUM_CLASSES,
)
from model import YOLOv1
from loss import YOLOv1Loss
from dataset import YOLODataset
from utils import decode_predictions, nms, compute_map, get_gt_boxes_for_eval


def get_lr(epoch):
    """
    Learning rate schedule from the paper (Section 2.2).

    First epochs: linearly ramp from 1e-3 to 1e-2.
    Then: 1e-2 for 75 epochs, 1e-3 for 30 epochs, 1e-4 for 30 epochs.
    """
    if epoch < LR_WARMUP_EPOCHS:
        # Linear warm-up
        alpha = epoch / max(LR_WARMUP_EPOCHS, 1)
        return LR_INIT + alpha * (LR_MAIN - LR_INIT)

    for end_epoch, lr in LR_SCHEDULE:
        if epoch < end_epoch:
            return lr

    return LR_SCHEDULE[-1][1]


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(images)

        # Compute loss (Eq. 3 from the paper)
        loss = criterion(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"  Epoch [{epoch+1}] Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}", flush=True)

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate and compute loss + mAP."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_pred_boxes = []
    all_gt_boxes = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)
        loss = criterion(predictions, targets)
        total_loss += loss.item()
        num_batches += 1

        # Decode for mAP
        pred_dets = decode_predictions(predictions.cpu(), conf_thresh=0.001)
        gt_dets = get_gt_boxes_for_eval(targets.cpu())

        # Apply NMS to predictions
        for i in range(len(pred_dets)):
            pred_dets[i] = nms(pred_dets[i])

        all_pred_boxes.extend(pred_dets)
        all_gt_boxes.extend(gt_dets)

    avg_loss = total_loss / max(num_batches, 1)
    mAP, per_class_ap = compute_map(all_pred_boxes, all_gt_boxes)

    return avg_loss, mAP, per_class_ap


def main():
    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = YOLODataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, augment=True)
    valid_dataset = YOLODataset(VALID_IMG_DIR, VALID_LABEL_DIR, augment=False)

    print(f"Train images: {len(train_dataset)}")
    print(f"Valid images: {len(valid_dataset)}")

    # Adjust batch size if dataset is small
    effective_batch = min(BATCH_SIZE, len(train_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch, shuffle=True,
        num_workers=2, pin_memory=(device.type == 'cuda'), drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=effective_batch, shuffle=False,
        num_workers=2, pin_memory=(device.type == 'cuda'),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = YOLOv1(num_classes=NUM_CLASSES, num_boxes=B, grid_size=S).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"YOLOv1 parameters: {total_params:,}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = YOLOv1Loss()

    # ── Optimizer (SGD as per the paper) ─────────────────────────────────────
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR_INIT,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    # ── Checkpoints ──────────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_mAP = 0.0

    # ── Training Loop ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Starting YOLOv1 training for {NUM_EPOCHS} epochs")
    print(f"{'='*60}\n")

    for epoch in range(NUM_EPOCHS):
        t_start = time.time()

        # Update learning rate (manual schedule as in the paper)
        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device, epoch)

        # Validate every 5 epochs (or last epoch)
        val_loss, mAP, per_class_ap = 0.0, 0.0, {}
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            val_loss, mAP, per_class_ap = validate(
                model, valid_loader, criterion, device
            )

        elapsed = time.time() - t_start

        # Print summary
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] "
              f"LR: {lr:.6f}  "
              f"Train Loss: {train_loss:.4f}  "
              f"Val Loss: {val_loss:.4f}  "
              f"mAP: {mAP:.4f}  "
              f"Time: {elapsed:.1f}s")

        if per_class_ap:
            for cls_name, ap in per_class_ap.items():
                print(f"  {cls_name}: AP={ap:.4f}")
            print()

        # Save best model
        if mAP > best_mAP:
            best_mAP = mAP
            ckpt_path = os.path.join(CHECKPOINT_DIR, "yolov1_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': mAP,
                'train_loss': train_loss,
            }, ckpt_path)
            print(f"  >> Saved best model (mAP={mAP:.4f})")

        # Save periodic checkpoints
        if (epoch + 1) % 25 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"yolov1_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': mAP,
                'train_loss': train_loss,
            }, ckpt_path)

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "yolov1_final.pth")
    torch.save({
        'epoch': NUM_EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mAP': mAP,
        'train_loss': train_loss,
    }, final_path)

    print(f"\n{'='*60}")
    print(f"Training complete!  Best mAP: {best_mAP:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
