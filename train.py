import os
import time

import torch
from torch.utils.data import DataLoader

from config import (
    TRAIN_IMG_DIR, TRAIN_LABEL_DIR,
    VALID_IMG_DIR, VALID_LABEL_DIR,
    BATCH_SIZE, MOMENTUM, WEIGHT_DECAY, NUM_EPOCHS,
    LR_WARMUP_EPOCHS, LR_INIT, LR_MAIN, LR_SCHEDULE,
    CHECKPOINT_DIR, DEVICE, S, B, NUM_CLASSES,
)
from model import YOLOv1
from loss import YOLOv1Loss
from dataset import YOLODataset
from utils import decode_predictions, nms, compute_map, get_gt_boxes_for_eval


def get_lr(epoch):
    if epoch < LR_WARMUP_EPOCHS:
        alpha = epoch / max(LR_WARMUP_EPOCHS, 1)
        return LR_INIT + alpha * (LR_MAIN - LR_INIT)
    for end_epoch, lr in LR_SCHEDULE:
        if epoch < end_epoch:
            return lr
    return LR_SCHEDULE[-1][1]


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)
        loss = criterion(predictions, targets)

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

        pred_dets = decode_predictions(predictions.cpu(), conf_thresh=0.001)
        gt_dets = get_gt_boxes_for_eval(targets.cpu())

        for i in range(len(pred_dets)):
            pred_dets[i] = nms(pred_dets[i])

        all_pred_boxes.extend(pred_dets)
        all_gt_boxes.extend(gt_dets)

    avg_loss = total_loss / max(num_batches, 1)
    mAP, per_class_ap = compute_map(all_pred_boxes, all_gt_boxes)

    return avg_loss, mAP, per_class_ap


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = YOLODataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, augment=True)
    valid_dataset = YOLODataset(VALID_IMG_DIR, VALID_LABEL_DIR, augment=False)

    print(f"Train: {len(train_dataset)}  Valid: {len(valid_dataset)}")

    effective_batch = min(BATCH_SIZE, len(train_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch, shuffle=True,
        num_workers=2, pin_memory=(device.type == 'cuda'), drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=effective_batch, shuffle=False,
        num_workers=2, pin_memory=(device.type == 'cuda'),
    )

    model = YOLOv1(num_classes=NUM_CLASSES, num_boxes=B, grid_size=S).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = YOLOv1Loss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR_INIT,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_mAP = 0.0

    print(f"\nStarting training for {NUM_EPOCHS} epochs\n")

    for epoch in range(NUM_EPOCHS):
        t_start = time.time()

        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        val_loss, mAP, per_class_ap = 0.0, 0.0, {}
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            val_loss, mAP, per_class_ap = validate(model, valid_loader, criterion, device)

        elapsed = time.time() - t_start

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] LR: {lr:.6f}  "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  "
              f"mAP: {mAP:.4f}  Time: {elapsed:.1f}s")

        if per_class_ap:
            for cls_name, ap in per_class_ap.items():
                print(f"  {cls_name}: {ap:.4f}")
            print()

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
            print(f"  Saved best model (mAP={mAP:.4f})")

        if (epoch + 1) % 25 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"yolov1_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': mAP,
                'train_loss': train_loss,
            }, ckpt_path)

    final_path = os.path.join(CHECKPOINT_DIR, "yolov1_final.pth")
    torch.save({
        'epoch': NUM_EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mAP': mAP,
        'train_loss': train_loss,
    }, final_path)

    print(f"\nTraining complete. Best mAP: {best_mAP:.4f}")


if __name__ == "__main__":
    main()
