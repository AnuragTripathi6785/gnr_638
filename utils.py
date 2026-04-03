"""
Utility functions for YOLOv1:
  - Decode predictions to bounding boxes
  - Non-Maximum Suppression (NMS)
  - Mean Average Precision (mAP) computation
  - Visualization helpers
"""

import torch
import numpy as np
from collections import defaultdict

from config import S, B, NUM_CLASSES, CLASS_NAMES, CONF_THRESHOLD, NMS_THRESHOLD


def decode_predictions(predictions, conf_thresh=CONF_THRESHOLD):
    """
    Decode the (N, S, S, B*5 + C) tensor into a list of detections.

    At test time, class-specific confidence = Pr(Class_i | Object) * IOU
    As per Eq. (1) in the paper:
        Pr(Class_i) * Pr(Object) * IOU = Pr(Class_i) * Confidence

    Args:
        predictions: (N, S, S, B*5+C) tensor
        conf_thresh: minimum class-specific confidence to keep

    Returns:
        all_detections: list of length N, where each element is a tensor
                        of shape (num_dets, 7):
                        [x1, y1, x2, y2, confidence, class_conf, class_id]
    """
    N = predictions.size(0)
    all_detections = []

    for n in range(N):
        pred = predictions[n]  # (S, S, B*5+C)
        detections = []

        for j in range(S):
            for i in range(S):
                cell = pred[j, i]

                # Class probabilities: Pr(Class_i | Object)
                class_probs = cell[B * 5:]  # (C,)

                for b in range(B):
                    box_start = b * 5
                    x_offset = cell[box_start + 0]
                    y_offset = cell[box_start + 1]
                    w = cell[box_start + 2]
                    h = cell[box_start + 3]
                    conf = cell[box_start + 4]  # Pr(Object) * IOU

                    # Class-specific confidence scores
                    class_scores = class_probs * conf  # Pr(Class_i) * IOU

                    # Best class
                    class_conf, class_id = class_scores.max(0)

                    if class_conf.item() < conf_thresh:
                        continue

                    # Convert to absolute image coordinates (0..1)
                    cx = (i + x_offset.item()) / S
                    cy = (j + y_offset.item()) / S
                    bw = abs(w.item())
                    bh = abs(h.item())

                    # Convert to corner format
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2

                    detections.append([x1, y1, x2, y2, conf.item(),
                                       class_conf.item(), class_id.item()])

        if detections:
            all_detections.append(torch.tensor(detections))
        else:
            all_detections.append(torch.zeros(0, 7))

    return all_detections


def nms(detections, iou_threshold=NMS_THRESHOLD):
    """
    Non-Maximum Suppression — per class.

    Args:
        detections: tensor (num_dets, 7)
                    [x1, y1, x2, y2, obj_conf, class_conf, class_id]
        iou_threshold: NMS IoU threshold

    Returns:
        filtered: tensor (num_kept, 7)
    """
    if detections.numel() == 0:
        return detections

    kept = []

    # Get unique classes
    classes = detections[:, 6].unique()

    for cls in classes:
        cls_mask = detections[:, 6] == cls
        cls_dets = detections[cls_mask]

        # Sort by class confidence (descending)
        scores = cls_dets[:, 5]
        order = scores.argsort(descending=True)
        cls_dets = cls_dets[order]

        keep = []
        while cls_dets.size(0) > 0:
            keep.append(cls_dets[0])
            if cls_dets.size(0) == 1:
                break

            # Compute IoU of the best box with all remaining
            ious = _box_iou(cls_dets[0, :4].unsqueeze(0),
                            cls_dets[1:, :4])  # (remaining,)
            # Keep boxes with IoU < threshold
            mask = ious < iou_threshold
            cls_dets = cls_dets[1:][mask]

        kept.extend(keep)

    if kept:
        return torch.stack(kept)
    return torch.zeros(0, 7)


def _box_iou(box1, box2):
    """
    IoU between one box and multiple boxes (corner format x1,y1,x2,y2).
    box1: (1, 4)  box2: (M, 4)
    Returns: (M,)
    """
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = area1 + area2 - inter_area + 1e-6
    return inter_area / union


def compute_ap(recall, precision):
    """
    Compute Average Precision using the 11-point interpolation method
    (standard for PASCAL VOC).
    """
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max()
    return ap / 11.0


def compute_map(all_pred_boxes, all_gt_boxes, iou_threshold=0.5, num_classes=NUM_CLASSES):
    """
    Compute mean Average Precision across all classes.

    Args:
        all_pred_boxes: list of (num_preds, 7) tensors per image
                        [x1, y1, x2, y2, obj_conf, class_conf, class_id]
        all_gt_boxes:   list of (num_gts, 6) tensors per image
                        [x1, y1, x2, y2, class_id, difficult(0/1)]

    Returns:
        mAP: float
        per_class_ap: dict {class_name: ap}
    """
    # Gather all predictions and ground truths per class
    pred_per_class = defaultdict(list)
    gt_per_class = defaultdict(list)
    n_gt_per_class = defaultdict(int)

    for img_idx, (preds, gts) in enumerate(zip(all_pred_boxes, all_gt_boxes)):
        if preds.numel() > 0:
            for p in preds:
                cls = int(p[6].item())
                pred_per_class[cls].append({
                    'img_idx': img_idx,
                    'conf': p[5].item(),
                    'box': p[:4],
                })

        if gts.numel() > 0:
            for g in gts:
                cls = int(g[4].item())
                gt_per_class[cls].append({
                    'img_idx': img_idx,
                    'box': g[:4],
                    'matched': False,
                })
                n_gt_per_class[cls] += 1

    aps = {}
    for cls in range(num_classes):
        preds = pred_per_class.get(cls, [])
        gts_list = gt_per_class.get(cls, [])
        n_gt = n_gt_per_class.get(cls, 0)

        if n_gt == 0:
            aps[cls] = 0.0
            continue

        # Sort predictions by confidence
        preds = sorted(preds, key=lambda x: x['conf'], reverse=True)

        # Reset matched flags
        for g in gts_list:
            g['matched'] = False

        # Group gts by image for efficient lookup
        gts_by_img = defaultdict(list)
        for g in gts_list:
            gts_by_img[g['img_idx']].append(g)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for i, pred in enumerate(preds):
            img_gts = gts_by_img.get(pred['img_idx'], [])

            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(img_gts):
                iou = _box_iou(
                    pred['box'].unsqueeze(0),
                    gt['box'].unsqueeze(0)
                ).item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                if not img_gts[best_gt_idx]['matched']:
                    tp[i] = 1
                    img_gts[best_gt_idx]['matched'] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        # Compute precision/recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / (n_gt + 1e-6)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        aps[cls] = compute_ap(recall, precision)

    per_class_ap = {CLASS_NAMES[c]: aps[c] for c in range(num_classes)}
    mAP = np.mean(list(aps.values()))

    return mAP, per_class_ap


def get_gt_boxes_for_eval(targets):
    """
    Convert target tensors (N, S, S, B*5+C) back to bounding boxes for
    mAP evaluation.

    Returns: list of (num_gts, 6) tensors  [x1, y1, x2, y2, class_id, 0]
    """
    N = targets.size(0)
    all_boxes = []

    for n in range(N):
        target = targets[n]
        boxes = []
        for j in range(S):
            for i in range(S):
                if target[j, i, 4] > 0.5:  # object present
                    x_off = target[j, i, 0].item()
                    y_off = target[j, i, 1].item()
                    w = target[j, i, 2].item()
                    h = target[j, i, 3].item()

                    cx = (i + x_off) / S
                    cy = (j + y_off) / S

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    # Class
                    class_start = B * 5
                    cls_probs = target[j, i, class_start:class_start + NUM_CLASSES]
                    cls_id = cls_probs.argmax().item()

                    boxes.append([x1, y1, x2, y2, cls_id, 0])

        if boxes:
            all_boxes.append(torch.tensor(boxes))
        else:
            all_boxes.append(torch.zeros(0, 6))

    return all_boxes
