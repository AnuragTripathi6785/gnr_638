import torch
import torch.nn as nn

from config import S, B, NUM_CLASSES, LAMBDA_COORD, LAMBDA_NOOBJ


def compute_iou(box1, box2):
    b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
    b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
    b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
    b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

    b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
    b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
    b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
    b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    b1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))

    union_area = b1_area + b2_area - inter_area + 1e-6
    return inter_area / union_area


class YOLOv1Loss(nn.Module):
    """
    Multi-part loss from Eq. 3 of the YOLO paper.
    Target tensor layout per cell: [x, y, w, h, conf] * B + class_probs
    """

    def __init__(self, S=S, B=B, C=NUM_CLASSES,
                 lambda_coord=LAMBDA_COORD, lambda_noobj=LAMBDA_NOOBJ):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        N = predictions.size(0)

        pred_boxes = predictions[..., :self.B * 5].contiguous().view(
            N, self.S, self.S, self.B, 5)
        pred_class = predictions[..., self.B * 5:]

        tgt_boxes = targets[..., :self.B * 5].contiguous().view(
            N, self.S, self.S, self.B, 5)
        tgt_class = targets[..., self.B * 5:]

        obj_mask = tgt_boxes[..., 0, 4]  # (N, S, S)
        gt_box = tgt_boxes[:, :, :, 0, :4]  # (N, S, S, 4)

        cell_y, cell_x = torch.meshgrid(
            torch.arange(self.S, device=predictions.device, dtype=torch.float32),
            torch.arange(self.S, device=predictions.device, dtype=torch.float32),
            indexing='ij'
        )
        cell_x = cell_x.unsqueeze(0).unsqueeze(-1)
        cell_y = cell_y.unsqueeze(0).unsqueeze(-1)

        pred_xy = pred_boxes[..., :2]
        pred_wh = pred_boxes[..., 2:4]

        pred_x_abs = (cell_x.view(1, self.S, self.S, 1, 1) + pred_xy[..., 0:1]) / self.S
        pred_y_abs = (cell_y.view(1, self.S, self.S, 1, 1) + pred_xy[..., 1:2]) / self.S
        pred_boxes_abs = torch.cat([pred_x_abs, pred_y_abs, pred_wh], dim=-1)

        gt_x = gt_box[..., 0:1]
        gt_y = gt_box[..., 1:2]
        gt_wh = gt_box[..., 2:4]

        gt_x_abs = (cell_x + gt_x) / self.S
        gt_y_abs = (cell_y + gt_y) / self.S
        gt_box_abs = torch.cat([gt_x_abs, gt_y_abs, gt_wh], dim=-1)

        # Find responsible predictor (highest IOU with GT)
        ious = torch.zeros(N, self.S, self.S, self.B, device=predictions.device)
        for b in range(self.B):
            pred_b = pred_boxes_abs[:, :, :, b, :]
            iou_flat = compute_iou(pred_b.reshape(-1, 4), gt_box_abs.reshape(-1, 4))
            ious[:, :, :, b] = iou_flat.view(N, self.S, self.S)

        best_box = ious.argmax(dim=-1)

        responsible = torch.zeros_like(ious)
        responsible.scatter_(-1, best_box.unsqueeze(-1), 1.0)
        responsible = responsible * obj_mask.unsqueeze(-1)

        # Term 1 & 2: coordinate loss
        pred_xy_resp = pred_boxes[..., :2]
        gt_xy = gt_box[..., :2].unsqueeze(3).expand_as(pred_xy_resp)
        resp_mask = responsible.unsqueeze(-1)

        xy_loss = self.lambda_coord * torch.sum(
            resp_mask * (pred_xy_resp - gt_xy) ** 2
        )

        pred_wh_resp = pred_boxes[..., 2:4]
        gt_wh_exp = gt_wh.unsqueeze(3).expand_as(pred_wh_resp)

        pred_wh_sqrt = torch.sign(pred_wh_resp) * torch.sqrt(torch.abs(pred_wh_resp) + 1e-6)
        gt_wh_sqrt = torch.sqrt(gt_wh_exp + 1e-6)

        wh_loss = self.lambda_coord * torch.sum(
            resp_mask * (pred_wh_sqrt - gt_wh_sqrt) ** 2
        )

        # Term 3: confidence loss for responsible predictor
        # Target = 1.0 (not IOU) to avoid training deadlock early on
        pred_conf = pred_boxes[..., 4]
        obj_conf_loss = torch.sum(
            responsible * (pred_conf - responsible) ** 2
        )

        # Term 4: confidence loss for non-responsible predictors
        noobj_mask = 1.0 - responsible
        noobj_conf_loss = self.lambda_noobj * torch.sum(
            noobj_mask * pred_conf ** 2
        )

        # Term 5: class loss
        class_loss = torch.sum(
            obj_mask.unsqueeze(-1) * (pred_class - tgt_class) ** 2
        )

        total_loss = (xy_loss + wh_loss + obj_conf_loss + noobj_conf_loss + class_loss) / N
        return total_loss
