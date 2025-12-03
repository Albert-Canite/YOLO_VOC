"""Metrics and loss helpers."""
from typing import List, Tuple

import torch
import torch.nn.functional as F


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute IoU for two sets of boxes in xyxy format."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=-1)

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = (inter_rect_x2 - inter_rect_x1).clamp(0) * (inter_rect_y2 - inter_rect_y1).clamp(0)
    b1_area = (b1_x2 - b1_x1).clamp(0) * (b1_y2 - b1_y1).clamp(0)
    b2_area = (b2_x2 - b2_x1).clamp(0) * (b2_y2 - b2_y1).clamp(0)
    union = b1_area + b2_area - inter_area + 1e-6
    return inter_area / union


def yolo_loss(pred: torch.Tensor, targets: List[dict], anchors: torch.Tensor, num_classes: int, img_size: int):
    """Compute simplified YOLO loss (MSE for box, BCE for obj/class)."""
    device = pred.device
    grid_size = pred.size(2)
    stride = img_size / grid_size

    obj_mask = torch.zeros_like(pred[..., 0], device=device)
    noobj_mask = torch.ones_like(pred[..., 0], device=device)
    tx = torch.zeros_like(pred[..., 0], device=device)
    ty = torch.zeros_like(pred[..., 1], device=device)
    tw = torch.zeros_like(pred[..., 2], device=device)
    th = torch.zeros_like(pred[..., 3], device=device)
    tcls = torch.zeros((*pred.shape[:4], num_classes), device=device)

    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"]
        labels = target["labels"]
        if boxes.numel() == 0:
            continue
        boxes_xyxy = boxes
        boxes_xywh = torch.zeros_like(boxes_xyxy, device=device)
        boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
        boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        gxy = boxes_xywh[:, 0:2] / stride
        gwh = boxes_xywh[:, 2:4] / anchors
        ij = gxy.long()
        i, j = ij[:, 0], ij[:, 1]
        anchor_idxs = torch.argmax(gwh, dim=1)
        obj_mask[batch_idx, anchor_idxs, j, i] = 1
        noobj_mask[batch_idx, anchor_idxs, j, i] = 0
        tx[batch_idx, anchor_idxs, j, i] = gxy[:, 0] - i
        ty[batch_idx, anchor_idxs, j, i] = gxy[:, 1] - j
        tw[batch_idx, anchor_idxs, j, i] = torch.log(boxes_xywh[:, 2] / anchors[anchor_idxs, 0] + 1e-6)
        th[batch_idx, anchor_idxs, j, i] = torch.log(boxes_xywh[:, 3] / anchors[anchor_idxs, 1] + 1e-6)
        tcls[batch_idx, anchor_idxs, j, i, labels] = 1

    pred_x = torch.sigmoid(pred[..., 0])
    pred_y = torch.sigmoid(pred[..., 1])
    pred_w = pred[..., 2]
    pred_h = pred[..., 3]
    pred_obj = torch.sigmoid(pred[..., 4])
    pred_cls = pred[..., 5:]

    loss_x = F.mse_loss(pred_x * obj_mask, tx, reduction="sum")
    loss_y = F.mse_loss(pred_y * obj_mask, ty, reduction="sum")
    loss_w = F.mse_loss(pred_w * obj_mask, tw, reduction="sum")
    loss_h = F.mse_loss(pred_h * obj_mask, th, reduction="sum")
    loss_obj = F.binary_cross_entropy(pred_obj, obj_mask, reduction="sum")
    loss_cls = F.binary_cross_entropy_with_logits(pred_cls, tcls, reduction="sum")
    total = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_cls
    return total / pred.size(0)


def non_max_suppression(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        idx = order[0]
        keep.append(idx.item())
        if order.numel() == 1:
            break
        ious = bbox_iou(boxes[idx].unsqueeze(0), boxes[order[1:]])
        order = order[1:][ious.squeeze(0) < iou_threshold]
    return keep


def evaluate_map(
    predictions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    targets: List[dict],
    num_classes: int,
    iou_threshold: float = 0.5,
):
    tp_list: List[torch.Tensor] = []
    conf_list: List[torch.Tensor] = []
    cls_list: List[torch.Tensor] = []
    gt_counter = torch.zeros(num_classes)

    for pred, target in zip(predictions, targets):
        boxes, scores, labels = pred
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        gt_counter += torch.bincount(gt_labels, minlength=num_classes)
        if boxes.numel() == 0:
            continue
        order = scores.argsort(descending=True)
        boxes = boxes[order]
        labels = labels[order]
        scores = scores[order]
        detected = []
        for box, score, cls in zip(boxes, scores, labels):
            if gt_boxes.numel() == 0:
                tp_list.append(torch.tensor(0.0))
                conf_list.append(score)
                cls_list.append(cls)
                continue
            ious = bbox_iou(box.unsqueeze(0), gt_boxes)
            best_iou, best_idx = ious.max(dim=1)
            if best_iou.item() >= iou_threshold and best_idx.item() not in detected and gt_labels[best_idx] == cls:
                tp_list.append(torch.tensor(1.0))
                detected.append(best_idx.item())
            else:
                tp_list.append(torch.tensor(0.0))
            conf_list.append(score)
            cls_list.append(cls)

    if not conf_list:
        return 0.0
    conf_tensor = torch.stack(conf_list)
    tp_tensor = torch.stack(tp_list)
    cls_tensor = torch.stack(cls_list)
    ap_per_class = []
    for c in range(num_classes):
        mask = cls_tensor == c
        if mask.sum() == 0 or gt_counter[c] == 0:
            continue
        scores_c = conf_tensor[mask]
        tp_c = tp_tensor[mask]
        sorted_idx = scores_c.argsort(descending=True)
        tp_c = tp_c[sorted_idx]
        fp_c = 1 - tp_c
        tp_c = tp_c.cumsum(0)
        fp_c = fp_c.cumsum(0)
        recalls = tp_c / (gt_counter[c] + 1e-6)
        precisions = tp_c / (tp_c + fp_c + 1e-6)
        ap = torch.trapz(precisions, recalls).item()
        ap_per_class.append(ap)
    if not ap_per_class:
        return 0.0
    return float(sum(ap_per_class) / len(ap_per_class))
