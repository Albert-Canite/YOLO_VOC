"""Bounding box helpers for YOLO decoding and evaluation."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> List[int]:
    keep: List[int] = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        current = idxs[0]
        keep.append(current.item())
        if idxs.numel() == 1:
            break
        ious = box_iou(boxes[current].unsqueeze(0), boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious <= iou_threshold]
    return keep


def decode_predictions(
    pred: torch.Tensor,
    anchors: List[Tuple[int, int]],
    num_classes: int,
    conf_threshold: float,
    nms_threshold: float,
    stride: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Decode model output to bounding boxes and scores for each batch item."""
    batch, num_anchors, grid, _, _ = pred.shape
    device = pred.device
    pred = pred.clone()
    grid_y, grid_x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    grid_x = grid_x.to(device)
    grid_y = grid_y.to(device)

    pred[..., 0] = (pred[..., 0].sigmoid() + grid_x) * stride
    pred[..., 1] = (pred[..., 1].sigmoid() + grid_y) * stride
    pred[..., 2] = torch.exp(pred[..., 2]) * torch.tensor([a[0] for a in anchors], device=device)[:, None, None]
    pred[..., 3] = torch.exp(pred[..., 3]) * torch.tensor([a[1] for a in anchors], device=device)[:, None, None]
    pred[..., 4:] = pred[..., 4:].sigmoid()

    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []

    for b in range(batch):
        boxes = pred[b, :, :, :, :4].reshape(-1, 4)
        objectness = pred[b, :, :, :, 4].reshape(-1)
        class_scores = pred[b, :, :, :, 5:].reshape(-1, num_classes)
        scores, labels = class_scores.max(dim=1)
        scores = scores * objectness
        mask = scores > conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        if boxes.numel() == 0:
            all_boxes.append(np.zeros((0, 6)))
            all_scores.append(np.zeros((0,)))
            continue
        boxes_xyxy = xywh_to_xyxy(boxes)
        keep = nms(boxes_xyxy, scores, nms_threshold)
        keep_boxes = boxes_xyxy[keep]
        keep_scores = scores[keep]
        keep_labels = labels[keep]
        merged = torch.cat(
            [keep_boxes, keep_labels.float().unsqueeze(1), keep_scores.unsqueeze(1)], dim=1
        )
        all_boxes.append(merged.cpu().numpy())
        all_scores.append(keep_scores.detach().cpu().numpy())

    return all_boxes, all_scores


def mean_average_precision(
    preds: List[np.ndarray],
    targets: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> float:
    """Compute mean average precision at a fixed IoU threshold."""
    aps: List[float] = []
    for cls in range(num_classes):
        cls_preds = []
        cls_gts = 0
        for b_idx, pred in enumerate(preds):
            pred_cls = pred[pred[:, 4] == cls]
            gt_cls = targets[b_idx][targets[b_idx][:, 4] == cls]
            cls_gts += len(gt_cls)
            for det in pred_cls:
                cls_preds.append((b_idx, det[:4], det[4], det[5]))

        cls_preds.sort(key=lambda x: x[3], reverse=True)
        matched = {}
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        for i, (b_idx, box, _, score) in enumerate(cls_preds):
            gt_cls = targets[b_idx][targets[b_idx][:, 4] == cls]
            if len(gt_cls) == 0:
                fp[i] = 1
                continue
            ious = _box_iou_np(box, gt_cls[:, :4])
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            if max_iou >= iou_threshold and (b_idx, max_iou_idx) not in matched:
                tp[i] = 1
                matched[(b_idx, max_iou_idx)] = True
            else:
                fp[i] = 1

        if cls_gts == 0:
            continue
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / (cls_gts + 1e-6)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
        ap = _compute_ap(recalls, precisions)
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap


def _box_iou_np(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter + 1e-6
    return inter / union

