"""Bounding box helpers, decoding, and evaluation metrics."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


Box = Tuple[float, float, float, float]


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) boxes to (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) boxes to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for two sets of boxes (N,4) and (M,4)."""
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def wh_iou(anchor_wh: torch.Tensor, gt_wh: torch.Tensor) -> torch.Tensor:
    """IoU between anchor widths/heights and ground truth widths/heights (shape: A x 2, G x 2)."""
    anchor_boxes = torch.cat([
        torch.zeros_like(anchor_wh),
        anchor_wh,
    ], dim=1)
    gt_boxes = torch.cat([
        torch.zeros_like(gt_wh),
        gt_wh,
    ], dim=1)
    return compute_iou(anchor_boxes, gt_boxes)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> List[int]:
    """Perform Non-Maximum Suppression returning indices of kept boxes."""
    keep: List[int] = []
    order = scores.argsort(descending=True)
    while order.numel() > 0:
        idx = order[0]
        keep.append(idx.item())
        if order.numel() == 1:
            break
        ious = compute_iou(boxes[idx].unsqueeze(0), boxes[order[1:]])[0]
        mask = ious <= iou_threshold
        order = order[1:][mask]
    return keep


def decode_predictions(
    preds: torch.Tensor,
    anchors: Sequence[Tuple[int, int]],
    num_classes: int,
    conf_threshold: float,
    nms_threshold: float,
    stride: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Decode raw model outputs into bounding boxes and class scores.

    Returns a tuple containing a list of arrays for boxes and scores per image.
    Each boxes array shape: (N, 6) [x1, y1, x2, y2, score, cls].
    """
    b, na, gs, _, _ = preds.shape
    device = preds.device
    preds = preds.detach()

    grid_y, grid_x = torch.meshgrid(
        torch.arange(gs, device=device), torch.arange(gs, device=device), indexing="ij"
    )
    outputs: List[np.ndarray] = []
    scores_out: List[np.ndarray] = []
    for img_idx in range(b):
        image_boxes: List[List[float]] = []
        image_scores: List[List[float]] = []
        pred = preds[img_idx]
        obj = torch.sigmoid(pred[..., 4])
        cls_logits = pred[..., 5:]
        cls_scores = torch.sigmoid(cls_logits)
        for a_idx, (aw, ah) in enumerate(anchors):
            x = (torch.sigmoid(pred[a_idx, :, :, 0]) + grid_x) * stride
            y = (torch.sigmoid(pred[a_idx, :, :, 1]) + grid_y) * stride
            w = torch.exp(pred[a_idx, :, :, 2]) * aw
            h = torch.exp(pred[a_idx, :, :, 3]) * ah
            box = torch.stack([x, y, w, h], dim=-1).reshape(-1, 4)
            box_xyxy = cxcywh_to_xyxy(box)
            obj_scores = obj[a_idx].reshape(-1, 1)
            cls_scores_flat = cls_scores[a_idx].reshape(-1, num_classes)
            combined = obj_scores * cls_scores_flat
            conf, cls_idx = combined.max(dim=1)
            mask = conf > conf_threshold
            if mask.sum() == 0:
                continue
            filtered_boxes = box_xyxy[mask]
            filtered_scores = conf[mask]
            filtered_cls = cls_idx[mask]
            keep = nms(filtered_boxes, filtered_scores, nms_threshold)
            filtered_boxes = filtered_boxes[keep]
            filtered_scores = filtered_scores[keep]
            filtered_cls = filtered_cls[keep]
            for i in range(filtered_boxes.size(0)):
                x1, y1, x2, y2 = filtered_boxes[i].tolist()
                image_boxes.append([x1, y1, x2, y2, filtered_scores[i].item(), float(filtered_cls[i].item())])
                image_scores.append([filtered_scores[i].item()])
        outputs.append(np.array(image_boxes, dtype=np.float32))
        scores_out.append(np.array(image_scores, dtype=np.float32))
    return outputs, scores_out


def mean_average_precision(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> float:
    """Compute mAP@IoU for predictions and targets.

    Predictions: list of (N, 6) arrays [x1,y1,x2,y2,score,cls].
    Targets: list of (M, 5) arrays [x1,y1,x2,y2,cls].
    """
    aps: List[float] = []
    for cls in range(num_classes):
        cls_preds: List[Tuple[int, float, Box]] = []
        cls_gts: Dict[int, List[Box]] = {}
        for img_idx, (pred, tgt) in enumerate(zip(predictions, targets)):
            pred_cls_mask = pred[:, 5] == cls if pred.size > 0 else []
            for entry in pred[pred_cls_mask]:
                cls_preds.append((img_idx, float(entry[4]), (entry[0], entry[1], entry[2], entry[3])))
            tgt_cls_mask = tgt[:, 4] == cls if tgt.size > 0 else []
            boxes = [tuple(box[:4]) for box in tgt[tgt_cls_mask]] if tgt.size > 0 else []
            cls_gts[img_idx] = boxes

        cls_preds.sort(key=lambda x: x[1], reverse=True)
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        gt_counter: Dict[int, List[bool]] = {k: [False] * len(v) for k, v in cls_gts.items()}

        for idx, (img_id, score, box) in enumerate(cls_preds):
            gt_boxes = cls_gts.get(img_id, [])
            if len(gt_boxes) == 0:
                fp[idx] = 1
                continue
            ious = []
            for gt in gt_boxes:
                iou = bbox_iou_np(np.array(box), np.array(gt))
                ious.append(iou)
            best_idx = int(np.argmax(ious))
            if ious[best_idx] >= iou_threshold and not gt_counter[img_id][best_idx]:
                tp[idx] = 1
                gt_counter[img_id][best_idx] = True
            else:
                fp[idx] = 1

        if len(cls_preds) == 0:
            continue
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / (np.array([len(v) for v in cls_gts.values()]).sum() + 1e-6)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def bbox_iou_np(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU for two xyxy boxes in numpy."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute area under the precision-recall curve with VOC 11-point interpolation."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t]
        ap += p.max() if p.size > 0 else 0
    return ap / 11.0


def build_target_tensor(
    targets: List[Dict[str, List[List[float]]]],
    anchors: Sequence[Tuple[int, int]],
    grid_size: int,
    num_classes: int,
    img_size: int,
) -> torch.Tensor:
    """Assign ground truth boxes to anchors and grid cells."""
    batch_size = len(targets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_tensor = torch.zeros((batch_size, len(anchors), grid_size, grid_size, 5 + num_classes), device=device)

    anchor_wh = torch.tensor(anchors, dtype=torch.float32, device=device) / float(img_size)
    for b_idx, sample in enumerate(targets):
        if len(sample["boxes"]) == 0:
            continue
        gt_boxes = torch.tensor(sample["boxes"], dtype=torch.float32, device=device)
        gt_labels = torch.tensor(sample["labels"], dtype=torch.long, device=device)
        normalized = xyxy_to_cxcywh(gt_boxes)
        normalized[:, [0, 2]] /= float(img_size)
        normalized[:, [1, 3]] /= float(img_size)

        ious = wh_iou(anchor_wh, normalized[:, 2:4])
        best_anchors = ious.argmax(dim=0)
        for gt_idx, anchor_idx in enumerate(best_anchors):
            cx, cy, w, h = normalized[gt_idx]
            gx = int(cx * grid_size)
            gy = int(cy * grid_size)
            if gx >= grid_size or gy >= grid_size:
                continue
            tx = cx * grid_size - gx
            ty = cy * grid_size - gy
            tw = math.log(w / (anchor_wh[anchor_idx, 0] + 1e-6))
            th = math.log(h / (anchor_wh[anchor_idx, 1] + 1e-6))
            target_tensor[b_idx, anchor_idx, gy, gx, 0:4] = torch.tensor([tx, ty, tw, th], device=device)
            target_tensor[b_idx, anchor_idx, gy, gx, 4] = 1.0
            target_tensor[b_idx, anchor_idx, gy, gx, 5 + gt_labels[gt_idx]] = 1.0
    return target_tensor

