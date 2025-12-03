"""Dataset utilities for Pascal VOC detection."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import VOC_CLASSES
from .utils import letterbox


class VOCDataset(Dataset):
    """Pascal VOC detection dataset with YOLO-style targets."""

    def __init__(
        self,
        root: str,
        year_split: str,
        image_set: str = "train",
        img_size: int = 416,
    ) -> None:
        self.root = Path(root)
        self.year_split = year_split
        self.image_set = image_set
        self.img_size = img_size
        self.class_to_idx = {name: idx for idx, name in enumerate(VOC_CLASSES)}

        split_file = (
            self.root
            / year_split
            / "ImageSets"
            / "Main"
            / f"{image_set}.txt"
        )
        if not split_file.exists():
            raise FileNotFoundError(f"Split file {split_file} not found")
        with open(split_file) as f:
            self.ids = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        img_path = self.root / self.year_split / "JPEGImages" / f"{image_id}.jpg"
        xml_path = self.root / self.year_split / "Annotations" / f"{image_id}.xml"

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        boxes, labels = self._parse_annotation(xml_path)

        resized, scale, pad = letterbox(image, self.img_size)
        left, top = pad
        boxes_rescaled = boxes.copy().astype(np.float32)
        boxes_rescaled[:, [0, 2]] = boxes_rescaled[:, [0, 2]] * scale + left
        boxes_rescaled[:, [1, 3]] = boxes_rescaled[:, [1, 3]] * scale + top

        target = {
            "boxes": boxes_rescaled,
            "labels": labels,
            "orig_size": (h, w),
            "image_id": image_id,
        }

        img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return img_tensor, target

    def _parse_annotation(self, xml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes: List[List[float]] = []
        labels: List[int] = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in self.class_to_idx:
                continue
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[cls_name])
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def build_target_tensor(
    targets: List[Dict],
    anchors: List[Tuple[int, int]],
    grid_size: int,
    num_classes: int,
    stride: int,
) -> torch.Tensor:
    """Construct YOLO target tensor from ground truth boxes."""
    batch_size = len(targets)
    num_anchors = len(anchors)
    anchors_norm = [(aw / stride, ah / stride) for aw, ah in anchors]
    y = torch.zeros((batch_size, num_anchors, grid_size, grid_size, 5 + num_classes))
    for b_idx, t in enumerate(targets):
        boxes = torch.tensor(t["boxes"], dtype=torch.float32)
        labels = torch.tensor(t["labels"], dtype=torch.long)
        if boxes.numel() == 0:
            continue
        boxes_xywh = boxes.clone()
        boxes_xywh[:, 0:2] = (boxes[:, 0:2] + boxes[:, 2:4]) / 2.0
        boxes_xywh[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
        boxes_xywh /= float(stride)
        for box, label in zip(boxes_xywh, labels):
            gx, gy, gw, gh = box
            gi, gj = int(gx), int(gy)
            if gi >= grid_size or gj >= grid_size:
                continue
            anchor_ious = []
            for aw, ah in anchors_norm:
                inter = min(gw, aw) * min(gh, ah)
                union = gw * gh + aw * ah - inter
                anchor_ious.append(inter / (union + 1e-6))
            best_anchor = int(np.argmax(anchor_ious))
            y[b_idx, best_anchor, gj, gi, 0] = gx - gi
            y[b_idx, best_anchor, gj, gi, 1] = gy - gj
            y[b_idx, best_anchor, gj, gi, 2] = torch.log(gw / anchors_norm[best_anchor][0] + 1e-6)
            y[b_idx, best_anchor, gj, gi, 3] = torch.log(gh / anchors_norm[best_anchor][1] + 1e-6)
            y[b_idx, best_anchor, gj, gi, 4] = 1.0
            y[b_idx, best_anchor, gj, gi, 5 + label] = 1.0
    return y
