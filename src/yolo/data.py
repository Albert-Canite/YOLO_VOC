"""VOC dataset loader with preprocessing utilities."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import VOC_CLASSES


@dataclass
class Sample:
    image: torch.Tensor
    boxes: torch.Tensor
    labels: torch.Tensor
    path: Path
    orig_size: Tuple[int, int]


class VOCDataset(Dataset):
    """Pascal VOC dataset supporting train and validation splits."""

    def __init__(self, data_root: Path, split: str, subset: str, img_size: int) -> None:
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.image_dir = self.data_root / split / "JPEGImages"
        self.ann_dir = self.data_root / split / "Annotations"
        sets_dir = self.data_root / split / "ImageSets" / "Main"
        set_file = sets_dir / ("trainval.txt" if subset == "train" else "val.txt")
        if subset == "train" and not set_file.exists():
            # Fallback to train split if present
            set_file = sets_dir / "train.txt"
        with open(set_file, "r") as f:
            self.ids = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, List[List[float]]]]:
        img_id = self.ids[idx]
        img_path = self.image_dir / f"{img_id}.jpg"
        ann_path = self.ann_dir / f"{img_id}.xml"

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        boxes, labels = self._load_annotation(ann_path)
        orig_h, orig_w = image.shape[:2]
        image, boxes = letterbox(image, boxes, self.img_size)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": boxes.tolist(),
            "labels": labels.tolist(),
            "path": str(img_path),
            "orig_size": (orig_h, orig_w),
        }
        return image, target

    def _load_annotation(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        tree = ET.parse(path)
        root = tree.getroot()
        boxes: List[List[float]] = []
        labels: List[int] = []
        for obj in root.findall("object"):
            difficult = int(obj.find("difficult").text or 0)
            if difficult:
                continue
            name = obj.find("name").text
            if name not in VOC_CLASSES:
                continue
            cls_id = VOC_CLASSES.index(name)
            bbox = obj.find("bndbox")
            x1 = float(bbox.find("xmin").text)
            y1 = float(bbox.find("ymin").text)
            x2 = float(bbox.find("xmax").text)
            y2 = float(bbox.find("ymax").text)
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def letterbox(image: np.ndarray, boxes: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resize image with unchanged aspect ratio using padding."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_top = (size - nh) // 2
    pad_left = (size - nw) // 2
    padded[pad_top : pad_top + nh, pad_left : pad_left + nw] = resized

    if boxes.shape[0] > 0:
        boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_top
    return padded, boxes


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

