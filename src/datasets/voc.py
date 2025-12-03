"""VOC dataset utilities for YOLO-style training."""
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@dataclass
class VOCAnnotation:
    boxes: torch.Tensor
    labels: torch.Tensor


def _parse_annotation(path: str) -> VOCAnnotation:
    tree = ET.parse(path)
    root = tree.getroot()
    boxes: List[List[float]] = []
    labels: List[int] = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in VOC_CLASSES:
            continue
        cls_id = VOC_CLASSES.index(cls_name)
        bbox = obj.find("bndbox")
        x_min = float(bbox.find("xmin").text)
        y_min = float(bbox.find("ymin").text)
        x_max = float(bbox.find("xmax").text)
        y_max = float(bbox.find("ymax").text)
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(cls_id)
    if not boxes:
        return VOCAnnotation(boxes=torch.zeros((0, 4)), labels=torch.zeros((0,), dtype=torch.long))
    return VOCAnnotation(boxes=torch.tensor(boxes, dtype=torch.float32), labels=torch.tensor(labels, dtype=torch.long))


def _build_transforms(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
        ]
    )


class VOCDataset(Dataset):
    """Dataset that reads PASCAL VOC formatted data."""

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        split_file: str,
        image_size: int,
        augment: bool = True,
    ) -> None:
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_size = image_size
        self.augment = augment
        with open(split_file, "r", encoding="utf-8") as f:
            self.ids = [line.strip() for line in f.readlines() if line.strip()]
        self.transforms = _build_transforms(image_size)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        annotation_path = os.path.join(self.annotation_dir, f"{image_id}.xml")
        image = Image.open(image_path).convert("RGB")
        annotation = _parse_annotation(annotation_path)
        w, h = image.size
        scale_x = self.image_size / w
        scale_y = self.image_size / h
        boxes = annotation.boxes.clone()
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        if self.augment:
            image = self.transforms(image)
        else:
            image = T.Compose([T.Resize((self.image_size, self.image_size)), T.ToTensor()])(image)
        target = {
            "boxes": boxes,
            "labels": annotation.labels,
            "orig_size": torch.tensor([h, w], dtype=torch.float32),
            "image_id": image_id,
        }
        return image, target


def detection_collate(batch: List[Tuple[torch.Tensor, dict]]):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets
