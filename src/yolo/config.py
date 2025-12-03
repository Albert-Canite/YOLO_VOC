"""Centralized configuration for VOC training."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


VOC_CLASSES: List[str] = [
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
class DataConfig:
    data_root: Path = Path("E:/VOC")
    train_set: str = "VOC2012_train_val"
    val_set: str = "VOC2012_test"
    img_size: int = 416
    cache_images: bool = False


@dataclass
class ModelConfig:
    num_classes: int = len(VOC_CLASSES)
    anchors: List[Tuple[int, int]] = field(
        default_factory=lambda: [(12, 16), (19, 36), (40, 28)]
    )
    grid_size: int = 13
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45


@dataclass
class TrainConfig:
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 5e-4
    num_epochs: int = 60
    val_interval: int = 1
    seed: int = 42
    device: str = "cuda"
    save_dir: Path = Path("runs")

