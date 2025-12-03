"""Configuration utilities for the YOLO VOC project."""
from dataclasses import dataclass
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
class ModelConfig:
    num_classes: int = len(VOC_CLASSES)
    anchors: List[Tuple[int, int]] = ((10, 13), (16, 30), (33, 23))
    img_size: int = 416
    grid_size: int = 13
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45


@dataclass
class TrainConfig:
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 5e-4
    warmup_epochs: int = 2
    num_epochs: int = 50
    val_interval: int = 1
    device: str = "cuda"  # fallback to cpu will be handled at runtime
    data_root: str = "E:/VOC"
    train_set: str = "VOC2012_train_val"
    val_set: str = "VOC2012_test"
    save_dir: str = "runs"
