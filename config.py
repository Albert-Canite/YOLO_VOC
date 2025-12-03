"""Configuration for YOLO VOC training."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    root: str = "E:/VOC"
    train_split: str = "VOC2012_train_val"
    test_split: str = "VOC2012_test"
    image_size: int = 416
    num_workers: int = 4
    batch_size: int = 8

    @property
    def train_image_dir(self) -> str:
        return f"{self.root}/{self.train_split}/JPEGImages"

    @property
    def train_annotation_dir(self) -> str:
        return f"{self.root}/{self.train_split}/Annotations"

    @property
    def train_split_file(self) -> str:
        return f"{self.root}/{self.train_split}/ImageSets/Main/trainval.txt"

    @property
    def test_image_dir(self) -> str:
        return f"{self.root}/{self.test_split}/JPEGImages"

    @property
    def test_annotation_dir(self) -> str:
        return f"{self.root}/{self.test_split}/Annotations"

    @property
    def test_split_file(self) -> str:
        return f"{self.root}/{self.test_split}/ImageSets/Main/test.txt"


@dataclass
class ModelConfig:
    num_classes: int = 20
    anchors: List[List[float]] = field(
        default_factory=lambda: [
            [10, 13],
            [16, 30],
            [33, 23],
            [30, 61],
            [62, 45],
            [59, 119],
            [116, 90],
            [156, 198],
            [373, 326],
        ]
    )
    grid_size: int = 13


@dataclass
class OptimConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 5e-4
    warmup_epochs: int = 3
    momentum: float = 0.9


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    mixed_precision: bool = True
    save_dir: str = "outputs"
    checkpoint_interval: int = 10


DEFAULT_CONFIG = TrainConfig()
