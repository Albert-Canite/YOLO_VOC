"""Entry point for training TinyYOLO on VOC."""
import argparse
import copy
import json
import os
from datetime import datetime
from typing import Optional

import torch

from config import (
    DEFAULT_CONFIG,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from src.utils.engine import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyYOLO on VOC")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config override")
    return parser.parse_args()


def load_config(path: Optional[str]) -> TrainConfig:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = TrainConfig(
        data=DataConfig(**data.get("data", {})),
        model=ModelConfig(**data.get("model", {})),
        optim=OptimConfig(**data.get("optim", {})),
        mixed_precision=data.get("mixed_precision", DEFAULT_CONFIG.mixed_precision),
        save_dir=data.get("save_dir", DEFAULT_CONFIG.save_dir),
        checkpoint_interval=data.get(
            "checkpoint_interval", DEFAULT_CONFIG.checkpoint_interval
        ),
    )
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.save_dir = os.path.join(cfg.save_dir, timestamp)
    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    _, history = train(cfg)
    print("Training complete. Metrics:")
    for k, v in history.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
