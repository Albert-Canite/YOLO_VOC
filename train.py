"""Entry point for training TinyYOLO on VOC."""
import argparse
import json
import os
from datetime import datetime

import torch

from config import DEFAULT_CONFIG, TrainConfig
from src.utils.engine import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyYOLO on VOC")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config override")
    return parser.parse_args()


def load_config(path: str | None) -> TrainConfig:
    if path is None:
        return DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = TrainConfig(**data)
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
