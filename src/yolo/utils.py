"""Utility helpers for training, logging, and visualization."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_writer(log_dir: Path) -> SummaryWriter:
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def plot_training_curves(history: List[Dict[str, float]], save_path: Path) -> None:
    if not history:
        return
    epochs = list(range(1, len(history) + 1))
    losses = [h["loss"] for h in history]
    map50 = [h.get("map50", 0) for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    color = "tab:blue"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(epochs, losses, color=color, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("mAP@0.5", color=color)
    ax2.plot(epochs, map50, color=color, label="mAP@0.5")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def save_history(history: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def visualize_predictions(
    images: torch.Tensor,
    predictions: Sequence[np.ndarray],
    targets: Sequence[Dict],
    class_names: Sequence[str],
    save_path: Path,
) -> None:
    cols = min(2, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    axes = np.array(axes).reshape(-1)
    for idx, (img, pred, tgt, ax) in enumerate(zip(images, predictions, targets, axes)):
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        ax.imshow(img_np)
        for box, label in zip(tgt["boxes"], tgt["labels"]):
            x1, y1, x2, y2 = box
            ax.add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2)
            )
            ax.text(x1, y1 - 2, class_names[label], color="lime", fontsize=10, weight="bold")
        if pred.size > 0:
            for box in pred:
                x1, y1, x2, y2, score, cls = box
                ax.add_patch(
                    plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=1.5)
                )
                ax.text(x1, y2 + 10, f"{class_names[int(cls)]}:{score:.2f}", color="red", fontsize=9)
        ax.set_axis_off()
        ax.set_title(f"Sample {idx}")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

