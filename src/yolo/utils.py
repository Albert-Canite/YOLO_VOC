"""Utility helpers for preprocessing and logging."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def letterbox(image: np.ndarray, new_size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image to square with unchanged aspect ratio using padding."""
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), 128, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top : top + nh, left : left + nw, :] = resized
    return canvas, scale, (left, top)


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)


def plot_training_curves(log_history: List[Dict[str, float]], save_path: Path) -> None:
    """Plot loss and metric curves from history dictionaries."""
    if not log_history:
        return
    steps = list(range(1, len(log_history) + 1))
    train_loss = [h.get("loss", 0.0) for h in log_history]
    val_loss = [h.get("val_loss", np.nan) for h in log_history]
    map50 = [h.get("map50", np.nan) for h in log_history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_loss, label="train_loss")
    plt.plot(steps, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(steps, map50, label="mAP@0.5")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.ylim(0, 1)
    plt.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_predictions_visualization(
    images: torch.Tensor,
    pred_boxes: List[np.ndarray],
    targets: List[Dict],
    class_names: List[str],
    save_path: Path,
    scores: List[np.ndarray],
) -> None:
    """Save side-by-side prediction vs GT visualization."""
    images_np = (images.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    grid_imgs: List[np.ndarray] = []
    for idx, img in enumerate(images_np):
        vis = img.copy()
        for box, score in zip(pred_boxes[idx], scores[idx]):
            x1, y1, x2, y2, cls = box.astype(int)
            label = f"{class_names[int(cls)]}:{score:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for gt_box, gt_cls in zip(targets[idx]["boxes"], targets[idx]["labels"]):
            gx1, gy1, gx2, gy2 = [int(c) for c in gt_box]
            label = f"GT:{class_names[int(gt_cls)]}"
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
            cv2.putText(vis, label, (gx1, gy1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        grid_imgs.append(vis)

    rows = math.ceil(len(grid_imgs) / 2)
    cols = 2
    cell_h, cell_w, _ = grid_imgs[0].shape
    canvas = np.full((cell_h * rows, cell_w * cols, 3), 255, dtype=np.uint8)
    for idx, img in enumerate(grid_imgs):
        r, c = divmod(idx, cols)
        canvas[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = img

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def create_writer(save_dir: Path) -> SummaryWriter:
    save_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(save_dir))


def seed_everything(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

