"""Visualization utilities for training metrics and predictions."""
import json
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont

from config import TrainConfig
from src.datasets.voc import VOCDataset, VOC_CLASSES
from src.models.yolo import TinyYOLO, decode_predictions
from src.utils.engine import build_model
from src.utils.metrics import non_max_suppression


plt.switch_backend("Agg")


def plot_history(history_path: str, output_path: str):
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, history["train_loss"], label="train_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["map50"], color="green", label="mAP@50")
    ax2.set_ylabel("mAP@50")
    fig.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)


def load_checkpoint(model: TinyYOLO, checkpoint_path: str):
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"])
    return model


def draw_boxes(image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"{VOC_CLASSES[label]} {score:.2f}"
        if font:
            draw.text((x1, y1), text, fill="yellow", font=font)
        else:
            draw.text((x1, y1), text, fill="yellow")
    return image


def visualize_samples(
    cfg: TrainConfig,
    checkpoint_path: str,
    output_dir: str,
    num_samples: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model = load_checkpoint(model, checkpoint_path)
    model.eval()
    dataset = VOCDataset(
        image_dir=cfg.data.val_image_dir,
        annotation_dir=cfg.data.val_annotation_dir,
        split_file=cfg.data.val_split_file,
        image_size=cfg.data.image_size,
        augment=False,
    )
    os.makedirs(output_dir, exist_ok=True)
    for idx in range(min(num_samples, len(dataset))):
        image, target = dataset[idx]
        with torch.no_grad():
            pred, _ = model(image.unsqueeze(0).to(device))
            boxes, scores = decode_predictions(pred, model.anchors, cfg.model.num_classes, cfg.data.image_size)
        box = boxes[0]
        score, cls = scores[0].max(dim=1)
        keep = score > 0.25
        box = box[keep]
        score = score[keep]
        cls = cls[keep]
        xyxy = torch.zeros_like(box)
        xyxy[:, 0] = box[:, 0] - box[:, 2] / 2
        xyxy[:, 1] = box[:, 1] - box[:, 3] / 2
        xyxy[:, 2] = box[:, 0] + box[:, 2] / 2
        xyxy[:, 3] = box[:, 1] + box[:, 3] / 2
        keep_idx = non_max_suppression(xyxy, score)
        xyxy = xyxy[keep_idx]
        score = score[keep_idx]
        cls = cls[keep_idx]
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
        vis_image = Image.fromarray(image_np)
        vis_image = draw_boxes(vis_image, xyxy, cls, score)
        # draw GT boxes
        gt_image = Image.fromarray(image_np)
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        gt_scores = torch.ones_like(gt_labels, dtype=torch.float32)
        gt_image = draw_boxes(gt_image, gt_boxes, gt_labels, gt_scores)
        vis_image.save(os.path.join(output_dir, f"pred_{idx}.png"))
        gt_image.save(os.path.join(output_dir, f"gt_{idx}.png"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--history", type=str, required=True, help="Path to history.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to load")
    parser.add_argument("--config", type=str, default=None, help="Optional config override")
    parser.add_argument("--output", type=str, default="outputs/vis", help="Directory to save visuals")
    args = parser.parse_args()

    cfg = TrainConfig() if args.config is None else TrainConfig(**json.load(open(args.config, "r", encoding="utf-8")))
    os.makedirs(args.output, exist_ok=True)
    plot_history(args.history, os.path.join(args.output, "training_curve.png"))
    visualize_samples(cfg, args.checkpoint, args.output)
