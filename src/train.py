"""Training entrypoint for the VOC YOLO tiny detector."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolo.boxes import build_target_tensor, decode_predictions, mean_average_precision
from yolo.config import DataConfig, ModelConfig, TrainConfig, VOC_CLASSES
from yolo.data import VOCDataset, collate_fn
from yolo.model import DetectionLoss, YOLOTiny
from yolo.utils import create_writer, plot_training_curves, save_history, seed_everything, visualize_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny YOLO model on Pascal VOC.")
    parser.add_argument("--data-root", type=str, default=str(DataConfig().data_root))
    parser.add_argument("--train-set", type=str, default=DataConfig().train_set)
    parser.add_argument("--val-set", type=str, default=DataConfig().val_set)
    parser.add_argument("--batch-size", type=int, default=TrainConfig().batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig().num_epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig().lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig().weight_decay)
    parser.add_argument("--num-workers", type=int, default=TrainConfig().num_workers)
    parser.add_argument("--device", type=str, default=TrainConfig().device)
    parser.add_argument("--save-dir", type=str, default=str(TrainConfig().save_dir))
    parser.add_argument("--val-interval", type=int, default=TrainConfig().val_interval)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    return parser.parse_args()


def evaluate(
    model: YOLOTiny, loader: DataLoader, device: torch.device, model_cfg: ModelConfig, img_size: int
) -> Dict[str, float]:
    model.eval()
    preds_accum: List[List] = []
    targets_accum: List[List] = []
    stride = img_size // model_cfg.grid_size
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            decoded, _ = decode_predictions(
                outputs, model_cfg.anchors, model_cfg.num_classes, model_cfg.conf_threshold, model_cfg.nms_threshold, stride
            )
            preds_accum.extend(decoded)
            for t in targets:
                if len(t["boxes"]) == 0:
                    targets_accum.append(torch.zeros((0, 5)).numpy())
                    continue
                boxes = torch.tensor(t["boxes"], dtype=torch.float32)
                labels = torch.tensor(t["labels"], dtype=torch.float32).unsqueeze(1)
                merged = torch.cat([boxes, labels], dim=1)
                targets_accum.append(merged.numpy())
    map50 = mean_average_precision(preds_accum, targets_accum, model_cfg.num_classes, iou_threshold=0.5)
    return {"map50": map50}


def main() -> None:
    args = parse_args()
    data_cfg = DataConfig(Path(args.data_root), args.train_set, args.val_set)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        val_interval=args.val_interval,
        device=args.device,
        save_dir=Path(args.save_dir),
    )

    seed_everything(train_cfg.seed)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    train_dataset = VOCDataset(data_cfg.data_root, data_cfg.train_set, "train", data_cfg.img_size)
    val_dataset = VOCDataset(data_cfg.data_root, data_cfg.val_set, "val", data_cfg.img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
    )

    model = YOLOTiny(model_cfg.num_classes, model_cfg.anchors, model_cfg.grid_size).to(device)
    criterion = DetectionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.num_epochs)

    save_dir = train_cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = create_writer(save_dir / "tensorboard")
    history: List[Dict[str, float]] = []
    best_map = 0.0
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0)
        history = checkpoint.get("history", [])
        best_map = checkpoint.get("best_map", 0.0)

    stride = data_cfg.img_size // model_cfg.grid_size
    for epoch in range(start_epoch, train_cfg.num_epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.num_epochs}")
        for images, targets in progress:
            images = images.to(device)
            target_tensor = build_target_tensor(targets, model_cfg.anchors, model_cfg.grid_size, model_cfg.num_classes, data_cfg.img_size)
            preds = model(images)
            loss = criterion(preds, target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})
        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        metrics: Dict[str, float] = {"loss": avg_loss}

        if (epoch + 1) % train_cfg.val_interval == 0:
            eval_metrics = evaluate(model, val_loader, device, model_cfg, data_cfg.img_size)
            metrics.update(eval_metrics)
            writer.add_scalar("Metrics/mAP50", eval_metrics["map50"], epoch + 1)
            writer.add_scalar("Loss/train", avg_loss, epoch + 1)

            if eval_metrics["map50"] > best_map:
                best_map = eval_metrics["map50"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch + 1,
                        "history": history,
                        "best_map": best_map,
                    },
                    save_dir / "best.pt",
                )

            images_cpu = images.cpu()
            decoded, _ = decode_predictions(
                preds.detach(),
                model_cfg.anchors,
                model_cfg.num_classes,
                model_cfg.conf_threshold,
                model_cfg.nms_threshold,
                stride,
            )
            visualize_predictions(
                images_cpu,
                decoded,
                targets,
                VOC_CLASSES,
                save_dir / "visualizations" / f"epoch_{epoch+1}.jpg",
            )

        history.append(metrics)
        save_history(history, save_dir / "history.json")

    plot_training_curves(history, save_dir / "training_curves.png")
    writer.close()


if __name__ == "__main__":
    main()

