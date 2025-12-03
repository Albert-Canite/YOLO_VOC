"""Training script for the lightweight YOLO VOC detector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolo.boxes import decode_predictions, mean_average_precision
from yolo.config import ModelConfig, TrainConfig, VOC_CLASSES
from yolo.data import VOCDataset, build_target_tensor
from yolo.model import YOLOSmallNet, YoloLoss
from yolo.utils import collate_fn, create_writer, plot_training_curves, save_predictions_visualization, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on Pascal VOC")
    parser.add_argument("--data-root", type=str, default=TrainConfig.data_root)
    parser.add_argument("--train-set", type=str, default=TrainConfig.train_set)
    parser.add_argument("--val-set", type=str, default=TrainConfig.val_set)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.num_epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--save-dir", type=str, default=TrainConfig.save_dir)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    return parser.parse_args()


def evaluate(
    model: YOLOSmallNet,
    loader: DataLoader,
    device: torch.device,
    cfg: ModelConfig,
) -> Dict[str, float]:
    model.eval()
    preds_list: List[List] = []
    targets_list: List[List] = []
    stride = cfg.img_size // cfg.grid_size
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            decoded, scores = decode_predictions(
                outputs,
                cfg.anchors,
                cfg.num_classes,
                cfg.conf_threshold,
                cfg.nms_threshold,
                stride,
            )
            preds_list.extend(decoded)
            batch_targets = []
            for t in targets:
                if len(t["boxes"]) == 0:
                    batch_targets.append(torch.zeros((0, 5)))
                    continue
                boxes = torch.tensor(t["boxes"], dtype=torch.float32)
                labels = torch.tensor(t["labels"], dtype=torch.float32).unsqueeze(1)
                merged = torch.cat([boxes, labels], dim=1)
                batch_targets.append(merged)
            targets_list.extend([bt.numpy() for bt in batch_targets])
    map50 = mean_average_precision(preds_list, targets_list, cfg.num_classes, iou_threshold=0.5)
    return {"map50": map50}


def main() -> None:
    args = parse_args()
    cfg = ModelConfig()
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        device=args.device,
        data_root=args.data_root,
        train_set=args.train_set,
        val_set=args.val_set,
        save_dir=args.save_dir,
    )
    seed_everything()
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    train_dataset = VOCDataset(train_cfg.data_root, train_cfg.train_set, "train", cfg.img_size)
    val_dataset = VOCDataset(train_cfg.data_root, train_cfg.val_set, "val", cfg.img_size)

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

    model = YOLOSmallNet(cfg.num_classes, cfg.anchors, cfg.grid_size).to(device)
    criterion = YoloLoss(cfg.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.num_epochs)

    start_epoch = 0
    best_map = 0.0
    save_dir = Path(train_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = create_writer(save_dir / "tensorboard")
    history: List[Dict[str, float]] = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0)
        history = checkpoint.get("history", [])
        best_map = checkpoint.get("best_map", 0.0)

    stride = cfg.img_size // cfg.grid_size
    for epoch in range(start_epoch, train_cfg.num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.num_epochs}")
        for images, targets in pbar:
            images = images.to(device)
            y_true = build_target_tensor(targets, cfg.anchors, cfg.grid_size, cfg.num_classes, stride).to(device)
            preds = model(images)
            loss = criterion(preds, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        metrics = {"loss": avg_loss}

        if (epoch + 1) % train_cfg.val_interval == 0:
            eval_metrics = evaluate(model, val_loader, device, cfg)
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

            # Visualization on first batch
            images_cpu = images.cpu()
            decoded, scores = decode_predictions(
                preds.detach(), cfg.anchors, cfg.num_classes, cfg.conf_threshold, cfg.nms_threshold, stride
            )
            save_predictions_visualization(
                images_cpu,
                decoded,
                targets,
                VOC_CLASSES,
                save_dir / "visualizations" / f"epoch_{epoch+1}.jpg",
                scores,
            )

        history.append({"loss": avg_loss, **{k: v for k, v in metrics.items() if k != "loss"}})
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    plot_training_curves(history, save_dir / "training_curves.png")
    writer.close()


if __name__ == "__main__":
    main()
