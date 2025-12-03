"""Training and evaluation loops."""
import json
import os
from dataclasses import asdict
from typing import Dict, List

import torch
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader

from config import TrainConfig
from src.datasets.voc import VOCDataset, detection_collate
from src.models.yolo import TinyYOLO, decode_predictions
from src.utils.metrics import evaluate_map, non_max_suppression, yolo_loss


def build_model(cfg: TrainConfig) -> TinyYOLO:
    anchors = torch.tensor(cfg.model.anchors, dtype=torch.float32)
    model = TinyYOLO(num_classes=cfg.model.num_classes, anchors=anchors, grid_size=cfg.model.grid_size)
    return model


def prepare_dataloaders(cfg: TrainConfig):
    train_ds = VOCDataset(
        image_dir=cfg.data.train_image_dir,
        annotation_dir=cfg.data.train_annotation_dir,
        split_file=cfg.data.train_split_file,
        image_size=cfg.data.image_size,
        augment=True,
    )
    test_ds = VOCDataset(
        image_dir=cfg.data.test_image_dir,
        annotation_dir=cfg.data.test_annotation_dir,
        split_file=cfg.data.test_split_file,
        image_size=cfg.data.image_size,
        augment=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=detection_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=detection_collate,
    )
    return train_loader, test_loader


def save_checkpoint(model: TinyYOLO, optimizer: optim.Optimizer, epoch: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_{epoch}.pth")
    torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch}, path)


def run_epoch(model, loader, optimizer, scaler, cfg: TrainConfig, device: torch.device, train: bool = True):
    model.train(train)
    epoch_loss = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        with amp.autocast(enabled=cfg.mixed_precision and train):
            pred, _ = model(images)
            loss = yolo_loss(pred, targets, model.anchors, cfg.model.num_classes, cfg.data.image_size)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def predict(model: TinyYOLO, images: torch.Tensor, cfg: TrainConfig, device: torch.device):
    model.eval()
    with torch.no_grad():
        pred, _ = model(images.to(device))
        boxes, scores = decode_predictions(pred, model.anchors, cfg.model.num_classes, cfg.data.image_size)
    batch_predictions = []
    for b in range(images.size(0)):
        box = boxes[b]
        score, cls = scores[b].max(dim=1)
        keep = score > 0.25
        box = box[keep]
        score = score[keep]
        cls = cls[keep]
        if box.numel() == 0:
            batch_predictions.append((torch.zeros((0, 4), device=device), torch.zeros((0,), device=device), torch.zeros((0,), dtype=torch.long, device=device)))
            continue
        xyxy = torch.zeros_like(box)
        xyxy[:, 0] = box[:, 0] - box[:, 2] / 2
        xyxy[:, 1] = box[:, 1] - box[:, 3] / 2
        xyxy[:, 2] = box[:, 0] + box[:, 2] / 2
        xyxy[:, 3] = box[:, 1] + box[:, 3] / 2
        keep_idx = non_max_suppression(xyxy, score)
        batch_predictions.append((xyxy[keep_idx], score[keep_idx], cls[keep_idx]))
    return batch_predictions


def evaluate(model: TinyYOLO, loader: DataLoader, cfg: TrainConfig, device: torch.device):
    all_predictions: List = []
    all_targets: List = []
    for images, targets in loader:
        images = images.to(device)
        preds = predict(model, images, cfg, device)
        all_predictions.extend([(p[0].cpu(), p[1].cpu(), p[2].cpu()) for p in preds])
        all_targets.extend(targets)
    map50 = evaluate_map(all_predictions, all_targets, cfg.model.num_classes, iou_threshold=0.5)
    return map50


def train(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.optim.lr,
        momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay,
    )
    scaler = amp.GradScaler(enabled=cfg.mixed_precision)
    train_loader, test_loader = prepare_dataloaders(cfg)
    history: Dict[str, List[float]] = {"train_loss": [], "map50": []}

    for epoch in range(1, cfg.optim.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, cfg, device, train=True)
        map50 = evaluate(model, test_loader, cfg, device)
        history["train_loss"].append(train_loss)
        history["map50"].append(map50)
        print(f"Epoch {epoch}/{cfg.optim.epochs} - loss: {train_loss:.4f} - mAP@50: {map50:.4f}")
        if epoch % cfg.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, cfg.save_dir)
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(cfg.save_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    return model, history
