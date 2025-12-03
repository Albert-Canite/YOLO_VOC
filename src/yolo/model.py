"""Lightweight YOLO model definition and loss."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class YOLOSmallNet(nn.Module):
    """A compact YOLO-style network for VOC detection."""

    def __init__(self, num_classes: int, anchors: List[Tuple[int, int]], grid_size: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.grid_size = grid_size
        self.features = nn.Sequential(
            ConvBlock(3, 16),
            nn.MaxPool2d(2, 2),
            ConvBlock(16, 32),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512),
            nn.MaxPool2d(2, 2),
            ConvBlock(512, 1024),
            ConvBlock(1024, 512, k=1, p=0),
            ConvBlock(512, 1024),
        )
        out_channels = len(anchors) * (5 + num_classes)
        self.detect = nn.Conv2d(1024, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.features(x)
        x = self.detect(x)
        x = x.view(b, len(self.anchors), 5 + self.num_classes, self.grid_size, self.grid_size)
        return x.permute(0, 1, 3, 4, 2)  # (B, anchors, S, S, 5+C)


class YoloLoss(nn.Module):
    def __init__(self, num_classes: int, lambda_box: float = 5.0, lambda_obj: float = 1.0, lambda_noobj: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.mse = nn.MSELoss(reduction="mean")
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        obj_mask = targets[..., 4] == 1
        noobj_mask = targets[..., 4] == 0

        pred_xy = preds[..., 0:2]
        pred_wh = preds[..., 2:4]
        pred_obj = preds[..., 4]
        pred_cls = preds[..., 5:]

        target_xy = targets[..., 0:2]
        target_wh = targets[..., 2:4]
        target_obj = targets[..., 4]
        target_cls = targets[..., 5:]

        box_loss = self.mse(torch.sigmoid(pred_xy)[obj_mask], target_xy[obj_mask])
        wh_loss = self.mse(pred_wh[obj_mask], target_wh[obj_mask])
        obj_loss = self.bce(pred_obj[obj_mask], target_obj[obj_mask])
        noobj_loss = self.bce(pred_obj[noobj_mask], target_obj[noobj_mask])
        class_loss = self.bce(pred_cls[obj_mask], target_cls[obj_mask])

        loss = self.lambda_box * (box_loss + wh_loss) + self.lambda_obj * obj_loss + self.lambda_noobj * noobj_loss + class_loss
        return loss
