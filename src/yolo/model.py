"""PyTorch implementation of a compact YOLO-style detector."""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CSPBlock(nn.Module):
    """A tiny CSP-like block for lightweight feature reuse."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = channels // 2
        self.conv1 = ConvBNAct(channels, hidden, k=1, p=0)
        self.conv2 = ConvBNAct(hidden, channels, k=3, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + shortcut


class YOLOTiny(nn.Module):
    """Compact detector with a single detection scale."""

    def __init__(self, num_classes: int, anchors: Sequence[Tuple[int, int]], grid_size: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.grid_size = grid_size

        self.stem = nn.Sequential(
            ConvBNAct(3, 32, k=3, s=2, p=1),
            ConvBNAct(32, 64, k=3, s=2, p=1),
            CSPBlock(64),
            ConvBNAct(64, 128, k=3, s=2, p=1),
            CSPBlock(128),
            ConvBNAct(128, 256, k=3, s=2, p=1),
            CSPBlock(256),
            ConvBNAct(256, 512, k=3, s=2, p=1),
            CSPBlock(512),
        )
        out_channels = len(anchors) * (5 + num_classes)
        self.detect = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        feats = self.stem(x)
        pred = self.detect(feats)
        pred = pred.view(b, len(self.anchors), 5 + self.num_classes, self.grid_size, self.grid_size)
        return pred.permute(0, 1, 3, 4, 2)  # (B, anchors, S, S, 5+C)


class DetectionLoss(nn.Module):
    """YOLO-style loss combining localization, objectness, and classification."""

    def __init__(self, lambda_box: float = 5.0, lambda_obj: float = 1.0, lambda_noobj: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")
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

        xy_loss = self.mse(torch.sigmoid(pred_xy)[obj_mask], target_xy[obj_mask])
        wh_loss = self.mse(pred_wh[obj_mask], target_wh[obj_mask])
        obj_loss = self.bce(pred_obj[obj_mask], target_obj[obj_mask])
        noobj_loss = self.bce(pred_obj[noobj_mask], target_obj[noobj_mask])
        cls_loss = self.bce(pred_cls[obj_mask], target_cls[obj_mask])

        loss = self.lambda_box * (xy_loss + wh_loss) + self.lambda_obj * obj_loss + self.lambda_noobj * noobj_loss + cls_loss
        return loss

