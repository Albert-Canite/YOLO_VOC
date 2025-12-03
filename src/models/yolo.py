"""A lightweight YOLO-style network for VOC."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.act(self.bn(self.conv(x)))


class TinyYOLO(nn.Module):
    """Small YOLO network with a single detection head."""

    def __init__(self, num_classes: int, anchors: torch.Tensor, grid_size: int | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = anchors.size(0)
        self.register_buffer("anchors", anchors)
        # Grid size can be inferred from the feature map; keeping a user-provided
        # value allows compatibility with saved configs while avoiding hardcoded
        # reshaping that can mismatch the actual spatial size.
        self.grid_size = grid_size

        self.backbone = nn.Sequential(
            ConvBlock(3, 16, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(16, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, padding=1),
            nn.MaxPool2d(2, 1, padding=1),
            ConvBlock(512, 1024, 3, padding=1),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3, padding=1),
            ConvBlock(1024, 512, 1),
        )
        self.det_head = nn.Conv2d(512, self.num_anchors * (5 + num_classes), 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        features = self.backbone(x)
        pred = self.det_head(features)
        grid_h, grid_w = pred.shape[2:]
        pred = pred.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_h, grid_w)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        # Update cached grid_size to reflect the true feature map resolution for downstream decoding.
        self.grid_size = grid_h
        return pred, features


def decode_predictions(pred: torch.Tensor, anchors: torch.Tensor, num_classes: int, img_size: int):
    batch_size = pred.size(0)
    device = pred.device
    grid_h, grid_w = pred.size(2), pred.size(3)
    stride = img_size / grid_w

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, device=device), torch.arange(grid_w, device=device), indexing="ij"
    )
    grid_x = grid_x.view(1, 1, grid_h, grid_w)
    grid_y = grid_y.view(1, 1, grid_h, grid_w)

    pred = pred.sigmoid()
    box_xy = pred[..., 0:2]
    box_wh = pred[..., 2:4]
    obj_score = pred[..., 4:5]
    class_score = pred[..., 5:]

    box_x = (box_xy[..., 0] + grid_x) * stride
    box_y = (box_xy[..., 1] + grid_y) * stride
    box_w = torch.exp(box_wh[..., 0]) * anchors[:, 0].view(-1, 1, 1)
    box_h = torch.exp(box_wh[..., 1]) * anchors[:, 1].view(-1, 1, 1)

    boxes = torch.stack([box_x, box_y, box_w, box_h], dim=-1)
    scores = obj_score * class_score
    boxes = boxes.view(batch_size, -1, 4)
    scores = scores.view(batch_size, -1, num_classes)
    return boxes, scores
