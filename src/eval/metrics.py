"""Semantic segmentation metrics: mIoU, pixel acc, mean class acc.

Convention: labels/predictions uint8 in [0..num_classes]. 0 = ignore.
Only pixels where gt != ignore_index contribute to the confusion matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegMetrics:
    confusion: np.ndarray           # (C,C) int64, no ignore row/col
    num_classes: int
    class_names: list[str]

    @property
    def per_class_iou(self) -> np.ndarray:
        cm = self.confusion.astype(np.float64)
        tp = np.diag(cm)
        denom = cm.sum(0) + cm.sum(1) - tp
        iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
        return iou

    @property
    def per_class_acc(self) -> np.ndarray:
        cm = self.confusion.astype(np.float64)
        tp = np.diag(cm)
        support = cm.sum(1)
        return np.where(support > 0, tp / np.maximum(support, 1), np.nan)

    @property
    def mean_iou(self) -> float:
        return float(np.nanmean(self.per_class_iou))

    @property
    def pixel_acc(self) -> float:
        cm = self.confusion.astype(np.float64)
        tot = cm.sum()
        return float(np.diag(cm).sum() / max(tot, 1))

    @property
    def mean_class_acc(self) -> float:
        return float(np.nanmean(self.per_class_acc))

    def table(self) -> str:
        iou = self.per_class_iou
        acc = self.per_class_acc
        lines = [f"{'class':<20} {'IoU':>8} {'Acc':>8}"]
        for i in range(self.num_classes):
            name = self.class_names[i + 1] if i + 1 < len(self.class_names) else f"c{i + 1}"
            lines.append(f"{name:<20} {iou[i] * 100:>7.2f}% {acc[i] * 100:>7.2f}%")
        lines.append("-" * 40)
        lines.append(f"{'mIoU':<20} {self.mean_iou * 100:>7.2f}%")
        lines.append(f"{'mean class acc':<20} {self.mean_class_acc * 100:>7.2f}%")
        lines.append(f"{'pixel acc':<20} {self.pixel_acc * 100:>7.2f}%")
        return "\n".join(lines)


class ConfusionAccumulator:
    def __init__(self, num_classes: int, ignore_index: int = 0,
                 class_names: list[str] | None = None) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or ["unlabeled"] + [
            f"c{i}" for i in range(1, num_classes + 1)
        ]
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """pred, gt: int arrays, same shape. Values in [0..num_classes].
        ignore_index (default 0) pixels in gt are excluded."""
        assert pred.shape == gt.shape
        mask = gt != self.ignore_index
        p = pred[mask].astype(np.int64) - 1  # 1..C -> 0..C-1
        g = gt[mask].astype(np.int64) - 1
        # clip invalid predictions (e.g. pred==0) into a throwaway bin; we ignore them by masking.
        valid = (p >= 0) & (p < self.num_classes) & (g >= 0) & (g < self.num_classes)
        p = p[valid]
        g = g[valid]
        idx = g * self.num_classes + p
        bincount = np.bincount(idx, minlength=self.num_classes ** 2)
        self.cm += bincount.reshape(self.num_classes, self.num_classes)

    def result(self) -> SegMetrics:
        return SegMetrics(
            confusion=self.cm.copy(),
            num_classes=self.num_classes,
            class_names=self.class_names,
        )
