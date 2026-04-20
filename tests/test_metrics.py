import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import ConfusionAccumulator


def test_perfect_prediction():
    gt = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint8)
    pred = gt.copy()
    acc = ConfusionAccumulator(num_classes=3, ignore_index=0)
    acc.update(pred, gt)
    m = acc.result()
    assert np.isclose(m.mean_iou, 1.0)
    assert np.isclose(m.pixel_acc, 1.0)
    assert np.isclose(m.mean_class_acc, 1.0)


def test_ignore_excluded():
    # ignore pixel must not affect metrics even if pred differs there.
    gt = np.array([[0, 1, 1], [2, 2, 2]], dtype=np.uint8)
    pred = np.array([[3, 1, 1], [2, 2, 2]], dtype=np.uint8)
    acc = ConfusionAccumulator(num_classes=3, ignore_index=0)
    acc.update(pred, gt)
    m = acc.result()
    assert np.isclose(m.pixel_acc, 1.0)
    assert np.isclose(m.mean_iou, 1.0)


def test_partial():
    gt = np.array([[1, 1, 2, 2]], dtype=np.uint8)
    pred = np.array([[1, 2, 2, 2]], dtype=np.uint8)
    acc = ConfusionAccumulator(num_classes=2, ignore_index=0)
    acc.update(pred, gt)
    m = acc.result()
    # class1: tp=1, fp=0, fn=1 -> iou 0.5
    # class2: tp=2, fp=1, fn=0 -> iou 2/3
    iou = m.per_class_iou
    assert np.isclose(iou[0], 0.5)
    assert np.isclose(iou[1], 2 / 3)
