# SPDX-License-Identifier: AGPL-3.0-only

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from dataloader import YOLODataset, read_label
from losses import YOLO26Loss
from metrics import DetectionMetrics
from model import YOLO26


class SmokeTests(unittest.TestCase):
    def test_model_parameter_count_and_forward(self):
        model = YOLO26(nc=3, size="n").train()
        self.assertEqual(model.num_parameters(), 2_504_970)
        output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(set(output), {"one2many", "one2one"})
        self.assertEqual(output["one2one"]["boxes"].shape, (1, 4, 84))
        self.assertEqual(output["one2one"]["scores"].shape, (1, 3, 84))

    def test_official_parameter_counts(self):
        expected = {"n": 2_572_280, "s": 10_009_784, "m": 21_896_248, "l": 26_299_704, "x": 58_993_368}
        for size, count in expected.items():
            self.assertEqual(YOLO26(nc=80, size=size).num_parameters(), count)

    def test_loss_backward(self):
        model = YOLO26(nc=3, size="n").train()
        images = torch.randn(2, 3, 64, 64)
        batch = {
            "batch_idx": torch.tensor([0, 1]),
            "cls": torch.tensor([[1.0], [2.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25], [0.4, 0.4, 0.2, 0.2]]),
        }
        loss, items = YOLO26Loss(model, epochs=10)(model(images), batch)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertEqual(set(items), {"box", "cls", "l1", "o2m_weight"})
        loss.sum().backward()
        self.assertIsNotNone(model.model[0].conv.weight.grad)

    def test_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images/train").mkdir(parents=True)
            (root / "labels/train").mkdir(parents=True)
            cv2.imwrite(str(root / "images/train/a.jpg"), np.full((40, 60, 3), 127, np.uint8))
            (root / "labels/train/a.txt").write_text("0 0.5 0.5 0.4 0.5\n", encoding="utf-8")
            dataset = YOLODataset(str(root / "images/train"), nc=1, image_size=64)
            sample = dataset[0]
            self.assertEqual(tuple(sample["img"].shape), (3, 64, 64))
            self.assertEqual(tuple(sample["bboxes"].shape), (1, 4))

    def test_label_routes_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "labels.txt"
            label.write_text("0 2 0.5 0.5 0.4 0.5\n", encoding="utf-8")
            multiclass = read_label(label, nc=3, use_multiclass=True)
            self.assertEqual(multiclass[0, :3].tolist(), [1.0, 0.0, 1.0])
            self.assertEqual(tuple(multiclass.shape), (1, 7))
            with self.assertRaises(ValueError):
                read_label(label, nc=3, use_multiclass=False)

    def test_multiclass_loss_backward(self):
        model = YOLO26(nc=3, size="n").train()
        batch = {
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([[1.0, 0.0, 1.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        }
        loss, _ = YOLO26Loss(model, epochs=10, use_multiclass=True)(model(torch.randn(1, 3, 64, 64)), batch)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        self.assertIsNotNone(model.model[0].conv.weight.grad)

    def test_perfect_metrics(self):
        metrics = DetectionMetrics(1, ["object"])
        batch = {
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        }
        pred = [torch.tensor([[16.0, 16.0, 48.0, 48.0, 0.99, 0.0]])]
        metrics.update(pred, batch, image_size=64)
        result = metrics.compute()
        self.assertGreater(result["map50_95"], 0.99)


if __name__ == "__main__":
    unittest.main()
