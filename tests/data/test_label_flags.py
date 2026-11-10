#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Rimon fork: the label flag columns survive the data pipeline.

import json
import os
import random
import tempfile
import unittest

import cv2
import numpy as np

from yolox.data import COCODataset, MosaicDetection, TrainTransform
from yolox.utils import CLS_MASKED_COL, IGNORE_COL, LABEL_WIDTH

SIZE = 64
# (x, y, w, h, category, ignore, class_masked) per image
BOXES = [(4, 4, 16, 16, 1, 0, 0), (40, 40, 20, 20, 1, 1, 0), (4, 40, 16, 16, 2, 0, 1)]


def write_dataset(root, n_images=4):
    os.makedirs(os.path.join(root, "annotations"))
    os.makedirs(os.path.join(root, "train"))
    images, anns = [], []
    for i in range(n_images):
        cv2.imwrite(os.path.join(root, "train", f"{i}.jpg"), np.full((SIZE, SIZE, 3), 100 + i, np.uint8))
        images.append({"id": i + 1, "file_name": f"{i}.jpg", "width": SIZE, "height": SIZE})
        for x, y, w, h, cat, ignore, masked in BOXES:
            anns.append({
                "id": len(anns) + 1, "image_id": i + 1, "category_id": cat,
                "bbox": [x, y, w, h], "area": w * h,
                "iscrowd": ignore, "ignore": ignore, "class_masked": bool(masked),
            })
    cats = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    with open(os.path.join(root, "annotations", "train.json"), "w") as f:
        json.dump({"images": images, "annotations": anns, "categories": cats}, f)


class TestLabelFlags(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        write_dataset(self.tmp.name)
        self.ds = COCODataset(
            data_dir=self.tmp.name, json_file="train.json", name="train",
            img_size=(SIZE, SIZE), preproc=None,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_dataset_keeps_ignore_rows_and_flags(self):
        res = self.ds.load_anno(0)
        self.assertEqual(res.shape, (3, LABEL_WIDTH))
        np.testing.assert_array_equal(res[:, 4], [0, 0, 1])
        np.testing.assert_array_equal(res[:, IGNORE_COL], [0, 1, 0])
        np.testing.assert_array_equal(res[:, CLS_MASKED_COL], [0, 0, 1])
        np.testing.assert_array_equal(res[1, :4], [40, 40, 60, 60])

    def test_train_transform_carries_flags(self):
        tt = TrainTransform(max_labels=10, flip_prob=0.0, hsv_prob=0.0)
        _, lab = tt(self.ds.load_image(0), self.ds.load_anno(0), (SIZE, SIZE))
        self.assertEqual(lab.shape, (10, LABEL_WIDTH))
        np.testing.assert_allclose(lab[1], [0, 50, 50, 20, 20, 1, 0])
        self.assertEqual((lab[2, 0], lab[2, CLS_MASKED_COL]), (1, 1))
        self.assertFalse(lab[3:].any())
        _, empty = tt(self.ds.load_image(0), np.zeros((0, LABEL_WIDTH)), (SIZE, SIZE))
        self.assertEqual(empty.shape, (10, LABEL_WIDTH))
        self.assertFalse(empty.any())

    def test_mosaic_and_mixup_carry_flags(self):
        random.seed(0)
        md = MosaicDetection(
            self.ds, img_size=(SIZE, SIZE), mosaic=True,
            preproc=TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=0.0),
            degrees=0.0, translate=0.0, mosaic_scale=(1.0, 1.0), mixup_scale=(1.0, 1.0),
            shear=0.0, enable_mixup=True, mosaic_prob=1.0, mixup_prob=1.0,
        )
        seen_ignore = seen_masked = False
        for i in range(20):
            _, lab, _, _ = md[i % len(self.ds)]
            self.assertEqual(lab.shape, (50, LABEL_WIDTH))
            rows = lab[lab.sum(1) > 0]
            self.assertTrue(set(np.unique(rows[:, IGNORE_COL])) <= {0.0, 1.0})
            self.assertTrue(set(np.unique(rows[:, CLS_MASKED_COL])) <= {0.0, 1.0})
            self.assertTrue(set(np.unique(rows[:, 0])) <= {0.0, 1.0})
            # a masked row is always class b, an ignore row always class a
            self.assertTrue((rows[rows[:, CLS_MASKED_COL] > 0, 0] == 1).all())
            self.assertTrue((rows[rows[:, IGNORE_COL] > 0, 0] == 0).all())
            seen_ignore |= bool((rows[:, IGNORE_COL] > 0).any())
            seen_masked |= bool((rows[:, CLS_MASKED_COL] > 0).any())
        self.assertTrue(seen_ignore and seen_masked)

        origin = np.full((SIZE, SIZE, 3), 114, np.uint8)
        _, lab = md.mixup(origin, self.ds.load_anno(0).copy(), (SIZE, SIZE))
        self.assertEqual(lab.shape, (6, LABEL_WIDTH))
        np.testing.assert_array_equal(lab[3:, IGNORE_COL], [0, 1, 0])
        np.testing.assert_array_equal(lab[3:, CLS_MASKED_COL], [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
