#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Rimon fork: ignore regions and class-masked boxes in the head's loss.

import unittest

import torch

from yolox.models import YOLOXHead
from yolox.utils import CLS_MASKED_COL, IGNORE_COL, LABEL_WIDTH

SIZE = 128
STRIDES = [8, 16, 32]
IN_CHANNELS = [256, 512, 1024]
WIDTH = 0.125

PLAIN = (0, 16, 16, 20, 20)        # cls, cx, cy, w, h: top left
REGION = (0, 108, 108, 24, 24)     # bottom right, far from both gts
MASKED = (1, 16, 112, 20, 20)      # bottom left


class TestIgnoreLoss(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.head = YOLOXHead(num_classes=2, width=WIDTH, strides=STRIDES, in_channels=IN_CHANNELS)
        self.head.initialize_biases(1e-2)
        self.head.train()
        self.xin = [
            torch.randn(1, int(c * WIDTH), SIZE // s, SIZE // s) for c, s in zip(IN_CHANNELS, STRIDES)
        ]
        centres = []
        for s in STRIDES:
            n = SIZE // s
            yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
            centres.append(torch.stack([(xx.flatten() + 0.5) * s, (yy.flatten() + 0.5) * s], 1))
        self.centres = torch.cat(centres, 0).float()   # output anchor order
        cx, cy, w, h = REGION[1:]
        self.in_region = (
            (self.centres[:, 0] - cx).abs() <= w / 2
        ) & ((self.centres[:, 1] - cy).abs() <= h / 2)
        self.near_masked = (self.centres - torch.tensor(MASKED[1:3]).float()).norm(dim=1) < (
            self.centres - torch.tensor(PLAIN[1:3]).float()
        ).norm(dim=1)

    @staticmethod
    def labels(rows, ignore=True, masked=True):
        lab = torch.zeros(1, 10, LABEL_WIDTH)
        for i, row in enumerate(rows):
            lab[0, i, :5] = torch.tensor(row).float()
            if row is REGION:
                lab[0, i, IGNORE_COL] = float(ignore)
            if row is MASKED:
                lab[0, i, CLS_MASKED_COL] = float(masked)
        return lab

    def run_loss(self, lab):
        """Loss tuple, objectness-logit gradients [n_anchors] and class-logit
        gradients [n_anchors, nc] in the head's anchor order."""
        obj_out, cls_out, hooks = [], [], []

        def keep(store):
            def hook(_m, _i, o):
                o.retain_grad()
                store.append(o)
            return hook

        for k in range(len(STRIDES)):
            hooks.append(self.head.obj_preds[k].register_forward_hook(keep(obj_out)))
            hooks.append(self.head.cls_preds[k].register_forward_hook(keep(cls_out)))
        self.head.zero_grad()
        losses = self.head(self.xin, lab, None)
        losses[0].backward()
        for h in hooks:
            h.remove()
        obj_grad = torch.cat([o.grad.flatten(2)[0, 0] for o in obj_out])
        cls_grad = torch.cat([c.grad.flatten(2)[0].T for c in cls_out])
        return losses, obj_grad, cls_grad

    def test_ignore_region_gets_no_negative_objectness_loss(self):
        self.assertGreater(int(self.in_region.sum()), 0)
        _, obj_grad, _ = self.run_loss(self.labels([PLAIN, REGION, MASKED]))
        # inside the region: no gradient at all (no gt there, so no positives either)
        self.assertTrue((obj_grad[self.in_region] == 0).all())
        # everywhere else every anchor is a negative (grad > 0) or a positive (grad < 0)
        self.assertTrue((obj_grad[~self.in_region] != 0).all())
        # the same box as a plain gt: positives inside, negatives around, nothing silent
        _, obj_grad, _ = self.run_loss(self.labels([PLAIN, REGION, MASKED], ignore=False))
        self.assertTrue((obj_grad[self.in_region] != 0).all())
        self.assertTrue((obj_grad[self.in_region] < 0).any())

    def test_ignore_only_image_is_a_negative_image_with_a_hole(self):
        losses, obj_grad, _ = self.run_loss(self.labels([REGION]))
        self.assertEqual(float(losses[1].detach()), 0.0)   # iou
        self.assertEqual(float(losses[3].detach()), 0.0)   # cls
        self.assertGreater(float(losses[2].detach()), 0.0)  # obj
        self.assertTrue((obj_grad[self.in_region] == 0).all())
        self.assertTrue((obj_grad[~self.in_region] > 0).all())

    def test_class_masked_gt_gives_no_class_loss(self):
        _, obj_grad, cls_grad = self.run_loss(self.labels([PLAIN, REGION, MASKED]))
        fg = obj_grad < 0
        self.assertTrue((fg & self.near_masked).any() and (fg & ~self.near_masked).any())
        self.assertTrue((cls_grad[fg & self.near_masked] == 0).all())
        self.assertTrue((cls_grad[fg & ~self.near_masked] != 0).any(dim=1).all())
        self.assertTrue((cls_grad[~fg] == 0).all())
        _, obj_grad, cls_grad = self.run_loss(self.labels([PLAIN, REGION, MASKED], masked=False))
        fg = obj_grad < 0
        self.assertTrue((cls_grad[fg & self.near_masked] != 0).any(dim=1).all())

    def test_five_wide_labels_still_train(self):
        lab = self.labels([PLAIN, REGION, MASKED])[:, :, :5]
        losses, obj_grad, _ = self.run_loss(lab)
        self.assertTrue(torch.isfinite(losses[0]))
        self.assertTrue((obj_grad != 0).all())


if __name__ == "__main__":
    unittest.main()
