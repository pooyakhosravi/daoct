"""Combined segmentation loss for retinal OCT layers.

Components (all weighted, toggle via weights=0):
  - DiceCE         : region overlap + per-pixel CE (MONAI)            -> Dice term of score
  - Boundary (DT)  : Hausdorff distance-transform loss (MONAI)        -> MASD term of score
  - Ordering       : soft monotonic layer-order penalty along depth   -> topology / disease robustness

The metric rewards both Dice AND boundary localization (exp(-MASD/0.02)), so we
explicitly optimize a boundary term in addition to region overlap.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, HausdorffDTLoss


def _onehot(target, C):
    return F.one_hot(target.squeeze(1).long(), C).permute(0, 3, 1, 2).float()


class GradientBoundaryLoss(nn.Module):
    """GPU-native boundary loss: match the spatial gradient (edge) maps of the
    predicted softmax and the one-hot GT. Emphasizes the depth-axis (row) gradient
    where retinal layer interfaces live (what MASD measures). No CPU distance
    transforms -> keeps the GPU busy, works with on-the-fly geometric augmentation."""

    def __init__(self, num_classes=10, depth_weight=2.0):
        super().__init__()
        self.C = num_classes
        self.dw = depth_weight

    @staticmethod
    def _edges(t, dw):
        dy = (t[:, :, 1:, :] - t[:, :, :-1, :]).abs()
        dx = (t[:, :, :, 1:] - t[:, :, :, :-1]).abs()
        return dw * dy.mean() + dx.mean()

    def forward(self, logits, target):
        p = torch.softmax(logits, dim=1)
        y = _onehot(target, self.C)
        ep = self._edges(p, self.dw)
        ey = self._edges(y, self.dw)
        # L1 between pred/GT edge energy + push pred edges to align with GT edges
        return (ep - ey).abs() + F.l1_loss(
            (p[:, :, 1:, :] - p[:, :, :-1, :]).abs(),
            (y[:, :, 1:, :] - y[:, :, :-1, :]).abs())


# Full top->bottom class order along the depth axis. Per the live challenge spec the
# 10 classes are 2 backgrounds bounding 8 layers: class 0 = preretinal (top) bg,
# classes 1..8 = intra-retinal layers, class 9 = choroid/sclera (bottom) bg. Verified
# monotonic non-decreasing down every column, so we constrain the whole 0..9 stack.
DEFAULT_LAYER_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class OrderingLoss(nn.Module):
    """Penalize layers whose soft depth-centroid violates the anatomical order.

    For each column we compute the expected row position of each layer
    (softmax-prob-weighted), then penalize relu(centroid[k] - centroid[k+1])
    for every adjacent ordered pair. Differentiable, cheap, no labels needed
    (can also be applied to unlabeled predictions as a topology prior)."""

    def __init__(self, order=DEFAULT_LAYER_ORDER, eps=1e-6):
        super().__init__()
        self.order = order
        self.eps = eps

    def forward(self, logits):
        # logits: [B, C, H, W]
        p = torch.softmax(logits, dim=1)
        B, C, H, W = p.shape
        rows = torch.arange(H, device=p.device, dtype=p.dtype).view(1, 1, H, 1)
        mass = p.sum(dim=2, keepdim=True).clamp_min(self.eps)      # [B,C,1,W]
        centroid = (p * rows).sum(dim=2, keepdim=True) / mass       # [B,C,1,W]
        centroid = centroid.squeeze(2)                             # [B,C,W]
        loss = logits.new_zeros(())
        for a, b in zip(self.order[:-1], self.order[1:]):
            # a should be ABOVE b -> centroid[a] < centroid[b]
            loss = loss + F.relu(centroid[:, a] - centroid[:, b]).mean()
        return loss / max(1, len(self.order) - 1)


class CombinedSegLoss(nn.Module):
    def __init__(self, num_classes=10, w_dicece=1.0, w_boundary=1.0, w_order=0.1,
                 include_background=True, boundary_type="gradient", boundary_ds=2):
        super().__init__()
        self.num_classes = num_classes
        self.w_dicece = w_dicece
        self.w_boundary = w_boundary
        self.w_order = w_order
        self.boundary_type = boundary_type
        self.boundary_ds = boundary_ds   # only used by the (slow CPU) hausdorff option
        self.dicece = DiceCELoss(to_onehot_y=True, softmax=True,
                                 include_background=include_background)
        if boundary_type == "gradient":
            self.boundary = GradientBoundaryLoss(num_classes)      # GPU, fast (default)
        else:
            self.boundary = HausdorffDTLoss(to_onehot_y=True, softmax=True,
                                            include_background=include_background)
        self.order = OrderingLoss()

    def forward(self, logits, target):
        # target: [B,1,H,W] integer labels
        out = {}
        total = logits.new_zeros(())
        if self.w_dicece:
            l = self.dicece(logits, target); out["dicece"] = l.detach(); total = total + self.w_dicece * l
        if self.w_boundary:
            if self.boundary_type == "gradient":
                l = self.boundary(logits, target)
            else:
                lo, ta = logits, target
                if self.boundary_ds > 1:             # CPU DT is the bottleneck — lower res
                    lo = F.interpolate(logits, scale_factor=1.0 / self.boundary_ds,
                                       mode="bilinear", align_corners=False)
                    ta = F.interpolate(target.float(), scale_factor=1.0 / self.boundary_ds,
                                       mode="nearest").long()
                l = self.boundary(lo, ta)
            out["boundary"] = l.detach(); total = total + self.w_boundary * l
        if self.w_order:
            l = self.order(logits); out["order"] = l.detach(); total = total + self.w_order * l
        out["total"] = total
        return total, out
