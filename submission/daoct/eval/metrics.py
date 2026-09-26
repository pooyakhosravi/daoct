"""Local copy of the challenge per-image scoring (mirrors baseline scripts/metrics.py).

image_score = mean_c 0.5*(Dice_c + exp(-MASD_c/TAU)), MASD normalized by image height.
Use for local validation so we optimize the real leaderboard objective, not a proxy.
"""
from __future__ import annotations
import numpy as np

NUM_CLASSES = 10
TAU = 0.02


def dice_score(pred, gt, num_classes=NUM_CLASSES, eps=1e-6):
    out = []
    for c in range(num_classes):
        pc, gc = (pred == c), (gt == c)
        inter = np.logical_and(pc, gc).sum()
        out.append((2.0 * inter + eps) / (pc.sum() + gc.sum() + eps))
    return np.array(out)


def _boundary(m):
    from scipy.ndimage import binary_erosion   # lazy: not in the default runtime image
    return m ^ binary_erosion(m)


def _surface_distance(pred_c, gt_c):
    from scipy.ndimage import distance_transform_edt
    pb, gb = _boundary(pred_c), _boundary(gt_c)
    if pb.sum() == 0 or gb.sum() == 0:
        return np.nan
    gd = distance_transform_edt(~gb)
    pd = distance_transform_edt(~pb)
    return (gd[pb].mean() + pd[gb].mean()) / 2.0


def masd_per_class(pred, gt):
    H = pred.shape[0]
    out = []
    for c in range(NUM_CLASSES):
        d = _surface_distance(pred == c, gt == c)
        out.append(np.nan if np.isnan(d) else d / H)
    return np.array(out)


def compute_image_score(pred, gt):
    dice = dice_score(pred, gt)
    masd = masd_per_class(pred, gt)
    masd_score = np.nan_to_num(np.exp(-masd / TAU), nan=0.0)
    return float((0.5 * (dice + masd_score)).mean())
