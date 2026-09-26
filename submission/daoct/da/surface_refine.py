"""Graph-search (dynamic-programming) surface refinement for retinal OCT layers.

The network gives per-pixel class probabilities. argmax can produce topology
violations (a deeper layer appearing above a shallower one), which wreck the MASD
term and are anatomically impossible. We replace argmax with a per-column DP that
finds the highest-probability labeling subject to the class index being
NON-DECREASING down the column (top->bottom order 0,1,2,...,9, verified in the data).

This guarantees valid layer ordering and tends to sharpen boundaries -> directly
helps the exp(-MASD/0.02) term. O(H * W * C) with a tiny C-loop; vectorized over
columns. Pure numpy, runs at inference only.
"""
from __future__ import annotations
import numpy as np


def dp_refine_prob(prob: np.ndarray) -> np.ndarray:
    """prob: [C,H,W] softmax probabilities. Returns [H,W] uint8 ordered labels."""
    C, H, W = prob.shape
    logp = np.log(prob.astype(np.float32) + 1e-8).transpose(1, 2, 0)  # [H,W,C]

    best = np.empty((H, W, C), np.float32)
    ptr = np.empty((H, W, C), np.int16)
    best[0] = logp[0]
    ptr[0] = np.arange(C)[None, :]

    for r in range(1, H):
        prev = best[r - 1]                       # [W,C]
        # prefix max / argmax over states s' <= s  (allowed transitions: stay or go deeper)
        cmax = prev.copy()
        carg = np.tile(np.arange(C, dtype=np.int16), (W, 1))
        for s in range(1, C):
            take = prev[:, s] > cmax[:, s - 1]
            cmax[:, s] = np.where(take, prev[:, s], cmax[:, s - 1])
            carg[:, s] = np.where(take, s, carg[:, s - 1])
        best[r] = logp[r] + cmax
        ptr[r] = carg

    labels = np.empty((H, W), np.uint8)
    s = np.argmax(best[H - 1], axis=1).astype(np.int16)   # [W]
    cols = np.arange(W)
    for r in range(H - 1, -1, -1):
        labels[r] = s.astype(np.uint8)
        s = ptr[r][cols, s]
    return labels


def dp_refine_logits(logits) -> np.ndarray:
    """Convenience: torch logits [1,C,H,W] or [C,H,W] -> refined [H,W] uint8."""
    import torch
    if logits.dim() == 4:
        logits = logits[0]
    prob = torch.softmax(logits, dim=0).detach().cpu().numpy()
    return dp_refine_prob(prob)


def _lateral_median(surf, ksize):
    try:
        from scipy.ndimage import median_filter
        return median_filter(surf, size=(1, ksize), mode="nearest")
    except Exception:                                   # pure-numpy sliding-window median
        pad = ksize // 2
        sp = np.pad(surf, ((0, 0), (pad, pad)), mode="edge")
        win = np.stack([sp[:, o:o + ksize] for o in range(surf.shape[1])], axis=1)
        return np.median(win, axis=2)


def smooth_surfaces(labels, ksize=9, empty=None, n_classes=10, despike_tol=5.0, despike_k=15):
    """Lateral (cross-column) cleanup of layer boundaries.

    dp_refine_prob orders each A-scan column INDEPENDENTLY (no coupling across columns),
    so a column can jump — the vertical spikes/notches on steep widefield/Spectralis and
    near vessel-shadow columns. Two passes on each boundary's per-column row:
      1) DESPIKE: where the row deviates > despike_tol px from a wide local median, replace
         it with the median; otherwise keep it EXACTLY. This kills isolated jumps of any
         width up to ~despike_k while leaving genuine steep walls (monotonic -> small
         deviation) and broad drusen (median follows them) untouched.
      2) light median (ksize) for residual single-px wobble.
    Empty/registration-gap columns are interpolated over first. Re-impose ordering, raster.
    Pure post-proc -> helps every model, no retrain."""
    H, W = labels.shape
    C = n_classes
    surf = np.stack([(labels < b).sum(0) for b in range(1, C)]).astype(np.float32)  # [C-1,W]
    if empty is not None and empty.any() and (~empty).sum() >= 2:
        xs = np.arange(W); v = ~empty
        for i in range(surf.shape[0]):
            surf[i] = np.interp(xs, xs[v], surf[i][v])
    if despike_tol:                                     # targeted outlier replacement
        med = _lateral_median(surf, despike_k)
        spike = np.abs(surf - med) > despike_tol
        surf = np.where(spike, med, surf)
    if ksize and ksize > 1:                             # gentle residual smoothing
        surf = _lateral_median(surf, ksize)
    surf = np.maximum.accumulate(surf, axis=0)          # re-enforce boundary order per column
    rows = np.arange(H)[:, None, None]
    lab = (surf[None] <= rows).sum(1).astype(np.uint8)
    if empty is not None and empty.any():
        lab[:, empty] = 0
    return lab


def surface_decode(surf_rows, H, blend_dp=None, w=1.0):
    """Decode labels from the surface-regression head (ordered boundary regression).
    surf_rows: [n_bound, W] normalized (0..1) boundary positions. Enforce non-decreasing
    order, optionally average with a DP-derived surface set (blend_dp same shape), rasterize.
    Order-by-construction -> a layer CANNOT be split into two classes, and two surfaces may
    coincide -> foveal zero-thickness represented naturally [Morelle 2023, He 2019]."""
    sr = np.clip(surf_rows.astype(np.float32), 0, 1)
    if blend_dp is not None:
        sr = w * sr + (1 - w) * np.clip(blend_dp.astype(np.float32), 0, 1)
    sr = np.maximum.accumulate(sr, axis=0)              # [n_bound, W]
    rows = (np.arange(H)[:, None, None] + 0.5) / H
    return (sr[None] <= rows).sum(1).astype(np.uint8)


def dp_surfaces(labels, n_classes=10):
    """Per-column normalized boundary rows from a label map (inverse of rasterize). [n_bound,W]."""
    H = labels.shape[0]
    return np.stack([(labels < b).sum(0) for b in range(1, n_classes)]).astype(np.float32) / H


def mask_empty_columns(pred, img01, thresh=0.06, bg=0):
    """Force no-signal columns to background.

    After volume registration the device shifts B-scans, leaving BLACK (no-tissue)
    columns at the edges; the model still draws layers there. Columns whose max
    intensity is below `thresh` carry no retina -> set them to the `bg` class so we
    don't emit spurious layers (which would be false positives on Dice + MASD)."""
    import numpy as np
    empty = img01.max(axis=0) < thresh
    if empty.any():
        pred = pred.copy()
        pred[:, empty] = bg
    return pred
