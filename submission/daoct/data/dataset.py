"""Vendor-aware datasets with in-RAM caching for high GPU utilization.

The previous version decoded+resized every PNG on each access (no cache) which
starved the GPU (util peaks then drops to 0 while workers catch up). We now cache
the decoded+resized arrays in RAM: supervised fully (tiny), unlabeled up to a cap.
Augmentation (domain-rand / geom / FDA) happens later on the GPU, so caching the
base resized array is correct.

Resize policy: 'square' resizes to (size,size); 'fixed_h' preserves aspect by
resizing to fixed height then pad/crop width to size (better for the boundary/MASD
term — keeps vertical layer geometry, normalizes cross-vendor axial sampling).
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def _load_gray(path):
    return np.asarray(Image.open(path).convert("L"))


def percentile_norm(arr, lo=2.0, hi=98.0, min_range=8.0):
    """Per-image contrast normalization: stretch [p_lo, p_hi] -> [0,1]. Vendor-agnostic —
    removes the cross-device contrast/range gap (Spectralis std~57/true-black vs Topcon
    std~18/gray-floor) and our local labeled(std47)/unlabeled(std18) Maestro2 mismatch.
    Robust on near-empty B-scans: if the dynamic range is tiny (black/registration-pad
    columns), fall back to plain /255 so background speckle is NOT amplified into structure.
    Returns float32 in [0,1]. Applied identically at train + infer (persisted in meta)."""
    a = arr.astype(np.float32)
    plo, phi = np.percentile(a, lo), np.percentile(a, hi)
    if phi - plo < min_range:
        return np.clip(a / 255.0, 0.0, 1.0)
    return np.clip((a - plo) / (phi - plo), 0.0, 1.0)


def clahe_norm(arr, clip=2.0, grid=8):
    """Local (tile) contrast equalization. Targets INTER-LAYER contrast differences across
    devices (e.g. Spectralis sub-RNFL texture causing layer splits) that a global stretch
    leaves. Returns float32 [0,1]."""
    import cv2
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return c.apply(arr.astype(np.uint8)).astype(np.float32) / 255.0


def apply_norm(arr, mode):
    """mode: falsy -> /255 ; 'clahe' -> CLAHE ; else (True/'percentile') -> percentile stretch."""
    if mode == "clahe":
        return clahe_norm(arr)
    if mode:
        return percentile_norm(arr)
    return arr.astype(np.float32) / 255.0


def _find_mask(ip):
    """Find a mask for an image path across plausible naming conventions."""
    base = ip[:-4] if ip.lower().endswith(".png") else ip
    for c in (ip.replace("-image.png", "-mask.png"), base + "-mask.png",
              base + "_mask.png", ip.replace("image", "mask")):
        if c != ip and os.path.exists(c):
            return c
    return None


def _resize_img(arr, size, mode="square"):
    img = Image.fromarray(arr)
    if mode == "square":
        img = img.resize((size, size), Image.BILINEAR)
    elif mode == "fixed_h":
        w = max(1, round(arr.shape[1] * size / arr.shape[0]))
        img = img.resize((w, size), Image.BILINEAR)
    return np.asarray(img)


def _resize_mask(arr, size, mode="square"):
    img = Image.fromarray(arr)
    if mode == "square":
        img = img.resize((size, size), Image.NEAREST)
    elif mode == "fixed_h":
        w = max(1, round(arr.shape[1] * size / arr.shape[0]))
        img = img.resize((w, size), Image.NEAREST)
    return np.asarray(img)


def _pad_or_crop_w(arr, target_w, pad_val=0):
    w = arr.shape[1]
    if w == target_w:
        return arr
    if w > target_w:
        s = (w - target_w) // 2
        return arr[:, s:s + target_w]
    pad = target_w - w
    l = pad // 2
    return np.pad(arr, ((0, 0), (l, pad - l)), constant_values=pad_val)


class SupervisedSeg(Dataset):
    """`<root>/**/*-image.png` + matching `-mask.png`. Fully cached in RAM."""

    def __init__(self, roots, size=512, mode="square", cache=True, max_cache=4000, norm=False):
        self.size, self.mode, self.cache, self.max_cache = size, mode, cache, max_cache
        self.norm = norm
        self._c = {}
        self.items = []
        for root in roots:
            root = Path(root)
            imgs = sorted(glob.glob(str(root / "**" / "*-image.png"), recursive=True))
            if imgs:                                  # template convention
                for ip in imgs:
                    mp = ip.replace("-image.png", "-mask.png")
                    if os.path.exists(mp):
                        self.items.append((ip, mp))
            else:                                     # robust fallback (v2 may rename)
                for ip in sorted(glob.glob(str(root / "**" / "*.png"), recursive=True)):
                    if "mask" in os.path.basename(ip).lower():
                        continue
                    mp = _find_mask(ip)
                    if mp:
                        self.items.append((ip, mp))

    def __len__(self):
        return len(self.items)

    def _prep(self, i):
        ip, mp = self.items[i]
        img = _resize_img(_load_gray(ip), self.size, self.mode)
        msk = _resize_mask(_load_gray(mp), self.size, self.mode)
        if self.mode == "fixed_h":
            img = _pad_or_crop_w(img, self.size); msk = _pad_or_crop_w(msk, self.size)
        if self.norm:                                  # precompute norm ONCE -> cache normalized
            img = (apply_norm(img, self.norm) * 255.0).astype(np.uint8)
        return img.astype(np.uint8), msk.astype(np.uint8)

    def __getitem__(self, i):
        if self.cache and i in self._c:
            img, msk = self._c[i]
        else:
            img, msk = self._prep(i)
            if self.cache and len(self._c) < self.max_cache:   # cap RAM on large label sets
                self._c[i] = (img, msk)
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None]   # img already normalized if norm
        y = torch.from_numpy(msk.astype(np.int64))[None]
        return {"image": x, "label": y}


class UnlabeledSeg(Dataset):
    """`*-image.png` with no masks (real AI-READI B-scans). Vendor-tagged.

    RAM cache is OFF by default (max_cache=0). It is a per-process cache, so with N
    spawned DataLoader workers it duplicates N times — enabling it on the 40k-image
    unlabeled set blew RAM to ~24GB. Leave off; parallel workers + prefetch handle
    throughput. For a cheap-load speedup without RAM cost, pre-resize to disk instead."""

    def __init__(self, roots, size=512, mode="square", max_cache=0, norm=False):
        self.size, self.mode, self.max_cache = size, mode, max_cache
        self.norm = norm
        self._c = {}
        self.items, self.vendors = [], []
        for root in roots:
            root = Path(root)
            ips = sorted(glob.glob(str(root / "**" / "*-image.png"), recursive=True))
            if not ips:                               # robust fallback (v2 may rename)
                ips = sorted(q for q in glob.glob(str(root / "**" / "*.png"), recursive=True)
                             if "mask" not in os.path.basename(q).lower())
            for ip in ips:
                rel = Path(ip).relative_to(root)
                # vendor = first dir under root, unless that's a status/flat folder
                # (then the root itself is the device dir, e.g. submission layout)
                top = rel.parts[0] if len(rel.parts) > 1 else None
                vendor = root.name if (top is None or top.lower() in
                                       ("diseased", "healthy", "images")) else top
                self.items.append(ip)
                self.vendors.append(vendor)

    def __len__(self):
        return len(self.items)

    def balanced_weights(self):
        from collections import Counter
        c = Counter(self.vendors)
        return [1.0 / c[v] for v in self.vendors]

    def __getitem__(self, i):
        if i in self._c:
            img = self._c[i]
        else:
            img = _resize_img(_load_gray(self.items[i]), self.size, self.mode)
            if self.mode == "fixed_h":
                img = _pad_or_crop_w(img, self.size)
            if self.norm:                              # precompute norm ONCE -> cache normalized
                img = (apply_norm(img, self.norm) * 255.0).astype(np.uint8)
            img = img.astype(np.uint8)
            if len(self._c) < self.max_cache:
                self._c[i] = img
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None]
        return {"image": x, "path": self.items[i]}
