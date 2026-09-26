"""Multi-head segmentation model + GPU-cheap boundary targets.

Trunk (SegResNet or MedNeXt) emits a full-res feature map; three heads sit on top:
  - seg      : per-pixel class logits (the primary output)
  - sdm      : per-pixel distance-to-nearest-layer-boundary (regression)   [#3]
  - surface  : per-A-scan-column boundary row positions (regression)       [#5]

All head targets are derived from the (monotonic 0..9) label on the GPU — no CPU
distance transforms — so training stays GPU-bound and works with on-the-fly geom aug.
Instance-norm trunk gives vendor invariance (IBN-flavoured)               [#1].
MedNeXt trunk option                                                      [#4].
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SegResNet, MedNeXt


class DeformRefine(nn.Module):
    """Modulated deformable conv (DCNv2) residual block — bakes curvature/tilt
    invariance into the features: the kernel learns per-location sampling offsets so
    it follows the (possibly steeply tilted) retinal layers instead of a rigid grid.
    Zero-initialised offsets/mask -> starts ~identity for stable training."""

    def __init__(self, ch, k=3, scale=2):
        super().__init__()
        from torchvision.ops import DeformConv2d   # lazy: only needed when use_deform=True
        pad = k // 2
        self.scale = scale                    # run deform at 1/scale res (DCN is slow at full res)
        self.offset = nn.Conv2d(ch, 2 * k * k, k, padding=pad)
        self.mask = nn.Conv2d(ch, k * k, k, padding=pad)
        self.dcn = DeformConv2d(ch, ch, k, padding=pad)
        self.norm = nn.InstanceNorm2d(ch, affine=True)
        self.act = nn.ReLU(inplace=True)
        for m in (self.offset, self.mask):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        xd = F.avg_pool2d(x, self.scale) if self.scale > 1 else x
        y = self.dcn(xd, self.offset(xd), torch.sigmoid(self.mask(xd)))
        y = self.act(self.norm(y))
        if self.scale > 1:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + y


def build_trunk(name, in_ch, feat, norm="instance", init_filters=16):
    name = name.lower()
    if name in ("segresnet", "segresnet_ibn"):
        return SegResNet(spatial_dims=2, in_channels=in_ch, out_channels=feat,
                         init_filters=init_filters, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
                         norm=norm)
    if name == "mednext":
        return MedNeXt(spatial_dims=2, in_channels=in_ch, out_channels=feat,
                       init_filters=init_filters, kernel_size=5, blocks_down=(2, 2, 2, 2),
                       blocks_up=(2, 2, 2, 2), norm_type="group")
    raise ValueError(f"unknown trunk {name}")


# ----- GPU label-derived targets (label: [B,1,H,W] long, monotonic 0..C-1) -----

def boundary_rows(label, n_bound):
    """normalized (0..1) row position of each of the n_bound layer boundaries, per column.
    boundary k = fraction of the column with class <= k. Returns [B, n_bound, W]."""
    lab = label[:, 0]                                   # [B,H,W]
    H = lab.shape[1]
    rows = [(lab <= k).sum(dim=1).float() / H for k in range(n_bound)]
    return torch.stack(rows, dim=1).clamp(0, 1)          # [B,n_bound,W]


def sdm_target(label, n_bound):
    """per-pixel normalized distance to the nearest layer boundary (rows). [B,1,H,W]."""
    br = boundary_rows(label, n_bound) * label.shape[2]  # actual rows [B,n_bound,W]
    H = label.shape[2]
    rr = torch.arange(H, device=label.device, dtype=torch.float32).view(1, 1, H, 1)
    d = (rr - br.unsqueeze(2)).abs()                     # [B,n_bound,H,W]
    return (d.min(dim=1)[0].unsqueeze(1) / H)            # [B,1,H,W]


class FlattenModule(nn.Module):
    """Learned, supervised retinal flattening as a complementary path. Predicts a
    per-column vertical offset (supervised by the GT retina centroid), warps the trunk
    features to centre the retina, segments the flattened features with a 2nd head, then
    un-warps. Combined with the plain head (which never flattens) as a safety net — a bad
    offset can't tank the output. Differentiable end-to-end; no hard reference detection."""

    def __init__(self, feat, num_classes):
        super().__init__()
        self.off = nn.Sequential(
            nn.Conv1d(feat, feat, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(feat, 1, 5, padding=2))
        self.seg2 = nn.Conv2d(feat, num_classes, 1)

    def _grid(self, B, H, W, off, dev, dtype, sign):
        ys = torch.linspace(-1, 1, H, device=dev, dtype=dtype)
        xs = torch.linspace(-1, 1, W, device=dev, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx.expand(B, H, W), gy.expand(B, H, W)], dim=-1).clone()
        grid[..., 1] = grid[..., 1] + sign * off[:, None, :]      # shift y per column
        return grid

    def forward(self, feat):
        B, C, H, W = feat.shape
        off = self.off(feat.mean(dim=2)).squeeze(1)               # [B,W] normalized offset
        off = torch.tanh(off)                                     # bound to [-1,1]
        f_flat = F.grid_sample(feat, self._grid(B, H, W, off, feat.device, feat.dtype, +1),
                               mode="bilinear", padding_mode="border", align_corners=True)
        flat_logits = self.seg2(f_flat)
        # un-warp the flattened-space logits back to original geometry
        unwarp = F.grid_sample(flat_logits, self._grid(B, H, W, off, feat.device, flat_logits.dtype, -1),
                               mode="bilinear", padding_mode="border", align_corners=True)
        return unwarp, off


def flatten_offset_target(label, n_classes=10):
    """normalized per-column shift that moves the retina centroid to the image centre."""
    lab = label[:, 0]                                            # [B,H,W]
    H = lab.shape[1]
    retina = ((lab >= 1) & (lab <= n_classes - 2)).float()       # intra-retinal layers
    rows = torch.arange(H, device=lab.device, dtype=torch.float32).view(1, H, 1)
    denom = retina.sum(dim=1).clamp_min(1.0)                     # [B,W]
    centroid = (retina * rows).sum(dim=1) / denom               # [B,W] in pixels
    return (H / 2 - centroid) / (H / 2)                          # normalized shift to centre


class MultiHeadSeg(nn.Module):
    def __init__(self, backbone="segresnet", in_ch=1, num_classes=10, feat=32,
                 norm="instance", use_sdm=True, use_surface=True, init_filters=16,
                 use_deform=False, use_flatten=False):
        super().__init__()
        self.num_classes = num_classes
        self.n_bound = num_classes - 1
        self.use_sdm, self.use_surface = use_sdm, use_surface
        self.trunk = build_trunk(backbone, in_ch, feat, norm, init_filters)
        self.deform = DeformRefine(feat) if use_deform else None
        self.seg = nn.Conv2d(feat, num_classes, 1)
        self.flatten = FlattenModule(feat, num_classes) if use_flatten else None
        self.sdm = nn.Conv2d(feat, 1, 1) if use_sdm else None
        self.surf = nn.Conv2d(feat, self.n_bound, 1) if use_surface else None

    def surface_rows(self, surf_logits):
        """expected normalized boundary row per column from a [B,n_bound,H,W] map."""
        p = torch.softmax(surf_logits, dim=2)            # softmax over rows
        H = surf_logits.shape[2]
        idx = torch.arange(H, device=surf_logits.device, dtype=p.dtype).view(1, 1, H, 1) / H
        return (p * idx).sum(dim=2)                      # [B,n_bound,W]

    def forward(self, x):
        f = self.trunk(x)
        if self.deform is not None:
            f = self.deform(f)
        main = self.seg(f)
        if self.flatten is not None:
            flat_unwarp, off = self.flatten(f)
            out = {"seg": 0.5 * (main + flat_unwarp), "flat_offset": off}
        else:
            out = {"seg": main}
        if self.sdm is not None:
            out["sdm"] = torch.sigmoid(self.sdm(f))      # in [0,1] (normalized dist)
        if self.surf is not None:
            out["surf_logits"] = self.surf(f)
            out["surf_rows"] = self.surface_rows(out["surf_logits"])
        return out


def head_losses(out, label, n_bound, w_sdm=0.5, w_surface=0.5, w_flatten=0.5, n_classes=10):
    """auxiliary head losses (GPU targets). Returns (total, parts)."""
    parts, total = {}, label.new_zeros((), dtype=torch.float32)
    if "sdm" in out and w_sdm:
        tgt = sdm_target(label, n_bound)
        l = F.smooth_l1_loss(out["sdm"], tgt)
        parts["sdm"] = l.detach(); total = total + w_sdm * l
    if "surf_rows" in out and w_surface:
        tgt = boundary_rows(label, n_bound)
        l = F.smooth_l1_loss(out["surf_rows"], tgt)
        parts["surface"] = l.detach(); total = total + w_surface * l
    if "flat_offset" in out and w_flatten:                       # supervise learned flattening
        tgt = flatten_offset_target(label, n_classes)
        l = F.smooth_l1_loss(out["flat_offset"], tgt)
        parts["flatten"] = l.detach(); total = total + w_flatten * l
    return total, parts
