"""Fourier Domain Adaptation (Yang & Soatto, CVPR'20).

Swap the low-frequency amplitude of a source (labeled Spectralis) B-scan with that
of a target-vendor B-scan. Cheap style transfer, no GAN, no target labels -> makes
the supervised stream look like other vendors and helps zero-shot generalization
to the unseen Triton. Phase (structure) is preserved, only amplitude (style) changes.
"""
from __future__ import annotations
import torch


def _low_freq_mask(h, w, beta, device):
    # square window of half-size b around the (shifted) spectrum center
    b = max(1, int(min(h, w) * beta))
    m = torch.zeros(h, w, device=device)
    cy, cx = h // 2, w // 2
    m[cy - b:cy + b + 1, cx - b:cx + b + 1] = 1.0
    return m


def fda_swap(src, tgt, beta=0.01):
    """src, tgt: [B,1,H,W] float in [0,1] (same spatial size). Returns styled src."""
    assert src.shape == tgt.shape, "FDA needs matching shapes (resize first)"
    B, C, H, W = src.shape
    fs = torch.fft.fftshift(torch.fft.fft2(src, dim=(-2, -1)), dim=(-2, -1))
    ft = torch.fft.fftshift(torch.fft.fft2(tgt, dim=(-2, -1)), dim=(-2, -1))
    amp_s, pha_s = fs.abs(), fs.angle()
    amp_t = ft.abs()
    m = _low_freq_mask(H, W, beta, src.device).view(1, 1, H, W)
    amp_new = amp_s * (1 - m) + amp_t * m
    fs_new = amp_new * torch.exp(1j * pha_s)
    out = torch.fft.ifft2(torch.fft.ifftshift(fs_new, dim=(-2, -1)), dim=(-2, -1)).real
    return out.clamp(0, 1)
