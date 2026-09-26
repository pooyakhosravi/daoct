"""Mean-Teacher (Tarvainen & Valpola) EMA teacher + consistency on unlabeled targets.

The teacher is an EMA copy of the student. On unlabeled multi-vendor B-scans we push
the student's (strongly-augmented) prediction toward the teacher's (weakly-augmented)
prediction, masked by teacher confidence. Pulls target-vendor features toward
confident, stable predictions without any target labels.
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmaTeacher:
    def __init__(self, student: nn.Module, decay=0.99):
        self.decay = decay
        self.teacher = copy.deepcopy(student).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module):
        d = self.decay
        for tp, sp in zip(self.teacher.parameters(), student.parameters()):
            tp.mul_(d).add_(sp.detach(), alpha=1 - d)
        for tb, sb in zip(self.teacher.buffers(), student.buffers()):
            tb.copy_(sb)

    @torch.no_grad()
    def predict(self, x):
        return self.teacher(x)


def consistency_loss(
    student_logits,
    teacher_logits,
    conf_thresh=0.7,
    mode="mse",
    class_balance=False,
    boundary_weight=0.0,
    boundary_radius=2,
    boundary_axes="both",
):
    """Confidence-masked consistency on unlabeled secure-domain images.

    ``mse`` preserves the V3 behavior. ``pseudo_ce`` gives confident teacher
    predictions enough gradient to matter relative to the supervised boundary
    objective; optional per-batch class balancing protects thin layers.
    """
    sp = torch.softmax(student_logits, dim=1)
    tp = torch.softmax(teacher_logits, dim=1)
    conf, pseudo = tp.max(dim=1)
    mask = conf >= conf_thresh
    if mode == "mse":
        se = ((sp - tp) ** 2).mean(dim=1)
        return (se * mask).sum() / mask.sum().clamp_min(1)
    if mode == "kl":
        per_pixel = F.kl_div(
            torch.log_softmax(student_logits, dim=1),
            tp,
            reduction="none",
        ).sum(dim=1)
    elif mode == "pseudo_ce":
        per_pixel = F.cross_entropy(
            student_logits, pseudo, reduction="none"
        )
    else:
        raise ValueError(f"unknown consistency mode: {mode}")
    weights = mask.float()
    if class_balance and mask.any():
        classes = student_logits.shape[1]
        counts = torch.bincount(
            pseudo[mask], minlength=classes
        ).float().clamp_min(1)
        inverse = counts.sum() / (classes * counts)
        inverse = inverse.clamp(0.25, 4.0)
        weights = weights * inverse[pseudo]
    if boundary_weight > 0 and mode == "pseudo_ce":
        boundary = torch.zeros_like(mask, dtype=torch.float32)
        axial = (pseudo[:, 1:, :] != pseudo[:, :-1, :]).float()
        boundary[:, 1:, :] = torch.maximum(boundary[:, 1:, :], axial)
        boundary[:, :-1, :] = torch.maximum(boundary[:, :-1, :], axial)
        if boundary_axes == "both":
            lateral = (pseudo[:, :, 1:] != pseudo[:, :, :-1]).float()
            boundary[:, :, 1:] = torch.maximum(
                boundary[:, :, 1:], lateral
            )
            boundary[:, :, :-1] = torch.maximum(
                boundary[:, :, :-1], lateral
            )
        elif boundary_axes != "axial":
            raise ValueError(
                f"unknown pseudo boundary axes: {boundary_axes}"
            )
        radius = max(0, int(boundary_radius))
        if radius:
            kernel = 2 * radius + 1
            boundary = F.max_pool2d(
                boundary.unsqueeze(1),
                kernel_size=kernel,
                stride=1,
                padding=radius,
            ).squeeze(1)
        weights = weights * (1.0 + float(boundary_weight) * boundary)
    return (per_pixel * weights).sum() / weights.sum().clamp_min(1.0)


def surface_consensus_loss(
    student_logits,
    teacher_logits,
    teacher_surfaces,
    student_surfaces=None,
    conf_thresh=0.75,
    class_balance=True,
    surface_weight=0.25,
):
    """Topology-safe pseudo labels from agreement of segmentation and surface heads."""
    batch, classes, height, width = student_logits.shape
    surfaces = torch.sort(
        teacher_surfaces.detach().clamp(0, 1), dim=1
    ).values
    rows = torch.arange(
        height,
        device=student_logits.device,
        dtype=surfaces.dtype,
    ).view(1, 1, height, 1) / height
    surface_pseudo = (
        rows >= surfaces.unsqueeze(2)
    ).sum(dim=1).long()
    teacher_probability = torch.softmax(teacher_logits, dim=1)
    confidence, segmentation_pseudo = teacher_probability.max(dim=1)
    mask = (
        (confidence >= conf_thresh)
        & (segmentation_pseudo == surface_pseudo)
    )
    per_pixel = F.cross_entropy(
        student_logits, surface_pseudo, reduction="none"
    )
    weights = mask.float()
    if class_balance and mask.any():
        counts = torch.bincount(
            surface_pseudo[mask], minlength=classes
        ).float().clamp_min(1)
        inverse = (counts.sum() / (classes * counts)).clamp(0.25, 4.0)
        weights = weights * inverse[surface_pseudo]
    loss = (per_pixel * weights).sum() / weights.sum().clamp_min(1.0)
    if student_surfaces is not None and surface_weight:
        loss = loss + surface_weight * F.smooth_l1_loss(
            student_surfaces, surfaces
        )
    return loss
