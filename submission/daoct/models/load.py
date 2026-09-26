"""Rebuild a model from a checkpoint + its sidecar model_meta.json (written by train()).

Lets inference/eval reconstruct the exact architecture (multi-head or plain, norm,
backbone) without the caller knowing it. Falls back to a plain SegResNet if no meta."""
from __future__ import annotations
import json
from pathlib import Path
import torch

from daoct.models.build import make_model, build_model


def seg_out(out):
    """multi-head models return a dict; plain nets return the seg tensor."""
    return out["seg"] if isinstance(out, dict) else out


def load_model(model_path, device, fallback="segresnet", num_classes=10):
    meta_p = Path(model_path).parent / "model_meta.json"
    if meta_p.exists():
        meta = json.load(open(meta_p))
        cfg = dict(model=meta.get("model", "segresnet"),
                   num_classes=meta.get("num_classes", num_classes),
                   multihead=meta.get("multihead", False), norm=meta.get("norm", "instance"),
                   feat=meta.get("feat", 32), use_sdm=meta.get("use_sdm", True),
                   use_surface=meta.get("use_surface", True),
                   use_deform=meta.get("use_deform", False),
                   use_flatten=meta.get("use_flatten", False),
                   trunk_filters=meta.get("trunk_filters", 16) or 16, init_filters=32)
        model = make_model(cfg)
    else:
        meta = {"size": 512, "resize_mode": "square"}
        model = build_model(fallback, num_classes=num_classes)
    try:
        state = torch.load(
            model_path, map_location=device, weights_only=True
        )
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval(), meta
