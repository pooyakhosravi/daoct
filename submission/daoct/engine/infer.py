"""Challenge-format inference: input dir of `*-image.png` -> output dir of `*-mask.png`.

Image-only input (no device/disease metadata), matching the challenge contract.
Predicts at the model's working size, then resizes the label map back to the native
input resolution with NEAREST (the evaluator also resizes preds to GT with nearest).
"""
from __future__ import annotations
import argparse, glob, os, re
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from daoct.models.build import build_model
from daoct.models.load import load_model, seg_out


def _preprocess(path, size, mode, norm=False):
    from daoct.data.dataset import apply_norm
    arr = np.asarray(Image.open(path).convert("L"))
    h0, w0 = arr.shape
    # Registered Spectralis cutoff A-scans are annotated as class 0. Detect them
    # before interpolation, which otherwise leaks neighboring signal into a
    # genuinely black native column.
    native_empty = arr.max(axis=0) == 0
    if mode in ("fixed_h", "fixed_h16"):
        w = max(1, round(w0 * size / h0))
        if mode == "fixed_h16":
            w = max(16, int(round(w / 16.0)) * 16)
        im = Image.fromarray(arr).resize((w, size), Image.BILINEAR)
    else:
        im = Image.fromarray(arr).resize((size, size), Image.BILINEAR)
    x = apply_norm(np.asarray(im), norm)                                 # match train-time normalization
    return x, (h0, w0), native_empty


import torch.nn.functional as F


def flip_prob(model, x):
    """Average identity and horizontal flip; depth order is never flipped."""
    base = torch.softmax(seg_out(model(x)), 1)
    flipped = torch.softmax(seg_out(model(torch.flip(x, [-1]))), 1)
    return 0.5 * (base + torch.flip(flipped, [-1]))


def axial_scale_prob(model, x, scales=(0.9, 1.15), include_identity=True):
    """Average identity with axial-only rescalings at fixed lateral sampling."""
    H, W = x.shape[-2:]
    acc = torch.softmax(seg_out(model(x)), 1) if include_identity else None
    n = 1 if include_identity else 0
    for s in scales:
        nh = max(16, int(round(H * s / 16)) * 16)   # keep divisible by 16 for the trunk
        xs = F.interpolate(x, size=(nh, W), mode="bilinear", align_corners=False)
        ps = torch.softmax(seg_out(model(xs)), 1)
        ps = F.interpolate(ps, size=(H, W), mode="bilinear", align_corners=False)
        acc = ps if acc is None else acc + ps
        n += 1
    return acc / n


def lateral_scale_prob(model, x, scales=(0.85, 1.15), include_flip=True):
    """Average mild lateral rescalings, mapped back to the trained square grid."""
    height, width = x.shape[-2:]
    base = torch.softmax(seg_out(model(x)), 1)
    acc = base
    count = 1
    if include_flip:
        flipped = torch.softmax(seg_out(model(torch.flip(x, [-1]))), 1)
        acc = acc + torch.flip(flipped, [-1])
        count += 1
    for scale in scales:
        scaled_width = max(
            16, int(round(width * float(scale) / 16)) * 16
        )
        scaled = F.interpolate(
            x,
            size=(height, scaled_width),
            mode="bilinear",
            align_corners=False,
        )
        probability = torch.softmax(seg_out(model(scaled)), 1)
        acc = acc + F.interpolate(
            probability,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        count += 1
    return acc / count


def tta_prob(model, x):
    """Legacy combined TTA: identity, horizontal flip, and two axial scales."""
    H, W = x.shape[-2:]
    base = torch.softmax(seg_out(model(x)), 1)
    flipped = torch.softmax(seg_out(model(torch.flip(x, [-1]))), 1)
    acc = base + torch.flip(flipped, [-1])
    n = 2
    for scale in (0.9, 1.15):
        nh = max(16, int(round(H * scale / 16)) * 16)
        scaled = F.interpolate(
            x, size=(nh, W), mode="bilinear", align_corners=False
        )
        probability = torch.softmax(seg_out(model(scaled)), 1)
        acc += F.interpolate(
            probability, size=(H, W), mode="bilinear", align_corners=False
        )
        n += 1
    return acc / n


def multiscale_prob(model, x, sizes=(640, 896, 1152)):
    """Average softmax over several input resolutions. The high scales give compressed,
    steeply-tilted layers (widefield/12x12) enough pixels to resolve into 8 distinct
    layers — the resolution limit that augmentation/deform alone can't fix. x:[1,1,H,W]."""
    H, W = x.shape[-2:]
    acc = None
    for s in sizes:
        s = max(64, (s // 16) * 16)                      # keep divisible by 16
        xs = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        p = torch.softmax(seg_out(model(xs)), 1)
        p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
        acc = p if acc is None else acc + p
    return acc / len(sizes)


@torch.no_grad()
def run(model, input_dir, output_dir, device, size=512, mode="square", refine=False,
        tta=False, mask_empty=True, multiscale=None, norm=False, smooth=True,
        smooth_k=9, anchor_model=None, presence_guard_label=2):
    from daoct.da.surface_refine import dp_refine_prob, mask_empty_columns, smooth_surfaces
    os.makedirs(output_dir, exist_ok=True)
    # Robust to filename convention: organizers warned val files "might have different
    # file names ending in .png". Prefer the template's *-image.png; if none, take every
    # .png that is not a mask. Output preserves the input basename's mask-name.
    paths = sorted(glob.glob(os.path.join(input_dir, "**", "*-image.png"), recursive=True))
    if not paths:
        paths = sorted(q for q in glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True)
                       if not q.endswith("-mask.png"))
    guarded = 0
    print(
        f"[infer] {len(paths)} images (refine={refine} tta={tta} "
        f"anchor={anchor_model is not None})"
    )
    for p in paths:
        x, (h0, w0), native_empty = _preprocess(p, size, mode, norm=norm)
        xt = torch.from_numpy(x)[None, None].to(device)

        def probability(current_model):
            if multiscale:
                return multiscale_prob(
                    current_model, xt, sizes=multiscale
                )
            if not tta:
                return torch.softmax(
                    seg_out(current_model(xt)), dim=1
                )
            if tta == "flip":
                return flip_prob(current_model, xt)
            if tta == "axial":
                return axial_scale_prob(current_model, xt)
            lateral = {
                "flip_lat75": (0.75,),
                "flip_lat85": (0.85,),
                "flip_lat115": (1.15,),
                "flip_lat125": (1.25,),
                "flip_lat135": (1.35,),
                "flip_lat150": (1.50,),
                "flip_lat175": (1.75,),
                "flip_lat200": (2.00,),
                "flip_lat115_125": (1.15, 1.25),
                "flip_lat85_115": (0.85, 1.15),
            }
            if tta in lateral:
                return lateral_scale_prob(
                    current_model, xt, scales=lateral[tta]
                )
            return tta_prob(current_model, xt)

        def decode(probability_tensor):
            if refine:
                decoded = dp_refine_prob(
                    probability_tensor[0].cpu().numpy()
                )
                if smooth:
                    empty = (
                        x.max(axis=0) < 0.06
                    ) if mask_empty else None
                    decoded = smooth_surfaces(
                        decoded, ksize=smooth_k, empty=empty
                    )
            else:
                decoded = torch.argmax(
                    probability_tensor, dim=1
                )[0].to(torch.uint8).cpu().numpy()
            if mask_empty:
                decoded = mask_empty_columns(decoded, x)
            decoded = np.asarray(
                Image.fromarray(decoded).resize(
                    (w0, h0), Image.NEAREST
                )
            ).astype(np.uint8)
            if mask_empty and native_empty.any():
                decoded[:, native_empty] = 0
            return decoded

        pred = decode(probability(model))
        if anchor_model is not None:
            anchor_pred = decode(probability(anchor_model))
            # FOV training may legitimately remove optional layers, but it
            # should not invent class 2 when the pre-FOV semantic anchor says
            # that interface is absent. This image-level rule requires no
            # device, protocol, or disease metadata.
            if (
                not np.any(anchor_pred == presence_guard_label)
                and np.any(pred == presence_guard_label)
            ):
                pred = anchor_pred
                guarded += 1
        # output name = "<id>-mask.png" where id is the input stem minus any image marker.
        # The scorer matches predictions by `<id>-mask.png` regardless of input naming.
        stem = os.path.splitext(os.path.basename(p))[0]
        stem = re.sub(r"[-_]image$", "", stem, flags=re.IGNORECASE)
        Image.fromarray(pred).save(os.path.join(output_dir, f"{stem}-mask.png"))
    print(f"[infer] done guarded={guarded}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model", default="segresnet")
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--size", type=int, default=None, help="override; else from model_meta")
    ap.add_argument("--mode", default="auto", help="auto uses model_meta resize_mode")
    ap.add_argument("--refine", action="store_true", help="DP ordered-surface refinement")
    ap.add_argument("--tta", action="store_true", help="test-time augmentation (#7)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(a.model_path, dev, fallback=a.model, num_classes=a.num_classes)
    size = a.size or meta.get("size", 512)
    mode = a.mode if a.mode != "auto" else meta.get("resize_mode", "square")
    run(model, a.input_dir, a.output_dir, dev, size, mode, a.refine, a.tta)


if __name__ == "__main__":
    main()
