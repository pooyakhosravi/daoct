"""Submission training (template-named; called as main(Namespace(data_root=...))).

Pre-resizes the server's train tree to a fixed size once (so loading is cheap and the
run fits the 2h limit), then trains the ablation-winning multi-head model. Labeled
device = Topcon_Maestro2; every other device dir = unlabeled DA. Saves the checkpoint
under checkpoints/unet_maestro2_semi.pth (+ model_meta.json) where main.py expects it.
"""
import os
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from daoct.engine.train import train          # noqa: E402
from preresize import preresize_tree          # noqa: E402

LABELED_DEVICE = "Topcon_Maestro2"
CKPT_NAME = "unet_maestro2_semi.pth"
ANCHOR_NAME = "unet_maestro2_anchor.pth"

# Final config (inlined as a dict so no PyYAML dependency at runtime).
# trunk16 (footprint bonus; matched A/B: augs carry the gain, trunk32 only +0.005 -> not worth it).
# Levers, each validated or principled:
#   geometry  : paired-geom + tilt(rotate) + curvature0.85 + steep_edge + crop_zoom
#   intensity : domain-rand + device_intensity(SS-OCT depth/contrast) + FDA + per-image percentile_norm
#   DA        : mean-teacher consistency, vendor-balanced unlabeled
#   steep_edge: +0.008 on near-vertical-wall stress (widefield = 50% of score); norm: device x protocol
#               intensity insurance (server test is heterogeneous, unmeasurable on Maestro2-only val).
# Inference applies DP ordered-surface + despike lateral smoothing + empty-col mask + matching norm.
CONFIG = dict(
    num_classes=10, size=512, resize_mode="square", val_frac=0.12,
    model="segresnet", multihead=True, norm="instance", trunk_filters=16, feat=32,
    use_sdm=True, use_surface=True, pretrained_trunk=None,
    w_dicece=1.0, boundary_type="gradient", w_boundary=10.0, w_order=0.1,
    w_sdm=0.5, w_surface=1.0,
    semi=True, balance_unlabeled=True, geom_aug=True, geom_vscale=[0.6, 1.5],
    geom_hscale=[0.55, 1.1], geom_prob=0.7,                 # lateral compress -> wider FOV
    curvature_aug=True, curv_amp=0.85, curv_prob=0.7,      # smooth widefield/12x12 U-curves
    steep_edge=True, steep_prob=0.4,                       # near-vertical peripheral walls (widefield)
    ascan_compress=True, ascomp_prob=0.6, ascomp_max=2.2,  # axial foreshortening: few-pixel layers at steep sides
    edge_blackout=True, edge_blackout_prob=0.5, dr_strength=1.0,
    rotate_aug=True, rot_deg=15.0, rot_prob=0.5,           # tilt / off-axis (diagonal retina)
    crop_zoom=True, cz_prob=0.35,                          # patch-detail (thin-layer sharpness)
    lesion_aug=True, lesion_prob=0.3,                      # intensity-blob robustness (label intact)
    device_intensity=True, dev_int_prob=0.7,              # SS-OCT depth gain + inter-layer contrast (Triton)
    input_norm=True,                                       # per-image percentile (p2-p98) -> device/protocol invariance
    use_fda=True, fda_prob=0.5, fda_beta=0.01, ema_decay=0.99, conf_thresh=0.7,
    w_cons=1.0, cons_rampup=500,
    steps=8000, batch_size=16, lr=0.0006, wd=0.0001, amp=True, workers=0, prefetch=2,  # trunk16 ~3it/s server -> well within 2h
    unlabeled_cache=6000,   # workers=0 -> single process, so caching is leak-free + fast
    log_every=200, val_every=1000, val_batches=None,
)


def interpolate_checkpoints(base_path, adapted_path, output_path, alpha):
    """Deploy a conservative soup between pre- and post-FOV weights."""
    try:
        base = torch.load(base_path, map_location="cpu", weights_only=True)
        adapted = torch.load(
            adapted_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        base = torch.load(base_path, map_location="cpu")
        adapted = torch.load(adapted_path, map_location="cpu")
    if base.keys() != adapted.keys():
        raise RuntimeError("FOV continuation changed checkpoint structure")
    mixed = {}
    for key, base_value in base.items():
        adapted_value = adapted[key]
        if (
            base_value.shape != adapted_value.shape
            or base_value.dtype != adapted_value.dtype
        ):
            raise RuntimeError(f"incompatible checkpoint tensor: {key}")
        if torch.is_floating_point(base_value):
            mixed[key] = torch.lerp(base_value, adapted_value, alpha)
        else:
            mixed[key] = (
                adapted_value.clone() if alpha >= 0.5 else base_value.clone()
            )
    torch.save(mixed, output_path)


def main(args):
    data_root = Path(args.data_root)
    # train tree may be data_root or data_root/train
    if (data_root / "train").exists():
        data_root = data_root / "train"

    ckpt_dir = HERE / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(CONFIG)
    if getattr(args, "steps", None):
        cfg["steps"] = args.steps
    cfg["steps"] = int(os.environ.get("DAOCT_STEPS", cfg["steps"]))   # platform/test override

    proc = ckpt_dir / "proc"
    print(f"[train] pre-resizing {data_root} -> {proc}", flush=True)
    preresize_tree(data_root, proc, cfg["size"], cfg["resize_mode"], workers=8)

    sup = proc / LABELED_DEVICE
    assert sup.exists(), f"labeled device dir not found: {sup}"
    cfg["sup_roots"] = [str(sup)]
    cfg["unlabeled_roots"] = [str(p) for p in sorted(proc.iterdir())
                              if p.is_dir() and p.name != LABELED_DEVICE]
    cfg["ckpt_dir"] = str(ckpt_dir)
    cfg["return_details"] = True
    print(f"[train] labeled={cfg['sup_roots']}", flush=True)
    print(f"[train] unlabeled={cfg['unlabeled_roots']}", flush=True)

    stage1_result = train(cfg)

    # Refine only weights learned from the secure data. The focused second stage
    # keeps V2's proven aspect/curvature transforms, removes the extra V3
    # perturbations, and uses target images through pseudo labels without FDA.
    refine_dir = ckpt_dir / "refine"
    stage1 = (
        ckpt_dir / "best.pth"
        if (ckpt_dir / "best.pth").exists()
        else ckpt_dir / "last.pth"
    )
    refine = dict(cfg)
    refine.update(
        init_checkpoint=str(stage1),
        ckpt_dir=str(refine_dir),
        steps=int(os.environ.get("DAOCT_REFINE_STEPS", "500")),
        lr=0.00015,
        cons_rampup=100,
        val_every=500,
        log_every=100,
        rotate_aug=False,
        steep_edge=False,
        ascan_compress=False,
        crop_zoom=False,
        lesion_aug=False,
        device_intensity=False,
        anatomy_aug=False,
        consistency_mode="pseudo_ce",
        cons_class_balance=True,
        pseudo_boundary_weight=2.0,
        pseudo_boundary_radius=2,
        pseudo_boundary_axes="axial",
        w_cons=0.25,
        conf_thresh=0.85,
        use_fda=False,
        noise_transfer=False,
    )
    print("[train] secure-data stage-2 refinement", flush=True)
    refine_result = train(refine)

    # Promote target refinement only when it remains source-valid on the
    # server's labeled split. Otherwise preserve the exact V3 student.
    min_delta = float(os.environ.get("DAOCT_REFINE_MIN_DELTA", "0.0"))
    promote_refine = (
        refine_result["selected_score"]
        >= stage1_result["student_score"] + min_delta
    )
    if promote_refine:
        src = refine_dir / "selected.pth"
        if not src.exists():
            src = refine_dir / "last.pth"
        refine_meta = refine_dir / "model_meta.json"
        if refine_meta.exists():
            shutil.copyfile(refine_meta, ckpt_dir / "model_meta.json")
    else:
        src = ckpt_dir / "best.pth"
        if not src.exists():
            src = ckpt_dir / "last.pth"
    print(
        "[train] refinement gate "
        f"stage1_student={stage1_result['student_score']:.4f} "
        f"refine={refine_result['selected_score']:.4f} "
        f"min_delta={min_delta:.4f} promoted={promote_refine}",
        flush=True,
    )

    # Apply the FOV continuation only after secure target refinement is valid.
    deployed = ckpt_dir / CKPT_NAME
    anchor = ckpt_dir / ANCHOR_NAME
    if anchor.exists():
        anchor.unlink()
    if promote_refine:
        fov_dir = ckpt_dir / "fov"
        fov = dict(refine)
        fov.update(
            init_checkpoint=str(src),
            ckpt_dir=str(fov_dir),
            steps=int(os.environ.get("DAOCT_FOV_STEPS", "250")),
            lr=0.000075,
            val_every=int(os.environ.get("DAOCT_FOV_STEPS", "250")),
            log_every=125,
            seed=20260719,
            semi=False,
            use_fda=False,
            anatomy_aug=True,
            anatomy_modes="onh_continuous",
            anatomy_prob=0.12,
            anatomy_severity=0.65,
            fov_exit_aug=True,
            fov_exit_prob=0.10,
            fov_exit_min_shift=0.55,
            fov_exit_max_shift=1.15,
            fov_exit_top_prob=0.85,
            fov_exit_max_tilt=0.08,
        )
        print("[train] supervised FOV/continuous-ONH cooldown", flush=True)
        try:
            fov_result = train(fov)
            max_source_drop = float(
                os.environ.get("DAOCT_FOV_MAX_SOURCE_DROP", "0.01")
            )
            promote_fov = (
                fov_result["selected_score"]
                >= refine_result["selected_score"] - max_source_drop
            )
            if promote_fov:
                fov_src = fov_dir / "selected.pth"
                if not fov_src.exists():
                    fov_src = fov_dir / "last.pth"
                soup_alpha = float(
                    os.environ.get("DAOCT_FOV_SOUP_ALPHA", "0.50")
                )
                if not 0.0 <= soup_alpha <= 1.0:
                    raise ValueError(
                        "DAOCT_FOV_SOUP_ALPHA must be in [0, 1]"
                    )
                interpolate_checkpoints(
                    src, fov_src, deployed, soup_alpha
                )
                # Retain the pre-FOV semantic anchor for the optional-layer
                # presence guard at inference. Both checkpoints were learned
                # entirely within this secure run.
                shutil.copyfile(src, anchor)
                fov_meta = fov_dir / "model_meta.json"
                if fov_meta.exists():
                    shutil.copyfile(
                        fov_meta, ckpt_dir / "model_meta.json"
                    )
            else:
                shutil.copyfile(src, deployed)
            print(
                "[train] FOV gate "
                f"refine={refine_result['selected_score']:.4f} "
                f"fov={fov_result['selected_score']:.4f} "
                f"max_drop={max_source_drop:.4f} "
                f"promoted={promote_fov}",
                flush=True,
            )
            if promote_fov:
                print(
                    f"[train] FOV soup alpha={soup_alpha:.2f}",
                    flush=True,
                )
        except Exception as exc:
            shutil.copyfile(src, deployed)
            if anchor.exists():
                anchor.unlink()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                "[train] FOV cooldown failed; preserving refinement "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
    else:
        shutil.copyfile(src, deployed)
    print(f"[train] saved {ckpt_dir / CKPT_NAME} (+ model_meta.json)", flush=True)


if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser(); p.add_argument("--data_root", required=True); main(p.parse_args())
