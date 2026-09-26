"""Training engine: supervised + domain-randomization + FDA + mean-teacher consistency.

Optimizes the challenge objective directly (DiceCE + boundary + ordering on labeled
source; EMA-teacher consistency + ordering prior on unlabeled multi-vendor target).
"""
from __future__ import annotations
import itertools, os, shutil, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as nnf
from torch.utils.data import DataLoader

from daoct.data.dataset import SupervisedSeg, UnlabeledSeg
from daoct.data.transforms import (domain_randomize, weak_aug, strong_aug,
                                    paired_geom_aug, edge_blackout, curvature_warp,
                                    rotate_aug, crop_zoom_aug, lesion_aug, pathology_warp,
                                    steep_edge_warp, device_intensity_aug,
                                    thickness_jitter, elastic_aug, noise_transfer,
                                    ascan_compress, anatomy_warp, axial_fov_exit)
from daoct.da.fda import fda_swap
from daoct.da.mean_teacher import (
    EmaTeacher,
    consistency_loss,
    surface_consensus_loss,
)
from daoct.losses.seg_losses import CombinedSegLoss
from daoct.models.build import make_model, count_params
from daoct.models.multihead import head_losses
from daoct.eval.metrics import compute_image_score


def _seg(out):
    """multi-head models return a dict; plain nets return the seg tensor."""
    return out["seg"] if isinstance(out, dict) else out


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def evaluate(model, loader, device, max_batches=None):
    was_training = model.training
    model.eval(); scores = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["image"].to(device); y = batch["label"].numpy()[:, 0]
            pred = torch.argmax(_seg(model(x)), dim=1).cpu().numpy()
            for p, g in zip(pred, y):
                scores.append(compute_image_score(p.astype(np.uint8), g.astype(np.uint8)))
            if max_batches and i + 1 >= max_batches:
                break
    model.train(was_training)
    return float(np.mean(scores)) if scores else 0.0


def _hard_example_weights(
    model,
    dataset,
    device,
    batch_size,
    workers,
    gamma=1.5,
    max_ratio=5.0,
    floor=0.02,
):
    """Rank source examples by the initialized checkpoint's challenge score."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    was_training = model.training
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].numpy()[:, 0]
            pred = torch.argmax(_seg(model(x)), dim=1).cpu().numpy()
            scores.extend(
                compute_image_score(
                    p.astype(np.uint8), g.astype(np.uint8)
                )
                for p, g in zip(pred, y)
            )
    model.train(was_training)
    score_array = np.asarray(scores, dtype=np.float64)
    raw = np.power(1.0 - score_array + float(floor), float(gamma))
    if max_ratio and max_ratio > 1:
        raw = np.maximum(raw, raw.max() / float(max_ratio))
    raw /= raw.mean()
    quantiles = np.quantile(score_array, (0.0, 0.1, 0.5, 0.9, 1.0))
    print(
        "[hard-mining] initialized source scores "
        + " ".join(
            f"{name}={value:.4f}"
            for name, value in zip(
                ("min", "p10", "median", "p90", "max"), quantiles
            )
        )
        + f" weight_ratio={raw.max() / raw.min():.2f}",
        flush=True,
    )
    return torch.as_tensor(raw, dtype=torch.double)


def _balanced_cross_entropy(
    logits,
    target,
    power=0.5,
    min_weight=0.35,
    max_weight=3.0,
):
    """Per-batch inverse-frequency CE with bounded thin-layer emphasis."""
    labels = target.squeeze(1).long()
    classes = logits.shape[1]
    counts = torch.bincount(
        labels.reshape(-1), minlength=classes
    ).to(logits.device, dtype=torch.float32)
    present = counts > 0
    weights = torch.ones(classes, device=logits.device)
    if present.any():
        inverse = counts[present].sum() / counts[present].clamp_min(1)
        inverse = inverse.pow(float(power))
        inverse /= inverse.mean().clamp_min(1e-6)
        weights[present] = inverse.clamp(
            float(min_weight), float(max_weight)
        )
    return nnf.cross_entropy(logits, labels, weight=weights)


def train(cfg):
    # Docker default /dev/shm is 64MB; DataLoader workers share tensors via shm and
    # fail with "unable to allocate shared memory". We can't change the platform's
    # docker flags. Route worker tensor sharing through the filesystem (defensive);
    # the submission also uses workers=0 (no IPC at all) to fully avoid shm.
    try:
        import torch.multiprocessing as _tmp
        _tmp.set_sharing_strategy("file_system")
    except Exception:
        pass
    seed = int(cfg.get("seed", 20260719))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sz, mode = cfg["size"], cfg["resize_mode"]

    norm = cfg.get("input_norm", False)
    sup = SupervisedSeg(cfg["sup_roots"], size=sz, mode=mode, norm=norm)
    assert len(sup) > 0, "no supervised pairs found"
    n_val = max(1, int(len(sup) * cfg.get("val_frac", 0.15)))
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(sup), generator=g).tolist()
    val_idx, tr_idx = set(perm[:n_val]), perm[n_val:]
    tr = torch.utils.data.Subset(sup, tr_idx)
    va = torch.utils.data.Subset(sup, list(val_idx))
    nw = cfg["workers"]
    dl = dict(num_workers=nw, pin_memory=True)
    if nw > 0:
        dl.update(persistent_workers=True, prefetch_factor=cfg.get("prefetch", 6))
    sup_loader = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True,
                            drop_last=True, **dl)
    val_loader = DataLoader(va, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=nw)

    semi = cfg.get("semi", True) and cfg.get("unlabeled_roots")
    if semi:
        ul = UnlabeledSeg(cfg["unlabeled_roots"], size=sz, mode=mode,
                          max_cache=cfg.get("unlabeled_cache", 8000), norm=norm)
        semi = len(ul) > 0
    if semi:
        if cfg.get("balance_unlabeled", True):
            from torch.utils.data import WeightedRandomSampler
            w = ul.balanced_weights()
            sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
            ul_loader = DataLoader(ul, batch_size=cfg["batch_size"], sampler=sampler,
                                   drop_last=True, **dl)
            nv = len(set(ul.vendors))
            print(f"[data] supervised={len(tr)} val={len(va)} unlabeled={len(ul)} "
                  f"(vendor-balanced over {nv} vendors)")
        else:
            ul_loader = DataLoader(ul, batch_size=cfg["batch_size"], shuffle=True,
                                   drop_last=True, **dl)
            print(f"[data] supervised={len(tr)} val={len(va)} unlabeled={len(ul)}")
        ul_iter = _cycle(ul_loader)
    else:
        print(f"[data] supervised={len(tr)} val={len(va)} (no unlabeled)")

    model = make_model(cfg).to(dev)
    init_checkpoint = cfg.get("init_checkpoint")
    if init_checkpoint:
        try:
            state = torch.load(
                init_checkpoint, map_location=dev, weights_only=True
            )
        except TypeError:
            state = torch.load(init_checkpoint, map_location=dev)
        model.load_state_dict(state)
        print(f"[model] initialized from {init_checkpoint}", flush=True)
    if cfg.get("hard_example_mining", False):
        from torch.utils.data import WeightedRandomSampler

        hard_weights = _hard_example_weights(
            model,
            tr,
            dev,
            batch_size=cfg["batch_size"],
            workers=nw,
            gamma=cfg.get("hard_mining_gamma", 1.5),
            max_ratio=cfg.get("hard_mining_max_ratio", 5.0),
            floor=cfg.get("hard_mining_floor", 0.02),
        )
        sampler = WeightedRandomSampler(
            hard_weights,
            num_samples=len(hard_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        sup_loader = DataLoader(
            tr,
            batch_size=cfg["batch_size"],
            sampler=sampler,
            drop_last=True,
            **dl,
        )
    mh = cfg.get("multihead", False)
    nb = cfg["num_classes"] - 1
    print(f"[model] {cfg['model']}{' +SDM+surface' if mh else ''} norm={cfg.get('norm','instance')} "
          f"params={count_params(model)/1e6:.2f}M")
    teacher = EmaTeacher(model, decay=cfg.get("ema_decay", 0.99)) if semi else None
    loss_fn = CombinedSegLoss(num_classes=cfg["num_classes"], w_dicece=cfg["w_dicece"],
                              w_boundary=cfg["w_boundary"], w_order=cfg["w_order"],
                              boundary_type=cfg.get("boundary_type", "gradient")).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["steps"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    sup_iter = _cycle(sup_loader)
    ckpt_dir = Path(cfg["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    # model meta so inference can rebuild the exact architecture
    import json
    meta = {k: cfg.get(k) for k in ("model", "num_classes", "multihead", "norm", "feat",
                                    "use_sdm", "use_surface", "use_deform", "use_flatten",
                                    "trunk_filters", "size", "resize_mode", "input_norm")}
    json.dump(meta, open(ckpt_dir / "model_meta.json", "w"), indent=2)
    best = -1.0
    best_teacher = -1.0
    t0 = time.time()
    for step in range(1, cfg["steps"] + 1):
        b = next(sup_iter)
        x = b["image"].to(dev, non_blocking=True); y = b["label"].to(dev, non_blocking=True)
        if cfg.get("geom_aug", True) and torch.rand(()) < cfg.get("geom_prob", 0.7):
            x, y = paired_geom_aug(x, y, vscale=tuple(cfg.get("geom_vscale", (0.75, 1.3))),
                                   hscale=tuple(cfg.get("geom_hscale", (0.9, 1.1))))
        if cfg.get("rotate_aug", False):              # tilted/off-axis B-scans (diagonal retina)
            x, y = rotate_aug(x, y, max_deg=cfg.get("rot_deg", 15.0),
                              prob=cfg.get("rot_prob", 0.5))
        if cfg.get("curvature_aug", True):            # simulate widefield/12x12 curvature
            x, y = curvature_warp(x, y, max_amp=cfg.get("curv_amp", 0.5),
                                  prob=cfg.get("curv_prob", 0.6))
        if cfg.get("steep_edge", False):              # near-vertical peripheral wall geometry
            x, y = steep_edge_warp(x, y, prob=cfg.get("steep_prob", 0.4))
        if cfg.get("ascan_compress", False):          # axial foreshortening (few-pixel layers at sides)
            x, y = ascan_compress(x, y, prob=cfg.get("ascomp_prob", 0.6),
                                  max_comp=cfg.get("ascomp_max", 2.0))
        if cfg.get("thickness_jitter", False):        # break uniform-thickness prior
            x, y = thickness_jitter(x, y, prob=cfg.get("thick_prob", 0.5))
        if cfg.get("elastic_aug", False):             # smooth elastic deformation
            x, y = elastic_aug(x, y, prob=cfg.get("elastic_prob", 0.5))
        if cfg.get("crop_zoom", False):               # patch-detail: zoom a sub-window -> thin layers
            x, y = crop_zoom_aug(x, y, prob=cfg.get("cz_prob", 0.4))
        if cfg.get("pathology_warp", False):          # focal drusen/fluid layer deformation
            x, y = pathology_warp(x, y, prob=cfg.get("path_prob", 0.3))
        if cfg.get("anatomy_aug", False):
            x, y = anatomy_warp(
                x,
                y,
                prob=cfg.get("anatomy_prob", 0.35),
                modes=cfg.get("anatomy_modes", "edema,drusen,onh"),
                severity=cfg.get("anatomy_severity", 0.8),
            )
        if cfg.get("fov_exit_aug", False):
            x, y = axial_fov_exit(
                x,
                y,
                prob=cfg.get("fov_exit_prob", 0.12),
                min_shift=cfg.get("fov_exit_min_shift", 0.45),
                max_shift=cfg.get("fov_exit_max_shift", 1.05),
                top_prob=cfg.get("fov_exit_top_prob", 0.8),
                max_tilt=cfg.get("fov_exit_max_tilt", 0.10),
            )
        if cfg.get("edge_blackout", True):
            x, y = edge_blackout(x, y, prob=cfg.get("edge_blackout_prob", 0.5))
        # multi-scale training: random input size per step -> the model learns to segment
        # at 512/640/768 natively, so high-res inference helps (not hurts) the steep layers.
        ms = cfg.get("ms_train_sizes")
        cur = int(x.shape[-1])
        if ms:
            cur = int(ms[int(torch.randint(len(ms), (1,)))])
            if cur != x.shape[-1]:
                x = nnf.interpolate(x, (cur, cur), mode="bilinear", align_corners=False)
                y = nnf.interpolate(y.float(), (cur, cur), mode="nearest").long()
        x = domain_randomize(x, strength=cfg.get("dr_strength", 1.0))
        if cfg.get("device_intensity", False):        # depth gain + inter-layer contrast (SS-OCT/device)
            x = device_intensity_aug(x, prob=cfg.get("dev_int_prob", 0.7))
        if cfg.get("lesion_aug", False):              # pathology robustness (image-only, label intact)
            x = lesion_aug(x, prob=cfg.get("lesion_prob", 0.3))
        if semi and cfg.get("use_fda", True) and torch.rand(()) < cfg.get("fda_prob", 0.5):
            tgt = next(ul_iter)["image"].to(dev)
            if tgt.shape[-1] != cur:
                tgt = nnf.interpolate(tgt, (cur, cur), mode="bilinear", align_corners=False)
            x = fda_swap(x, tgt[: x.shape[0]], beta=cfg.get("fda_beta", 0.01))
        if semi and cfg.get("noise_transfer", False):     # SVDNA-style device-noise transfer
            tn = next(ul_iter)["image"].to(dev)
            if tn.shape[-1] != cur:
                tn = nnf.interpolate(tn, (cur, cur), mode="bilinear", align_corners=False)
            x = noise_transfer(x, tn, prob=cfg.get("noise_prob", 0.4))

        with torch.amp.autocast("cuda", enabled=cfg["amp"]):
            out = model(x)
            seg_logits = _seg(out)
            sup_loss, parts = loss_fn(seg_logits, y)
            balanced_ce_weight = cfg.get("w_balanced_ce", 0.0)
            if balanced_ce_weight:
                balanced_ce = _balanced_cross_entropy(
                    seg_logits,
                    y,
                    power=cfg.get("balanced_ce_power", 0.5),
                    min_weight=cfg.get("balanced_ce_min", 0.35),
                    max_weight=cfg.get("balanced_ce_max", 3.0),
                )
                sup_loss = (
                    sup_loss
                    + float(balanced_ce_weight) * balanced_ce
                )
                parts["balanced_ce"] = balanced_ce.detach()
            if mh:
                hl, hp = head_losses(out, y, nb, w_sdm=cfg.get("w_sdm", 0.5),
                                     w_surface=cfg.get("w_surface", 0.5),
                                     w_flatten=cfg.get("w_flatten", 0.5),
                                     n_classes=cfg["num_classes"])
                sup_loss = sup_loss + hl; parts.update(hp)
            cons = seg_logits.new_zeros(())
            if semi:
                ub = next(ul_iter)["image"].to(dev)
                if ub.shape[-1] != cur:
                    ub = nnf.interpolate(ub, (cur, cur), mode="bilinear", align_corners=False)
                xs, xw = strong_aug(ub), weak_aug(ub)
                with torch.no_grad():
                    t_out = teacher.predict(xw)
                    t_logits = _seg(t_out)
                    if cfg.get("teacher_flip_consensus", False):
                        flipped = teacher.predict(torch.flip(xw, dims=(-1,)))
                        flipped_logits = torch.flip(
                            _seg(flipped), dims=(-1,)
                        )
                        teacher_probability = 0.5 * (
                            torch.softmax(t_logits, dim=1)
                            + torch.softmax(flipped_logits, dim=1)
                        )
                        t_logits = torch.log(
                            teacher_probability.clamp_min(1e-7)
                        )
                s_out = model(xs)
                s_logits = _seg(s_out)
                if (
                    cfg.get("consistency_mode") == "surface_consensus"
                    and isinstance(t_out, dict)
                    and "surf_rows" in t_out
                ):
                    cons = surface_consensus_loss(
                        s_logits,
                        t_logits,
                        t_out["surf_rows"],
                        s_out.get("surf_rows")
                        if isinstance(s_out, dict) else None,
                        conf_thresh=cfg.get("conf_thresh", 0.75),
                        class_balance=cfg.get(
                            "cons_class_balance", True
                        ),
                        surface_weight=cfg.get(
                            "target_surface_weight", 0.25
                        ),
                    )
                else:
                    cons = consistency_loss(
                        s_logits,
                        t_logits,
                        cfg.get("conf_thresh", 0.7),
                        mode=cfg.get("consistency_mode", "mse"),
                        class_balance=cfg.get("cons_class_balance", False),
                        boundary_weight=cfg.get(
                            "pseudo_boundary_weight", 0.0
                        ),
                        boundary_radius=cfg.get(
                            "pseudo_boundary_radius", 2
                        ),
                        boundary_axes=cfg.get(
                            "pseudo_boundary_axes", "both"
                        ),
                    )
            ramp = min(1.0, step / max(1, cfg.get("cons_rampup", 500)))
            total = sup_loss + cfg.get("w_cons", 1.0) * ramp * cons

        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.step(opt); scaler.update(); sched.step()
        if teacher:
            teacher.update(model)

        if step % cfg.get("log_every", 50) == 0:
            sps = step / (time.time() - t0)
            msg = (f"step {step}/{cfg['steps']} total={total.item():.4f} "
                   f"dicece={parts.get('dicece', 0):.3f} bnd={parts.get('boundary', 0):.3f} "
                   f"ord={parts.get('order', 0):.3f} sdm={parts.get('sdm', 0):.3f} "
                   f"surf={parts.get('surface', 0):.3f} flat={parts.get('flatten', 0):.3f} "
                   f"bce={parts.get('balanced_ce', 0):.3f} "
                   f"cons={float(cons.detach()):.4f} "
                   f"lr={sched.get_last_lr()[0]:.2e} {sps:.1f}it/s")
            print(msg, flush=True)
        if step % cfg.get("val_every", 500) == 0 or step == cfg["steps"]:
            try:
                s = evaluate(model, val_loader, dev, cfg.get("val_batches"))
                ts = (
                    evaluate(
                        teacher.teacher,
                        val_loader,
                        dev,
                        cfg.get("val_batches"),
                    )
                    if teacher else None
                )
            except Exception as e:                       # e.g. scipy absent on default image
                s = None
                ts = None
                if step == cfg["steps"]:
                    print(f"[val] skipped ({type(e).__name__}: {e})", flush=True)
            if s is not None:
                tag = ""
                if s > best:
                    best = s; torch.save(model.state_dict(), ckpt_dir / "best.pth"); tag = " *best"
                print(f"[val] step {step} image_score={s:.4f} (best={best:.4f}){tag}", flush=True)
            if ts is not None:
                tag = ""
                if ts > best_teacher:
                    best_teacher = ts
                    torch.save(
                        teacher.teacher.state_dict(),
                        ckpt_dir / "best_teacher.pth",
                    )
                    tag = " *best"
                print(
                    f"[val-teacher] step {step} image_score={ts:.4f} "
                    f"(best={best_teacher:.4f}){tag}",
                    flush=True,
                )
    torch.save(model.state_dict(), ckpt_dir / "last.pth")
    if teacher:
        torch.save(
            teacher.teacher.state_dict(), ckpt_dir / "last_teacher.pth"
        )
    student_path = ckpt_dir / "best.pth"
    teacher_path = ckpt_dir / "best_teacher.pth"
    if best_teacher > best and teacher_path.exists():
        selected_path = teacher_path
        selected_score = best_teacher
        selected_kind = "teacher"
    else:
        selected_path = student_path if student_path.exists() else ckpt_dir / "last.pth"
        selected_score = best
        selected_kind = "student"
    shutil.copyfile(selected_path, ckpt_dir / "selected.pth")
    print(
        f"[done] selected={selected_kind} image_score={selected_score:.4f} "
        f"student={best:.4f} teacher={best_teacher:.4f} ckpt={ckpt_dir}",
        flush=True,
    )
    if cfg.get("return_details", False):
        return {
            "selected_score": selected_score,
            "student_score": best,
            "teacher_score": best_teacher,
            "selected_kind": selected_kind,
        }
    return selected_score
