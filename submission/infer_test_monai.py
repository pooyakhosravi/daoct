"""Submission inference (template-named; called as main(Namespace(input_dir, output_dir,
model_path))). Robust to any *.png naming; rebuilds the exact architecture from
model_meta.json; DP-refine + empty-column masking; preds resized to native (nearest).
"""
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from daoct.models.load import load_model      # noqa: E402
from daoct.engine.infer import run            # noqa: E402


def main(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.model_path, dev, fallback="segresnet", num_classes=10)
    anchor_path = Path(args.model_path).with_name("unet_maestro2_anchor.pth")
    anchor_model = None
    if anchor_path.is_file():
        anchor_model, _ = load_model(
            anchor_path, dev, fallback="segresnet", num_classes=10
        )
    size = meta.get("size", 512)
    mode = meta.get("resize_mode", "square")
    norm = bool(meta.get("input_norm", False))    # match train-time per-image normalization
    # Left/right anatomy is symmetric. A mild lateral-expansion view adds aspect
    # robustness while staying close to the square geometry used for training.
    run(model, args.input_dir, args.output_dir, dev, size, mode,
        refine=True, tta="flip_lat115", mask_empty=True, multiscale=None,
        norm=norm, anchor_model=anchor_model, presence_guard_label=2)


if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_path", required=True)
    main(p.parse_args())
