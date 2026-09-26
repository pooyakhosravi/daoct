"""Pre-resize a B-scan tree to the training size ONCE, so the dataloader reads cheap
fixed-size PNGs (no per-access resize, no RAM cache -> fast + leak-free).

  python scripts/preresize.py --src work/extracted --dst work/extracted_512 --size 512 --mode square
Then point unlabeled_roots at the --dst directory and keep unlabeled_cache=0.
"""
import argparse, glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image


def resize_one(args):
    src, dst, size, mode = args
    try:
        is_mask = "mask" in Path(src).name.lower()
        rs = Image.NEAREST if is_mask else Image.BILINEAR   # masks must not interpolate labels
        arr = np.asarray(Image.open(src).convert("L"))
        if mode == "square":
            im = Image.fromarray(arr).resize((size, size), rs)
        else:  # fixed_h: fixed height, aspect width, pad/crop to size
            w = max(1, round(arr.shape[1] * size / arr.shape[0]))
            im = Image.fromarray(arr).resize((w, size), rs)
            a = np.asarray(im)
            if w != size:
                if w > size:
                    s = (w - size) // 2; a = a[:, s:s + size]
                else:
                    p = size - w; l = p // 2
                    a = np.pad(a, ((0, 0), (l, p - l)))
            im = Image.fromarray(a)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        im.save(dst)
        return True
    except Exception as e:
        print(f"[skip] {src}: {e}")
        return False


def preresize_tree(src, dst, size=512, mode="square", workers=12):
    """Resize every .png under src (images bilinear, masks nearest) into dst, mirroring
    the directory structure. Importable for the submission entrypoint."""
    src, dst = Path(src), Path(dst)
    files = glob.glob(str(src / "**" / "*.png"), recursive=True)
    jobs = [(f, str(dst / Path(f).relative_to(src)), size, mode) for f in files]
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(resize_one, jobs):
            ok += r
    print(f"[preresize] {ok}/{len(files)} -> {dst} ({size},{mode})", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--mode", default="square")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    preresize_tree(a.src, a.dst, a.size, a.mode, a.workers)


if __name__ == "__main__":
    main()
