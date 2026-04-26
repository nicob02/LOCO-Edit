"""Pixel-level "what does the defense actually do?" visual.

Stacks a clean LOCO edit strip, the attacked LOCO edit strip, and a row of
zoom panels (clean x_0 / adversarial x_0 / bits:4(adv) / jpeg:75(adv) /
blur:1.5(adv)) so the reader can see by eye what each purification removes.

This is a CPU-only script: it consumes already-saved PNG/PT artefacts and
applies the purifications via PIL/numpy. For a *true* defended LOCO edit
strip (post-purify pipeline rerun) use src/tools/phase3_defense_strip.py
on a GPU node.

Example
-------
RUN=src/runs/CelebA_HQ_HF-CelebA_HQ_mask-phase2_attackB_ms_s4729/results/sample_idx4729/attackB-linf-eps_img0.031-40steps
CLEAN=$(find src/runs -path '*phase1*' \
    -name '*Edit-randomFalse_xt-noise-False_l_eye-edit_0.6T_null_proj_True_*' \
    | head -n1)
python tools/fig_defense_visual_strip.py \
    --clean_strip "$CLEAN" \
    --attacked_strip "$RUN/attackB-4729-l_eye-eps0.031-linf-edit_strip-pc_0.png" \
    --x0_clean "$RUN/x0_clean.png" \
    --x0_adv   "$RUN/x0_adv.png" \
    --out tools/report_figures/fig_defense_visual_strip_eye_eps0.031_4729.png
"""
from __future__ import annotations

import argparse
import io
import os

import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt


def _load(p: str) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def bits_reduce(arr: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** int(bits) - 1
    x = arr.astype(np.float32) / 255.0
    return (np.round(x * levels) / levels * 255).clip(0, 255).astype(np.uint8)


def jpeg_reencode(arr: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def gauss_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=float(sigma))))


def main() -> None:
    P = argparse.ArgumentParser(description=__doc__)
    P.add_argument("--clean_strip", required=True)
    P.add_argument("--attacked_strip", required=True)
    P.add_argument("--x0_clean", required=True)
    P.add_argument("--x0_adv", required=True)
    P.add_argument("--out", required=True)
    P.add_argument("--bits", type=int, default=4)
    P.add_argument("--jpeg_q", type=int, default=75)
    P.add_argument("--blur_sigma", type=float, default=1.5)
    P.add_argument("--title", default="What the defense does at the pixel level (eps_img=0.031)")
    args = P.parse_args()

    cs = _load(args.clean_strip)
    asr = _load(args.attacked_strip)
    x_clean = _load(args.x0_clean)
    x_adv = _load(args.x0_adv)
    bits = bits_reduce(x_adv, args.bits)
    jpg = jpeg_reencode(x_adv, args.jpeg_q)
    blr = gauss_blur(x_adv, args.blur_sigma)
    delta = (x_adv.astype(np.float32) - x_clean.astype(np.float32))
    delta = (delta - delta.min()) / (delta.max() - delta.min() + 1e-8)
    delta = (delta * 255).astype(np.uint8)

    plt.rcParams.update({"font.size": 11})
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.2, 1.2, 0.95],
                          hspace=0.22, wspace=0.04)

    ax_clean = fig.add_subplot(gs[0, :])
    ax_clean.imshow(cs); ax_clean.axis("off")
    ax_clean.set_title("clean LOCO edit strip   (-2dλ ... +2dλ)", fontsize=12)

    ax_attacked = fig.add_subplot(gs[1, :])
    ax_attacked.imshow(asr); ax_attacked.axis("off")
    ax_attacked.set_title("attacked LOCO edit strip   (eps_img=0.031, 40 PGD steps)", fontsize=12)

    panels = [
        ("clean x_0",          x_clean),
        ("adversarial x_0",    x_adv),
        (f"bits:{args.bits}(adv)",   bits),
        (f"jpeg:{args.jpeg_q}(adv)", jpg),
        (f"blur:{args.blur_sigma:g}(adv)", blr),
    ]
    for k, (lbl, im) in enumerate(panels):
        ax = fig.add_subplot(gs[2, k])
        ax.imshow(im); ax.axis("off"); ax.set_title(lbl, fontsize=11)

    fig.suptitle(args.title, fontsize=13, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"[def-vis] -> {args.out}")


if __name__ == "__main__":
    main()
