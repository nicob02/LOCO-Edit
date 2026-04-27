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

    plt.rcParams.update({"font.size": 12})
    fig = plt.figure(figsize=(14.5, 10.5))
    gs = fig.add_gridspec(
        4, 5,
        height_ratios=[1.30, 1.30, 0.18, 0.95],
        hspace=0.18, wspace=0.05,
        left=0.04, right=0.99, top=0.92, bottom=0.04,
    )

    ax_clean = fig.add_subplot(gs[0, :])
    ax_clean.imshow(cs); ax_clean.axis("off")
    ax_clean.set_title(
        r"(a) clean LOCO edit strip   $-2\,\mathrm{d}\lambda \ldots +2\,\mathrm{d}\lambda$",
        fontsize=15, fontweight="bold", pad=8,
    )

    ax_attacked = fig.add_subplot(gs[1, :])
    ax_attacked.imshow(asr); ax_attacked.axis("off")
    ax_attacked.set_title(
        r"(b) attacked LOCO edit strip   ($\varepsilon_{\mathrm{img}}=0.031$, 40 PGD steps)",
        fontsize=15, fontweight="bold", pad=8,
    )

    # Centered bold row title for the zoom panels, in its own gridspec band.
    ax_rowtitle = fig.add_subplot(gs[2, :])
    ax_rowtitle.axis("off")
    ax_rowtitle.text(
        0.5, 0.5,
        r"(c) what the defense does at the pixel level",
        ha="center", va="center", fontsize=15, fontweight="bold",
        transform=ax_rowtitle.transAxes,
    )

    panels = [
        (r"clean $x_0$",                                          x_clean),
        (r"adversarial $x_0$",                                    x_adv),
        (rf"bits:{args.bits}  ($x_{{0,\mathrm{{adv}}}}$)",        bits),
        (rf"jpeg:{args.jpeg_q}  ($x_{{0,\mathrm{{adv}}}}$)",      jpg),
        (rf"blur:{args.blur_sigma:g}  ($x_{{0,\mathrm{{adv}}}}$)", blr),
    ]
    for k, (lbl, im) in enumerate(panels):
        ax = fig.add_subplot(gs[3, k])
        ax.imshow(im); ax.axis("off")
        ax.set_title(lbl, fontsize=12.5, fontweight="bold")

    fig.suptitle(args.title, fontsize=14, y=0.985)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"[def-vis] -> {args.out}")


if __name__ == "__main__":
    main()
