"""Evaluate locality & linearity under clean / attacked / defended conditions.

The original LOCO Edit paper measures direction quality with three properties:
    misalignment   = 1 - |<v_clean, v_adv>|             (already plotted)
    locality       = outside-mask change / inside-mask change
    linearity      = Pearson(|lambda|, inside-mask change)

This script extracts (locality, linearity) for an arbitrary list of saved
5-frame edit strips, so we can answer "does the attack/defense change
locality and linearity, not just misalignment?" The strips can come from:

  * Phase 1                 (clean LOCO; reproduction baseline)
  * Phase 2 Attack B        (attacked strip rendered by _render_attacked_edit)
  * Phase 3 defense pipeline (defended strip rendered by phase3_defense_strip.py)

All strips must have 5 frames at offsets {-2dλ, -dλ, 0, +dλ, +2dλ} so the
original frame is index 2.

Outputs:
    --out_csv   per-condition (inside, outside, locality, linearity) table
    --out_bar   side-by-side bar chart (locality, linearity) per condition
    --out_inside_curve  line plot inside-change vs |offset| per condition

Example
-------
python tools/eval_locality_linearity_under_attack.py \
    --celeba_root /project/.../CelebAMask-HQ \
    --sample_idx 4729 --choose_sem l_eye \
    --strip "clean=src/runs/.../phase1_with_null/.../*l_eye*null_proj_True*.png" \
    --strip "attacked B (eps=0.031)=src/runs/.../attackB-...edit_strip-pc_0.png" \
    --strip "defended bits:4=src/runs/.../defenseD2-bits-4-...-edit_strip-pc_0.png" \
    --strip "defended jpeg:75=src/runs/.../defenseD2-jpeg-75-...-edit_strip-pc_0.png" \
    --strip "defended blur:1.5=src/runs/.../defenseD2-blur-1.5-...-edit_strip-pc_0.png" \
    --out_csv tools/report_figures/locality_linearity_under_attack_4729_eye.csv \
    --out_bar tools/report_figures/fig_locality_linearity_under_attack_4729_eye.png \
    --out_inside_curve tools/report_figures/fig_inside_curve_under_attack_4729_eye.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

STRIP_FRAMES = 5
ORIGINAL_FRAME = 2
FRAME_OFFSETS = np.array([-2, -1, 0, 1, 2], dtype=np.float32)


def split_strip(img: Image.Image, n_frames: int = STRIP_FRAMES) -> list[Image.Image]:
    w, h = img.size
    frame_w = w // n_frames
    return [img.crop((i * frame_w, 0, (i + 1) * frame_w, h)) for i in range(n_frames)]


def load_mask(celeba_root: Path, sample_idx: int, choose_sem: str,
              res: int) -> np.ndarray:
    mask_root = celeba_root / "CelebAMask-HQ-mask-anno"
    bucket = sample_idx // 2000
    fname = f"{sample_idx:05d}_{choose_sem}.png"
    candidate = mask_root / str(bucket) / fname
    if not candidate.is_file():
        matches = list(mask_root.glob(f"*/{fname}"))
        if not matches:
            raise FileNotFoundError(
                f"Mask not found for idx={sample_idx} sem={choose_sem} under {mask_root}"
            )
        candidate = matches[0]
    mask = Image.open(candidate).convert("L").resize((res, res), Image.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 0


def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape == (h, w):
        return mask
    m_img = Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
    return np.asarray(m_img, dtype=np.uint8) > 0


def per_frame_changes(strip_path: Path, mask: np.ndarray) -> np.ndarray:
    """Return shape [STRIP_FRAMES, 2] = (inside_change, outside_change) per frame."""
    img = Image.open(strip_path).convert("RGB")
    frames = split_strip(img)
    arrs = [np.asarray(f, dtype=np.float32) / 255.0 for f in frames]
    orig = arrs[ORIGINAL_FRAME]
    h, w, _ = orig.shape
    m = _resize_mask(mask, h, w)
    out = np.zeros((STRIP_FRAMES, 2), dtype=np.float32)
    for i, a in enumerate(arrs):
        diff = np.abs(a - orig).mean(axis=2)
        out[i, 0] = float(diff[m].mean()) if m.any() else 0.0
        out[i, 1] = float(diff[~m].mean()) if (~m).any() else 0.0
    return out


def _resolve_strip(spec: str) -> Path:
    """Each --strip is 'label=glob_or_path'. Return the first match."""
    matches = sorted(glob.glob(spec))
    if not matches:
        raise FileNotFoundError(f"no file matches strip spec: {spec}")
    return Path(matches[0])


def main() -> None:
    P = argparse.ArgumentParser(description=__doc__)
    P.add_argument("--celeba_root", type=Path, required=True)
    P.add_argument("--sample_idx", type=int, required=True)
    P.add_argument("--choose_sem", type=str, required=True)
    P.add_argument("--mask_res", type=int, default=256)
    P.add_argument(
        "--strip",
        action="append",
        required=True,
        help=("Repeatable. Format: 'label=path-or-glob'. "
              "First matching file is used."),
    )
    P.add_argument("--out_csv", type=Path, required=True)
    P.add_argument("--out_bar", type=Path, required=True)
    P.add_argument("--out_inside_curve", type=Path, default=None)
    args = P.parse_args()

    if args.out_csv.parent:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.out_bar.parent:
        args.out_bar.parent.mkdir(parents=True, exist_ok=True)
    if args.out_inside_curve and args.out_inside_curve.parent:
        args.out_inside_curve.parent.mkdir(parents=True, exist_ok=True)

    mask = load_mask(args.celeba_root, args.sample_idx, args.choose_sem, args.mask_res)

    rows: list[dict] = []
    inside_curves: list[tuple[str, np.ndarray]] = []

    for spec in args.strip:
        if "=" not in spec:
            raise SystemExit(f"bad --strip; need 'label=path' got {spec!r}")
        label, target = spec.split("=", 1)
        label = label.strip()
        path = _resolve_strip(target.strip())
        per_frame = per_frame_changes(path, mask)
        inside, outside = per_frame[:, 0], per_frame[:, 1]

        # Locality at the OUTER frames (where editing is strongest), averaged
        # across the +max and -max sides for stability.
        lo_pos = float(outside[POS := STRIP_FRAMES - 1] / max(inside[POS], 1e-8))
        lo_neg = float(outside[NEG := 0]                 / max(inside[NEG], 1e-8))
        locality = 0.5 * (lo_pos + lo_neg)

        # Linearity: Pearson( |offset|, inside_change ) over the 5 frames.
        absoff = np.abs(FRAME_OFFSETS)
        if inside.std() > 0:
            linearity = float(np.corrcoef(absoff, inside)[0, 1])
        else:
            linearity = float("nan")

        # Linearity (signed): if you also want to know whether positive and
        # negative directions move equally, fit a linear model |inside| ~ |off|
        # and look at residual ratio. We log it but don't plot it.
        slope = float(np.polyfit(absoff, inside, 1)[0])

        rows.append({
            "label": label,
            "path": str(path),
            "inside_pos_max": float(inside[POS]),
            "outside_pos_max": float(outside[POS]),
            "inside_neg_max": float(inside[NEG]),
            "outside_neg_max": float(outside[NEG]),
            "locality": locality,
            "linearity_pearson": linearity,
            "linearity_slope": slope,
        })
        inside_curves.append((label, inside))
        print(f"[eval] {label:<32}  locality={locality:.3f}  "
              f"linearity={linearity:+.3f}  inside_max={inside[POS]:.3f}")

    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[eval] csv -> {args.out_csv}")

    labels = [r["label"] for r in rows]
    locality = np.array([r["locality"] for r in rows], dtype=np.float32)
    linearity = np.array([r["linearity_pearson"] for r in rows], dtype=np.float32)

    plt.rcParams.update({"font.size": 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    bar_x = np.arange(len(labels))
    ax1.bar(bar_x, locality, color="#4878d0", edgecolor="black")
    ax1.set_xticks(bar_x); ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("locality   (outside / inside change)")
    ax1.set_title("Locality   (smaller = more local; want < 0.25 like in Phase 1)")
    ax1.axhline(0.25, ls=":", color="grey")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(locality):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    ax2.bar(bar_x, linearity, color="#d62728", edgecolor="black")
    ax2.set_xticks(bar_x); ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel(r"Pearson($|\lambda|$, inside-change)")
    ax2.set_title("Linearity   (closer to 1 = more linear; Phase 1 ≈ 0.99)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(1.0, ls=":", color="grey")
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(linearity):
        ax2.text(i, v + 0.02, f"{v:+.2f}", ha="center", fontsize=10)

    fig.suptitle(
        f"Locality + linearity under attack/defense  "
        f"(idx={args.sample_idx}, sem={args.choose_sem})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(args.out_bar, dpi=200, bbox_inches="tight")
    print(f"[eval] bar -> {args.out_bar}")

    if args.out_inside_curve is not None:
        fig2, ax = plt.subplots(figsize=(7, 4.5))
        absoff = np.abs(FRAME_OFFSETS)
        for lbl, ins in inside_curves:
            ax.plot(absoff, ins, marker="o", lw=2, label=lbl)
        ax.set_xlabel(r"$|\lambda|$  (frame offset)")
        ax.set_ylabel("inside-mask mean change")
        ax.set_title("Inside-change vs |λ| under attack/defense")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=10)
        fig2.tight_layout()
        fig2.savefig(args.out_inside_curve, dpi=200, bbox_inches="tight")
        print(f"[eval] curve -> {args.out_inside_curve}")


if __name__ == "__main__":
    main()
