"""Phase 1 figures for LOCO-Edit reproduction.

Given a results root and the celebA-HQ dataset path, this script:

  1. Locates the saved 5-frame edit strips produced by
     ``run_edit_null_space_projection`` (CelebA_HQ_HF + CelebA_HQ_mask).
     Strip frames are: [-2dλ, -dλ, original, +dλ, +2dλ].
  2. Builds the "edit row" figure (with vs without null-space projection).
  3. Computes locality and a linearity proxy across a λ-sweep and saves
     the corresponding plots.

Outputs (under --out_dir):
    edit_row.png          stacked strips, with vs without null-space
    locality_vs_lambda.png
    linearity_vs_lambda.png
    metrics.csv           raw numbers (lambda, inside, outside, locality)

Run on a CPU node (no GPU needed). Example:

    python tools/phase1_plot.py \
        --runs_root src/runs \
        --celeba_root /project/6001170/nicob0/data/CelebAMask-HQ \
        --sample_idx 4729 \
        --choose_sem l_eye \
        --with_null_note phase1_with_null \
        --without_null_note phase1_without_null \
        --sweep_note phase1_lambda_sweep \
        --out_dir figures/phase1
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


STRIP_FRAMES = 5
ORIGINAL_FRAME = 2
POS_MAX_FRAME = STRIP_FRAMES - 1
NEG_MAX_FRAME = 0


STRIP_RE = re.compile(
    r"^(?P<idx>\d+)-Edit-random(?:True|False)_xt-noise-(?:True|False)_"
    r"(?P<sem>[A-Za-z_]+)-edit_(?P<edit_t>[0-9.]+)T_null_proj_(?P<null>True|False)_"
    r"rank(?P<rank>\d+)_scale_(?P<scale>[0-9.]+)-pc_\d+\.png$"
)


def find_strips(results_dir: Path) -> list[dict]:
    """Return metadata dicts for every 5-frame edit strip we recognize."""
    strips: list[dict] = []
    if not results_dir.is_dir():
        return strips
    for png in sorted(results_dir.glob("*.png")):
        m = STRIP_RE.match(png.name)
        if not m:
            continue
        d = m.groupdict()
        strips.append(
            {
                "path": png,
                "sample_idx": int(d["idx"]),
                "sem": d["sem"],
                "edit_t": float(d["edit_t"]),
                "null": d["null"] == "True",
                "rank_null": int(d["rank"]),
                "scale": float(d["scale"]),
            }
        )
    return strips


def split_strip(img: Image.Image, n_frames: int = STRIP_FRAMES) -> list[Image.Image]:
    """Split a horizontal grid of n_frames equally-sized images."""
    w, h = img.size
    if w % n_frames != 0:
        # tvu.save_image with nrow=N uses 2px padding; tolerate small remainder
        frame_w = w // n_frames
    else:
        frame_w = w // n_frames
    return [
        img.crop((i * frame_w, 0, (i + 1) * frame_w, h))
        for i in range(n_frames)
    ]


def load_mask(celeba_root: Path, sample_idx: int, choose_sem: str, res: int) -> np.ndarray:
    """Load the CelebA-HQ semantic mask as a [res,res] bool array."""
    mask_root = celeba_root / "CelebAMask-HQ-mask-anno"
    bucket = sample_idx // 2000
    fname = f"{sample_idx:05d}_{choose_sem}.png"
    candidate = mask_root / str(bucket) / fname
    if not candidate.is_file():
        # Fall back: brute-force search across all bucket folders.
        matches = list(mask_root.glob(f"*/{fname}"))
        if not matches:
            raise FileNotFoundError(
                f"Mask not found for idx={sample_idx} sem={choose_sem} under {mask_root}"
            )
        candidate = matches[0]
    mask = Image.open(candidate).convert("L").resize((res, res), Image.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 0


def per_lambda_metrics(
    strip_path: Path,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Return (inside_change, outside_change) in [0,1] units (mean abs diff)."""
    img = Image.open(strip_path).convert("RGB")
    frames = split_strip(img)
    orig = np.asarray(frames[ORIGINAL_FRAME], dtype=np.float32) / 255.0
    edited = np.asarray(frames[POS_MAX_FRAME], dtype=np.float32) / 255.0

    # Resize mask to frame resolution if needed.
    fh, fw, _ = orig.shape
    if mask.shape != (fh, fw):
        m_img = Image.fromarray(mask.astype(np.uint8) * 255).resize((fw, fh), Image.NEAREST)
        m = np.asarray(m_img, dtype=np.uint8) > 0
    else:
        m = mask

    diff = np.mean(np.abs(edited - orig), axis=2)  # [H, W]
    inside = float(diff[m].mean()) if m.any() else 0.0
    outside = float(diff[~m].mean()) if (~m).any() else 0.0
    return inside, outside


def build_edit_row(
    strip_with: Optional[Path],
    strip_without: Optional[Path],
    out_path: Path,
) -> None:
    """Stack the with-null and without-null 5-frame strips vertically."""
    rows: list[Image.Image] = []
    labels: list[str] = []
    if strip_with is not None and strip_with.is_file():
        rows.append(Image.open(strip_with).convert("RGB"))
        labels.append("with null-space projection")
    if strip_without is not None and strip_without.is_file():
        rows.append(Image.open(strip_without).convert("RGB"))
        labels.append("without null-space projection")
    if not rows:
        print("[edit_row] no strips to assemble; skipping", file=sys.stderr)
        return

    target_w = max(r.width for r in rows)
    rows = [r.resize((target_w, int(r.height * target_w / r.width))) for r in rows]

    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 2.5 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax, im, lab in zip(axes, rows, labels):
        ax.imshow(im)
        ax.set_title(lab, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[edit_row] wrote {out_path}")


def build_lambda_plots(
    sweep_dir: Path,
    mask: np.ndarray,
    sample_idx: int,
    choose_sem: str,
    out_dir: Path,
) -> None:
    strips = [
        s for s in find_strips(sweep_dir)
        if s["sample_idx"] == sample_idx and s["sem"] == choose_sem
    ]
    if not strips:
        print(f"[sweep] no strips found in {sweep_dir}", file=sys.stderr)
        return
    strips.sort(key=lambda s: s["scale"])

    rows = []
    for s in strips:
        inside, outside = per_lambda_metrics(s["path"], mask)
        locality = outside / inside if inside > 0 else float("nan")
        rows.append(
            {
                "lambda": s["scale"],
                "null": s["null"],
                "inside": inside,
                "outside": outside,
                "locality": locality,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[sweep] wrote {csv_path}")

    lambdas = np.array([r["lambda"] for r in rows], dtype=np.float32)
    inside = np.array([r["inside"] for r in rows], dtype=np.float32)
    outside = np.array([r["outside"] for r in rows], dtype=np.float32)
    locality = np.array([r["locality"] for r in rows], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(lambdas, locality, marker="o")
    ax.set_xlabel(r"$\lambda$ (x_space_guidance_scale)")
    ax.set_ylabel("locality = outside / inside change")
    ax.set_title(f"Locality vs $\\lambda$  (idx={sample_idx}, sem={choose_sem})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "locality_vs_lambda.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"[sweep] wrote {p}")

    if len(lambdas) >= 2 and inside.std() > 0:
        corr = float(np.corrcoef(lambdas, inside)[0, 1])
    else:
        corr = float("nan")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(lambdas, inside, marker="o", label="inside (ROI)")
    ax.plot(lambdas, outside, marker="s", label="outside (background)")
    ax.set_xlabel(r"$\lambda$ (x_space_guidance_scale)")
    ax.set_ylabel("mean |edited - original|")
    ax.set_title(
        f"Linearity proxy  (Pearson $\\rho(\\lambda$, inside)$={corr:.2f}$)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "linearity_vs_lambda.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"[sweep] wrote {p}")


def first_strip(results_dir: Path, sample_idx: int, choose_sem: str,
                want_null: bool) -> Optional[Path]:
    cands = [
        s for s in find_strips(results_dir)
        if s["sample_idx"] == sample_idx and s["sem"] == choose_sem
        and s["null"] == want_null
    ]
    if not cands:
        return None
    cands.sort(key=lambda s: s["scale"])
    return cands[0]["path"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, required=True,
                   help="Path to src/runs (parent of <model>-<dataset>-<note> dirs).")
    p.add_argument("--celeba_root", type=Path, required=True,
                   help="Path to CelebAMask-HQ (parent of CelebA-HQ-img and CelebAMask-HQ-mask-anno).")
    p.add_argument("--sample_idx", type=int, default=4729)
    p.add_argument("--choose_sem", type=str, default="l_eye")
    p.add_argument("--mask_res", type=int, default=256)
    p.add_argument("--with_null_note", type=str, default="phase1_with_null")
    p.add_argument("--without_null_note", type=str, default="phase1_without_null")
    p.add_argument("--sweep_note", type=str, default="phase1_lambda_sweep")
    p.add_argument("--model_name", type=str, default="CelebA_HQ_HF")
    p.add_argument("--dataset_name", type=str, default="CelebA_HQ_mask")
    p.add_argument("--out_dir", type=Path, default=Path("figures/phase1"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = lambda note: (
        args.runs_root
        / f"{args.model_name}-{args.dataset_name}-{note}"
        / "results"
        / f"sample_idx{args.sample_idx}"
    )

    mask = load_mask(args.celeba_root, args.sample_idx, args.choose_sem, args.mask_res)

    strip_with = first_strip(base(args.with_null_note), args.sample_idx,
                             args.choose_sem, want_null=True)
    strip_without = first_strip(base(args.without_null_note), args.sample_idx,
                                args.choose_sem, want_null=False)
    build_edit_row(strip_with, strip_without, args.out_dir / "edit_row.png")

    build_lambda_plots(
        sweep_dir=base(args.sweep_note),
        mask=mask,
        sample_idx=args.sample_idx,
        choose_sem=args.choose_sem,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
