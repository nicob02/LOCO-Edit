"""Phase 2 visual comparison figures.

Assembles the single most-convincing figure for the report: for each epsilon
in an attack sweep, a 3-row panel

    [  x0_clean  |  x0_adv  |  delta_vis       ]   <- imperceptibility (B only)
    [  ---- clean LOCO edit strip at matched lambda  ----                    ]
    [  ---- attacked LOCO edit strip at matched lambda ----                  ]

Reads raw PNGs produced by the attack runs and by Phase 1. No GPU needed;
runs on CPU.

Inputs:
  --clean_edit_strip   path to Phase 1's saved edit strip (produced by
                       run_edit_null_space_projection). e.g.
                       src/runs/<phase1_exp>/results/sample_idx4729/
                         .../Semantic_Edit_xt_*-null_space_projection_True_*.png
  --attack_sweep_dir   path to src/runs/<phase2_exp>/results/sample_idx4729
                       (contains attackA-*/ or attackB-*/ subfolders)
  --attack_type        A or B
  --out                output PNG path
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _natural_eps_sort(path):
    """Sort attack-run folder names by numeric eps (epsX or eps_imgX)."""
    name = os.path.basename(path)
    for tok in name.split("-"):
        if tok.startswith("eps_img"):
            try: return float(tok[len("eps_img"):])
            except ValueError: pass
        if tok.startswith("eps"):
            try: return float(tok[len("eps"):])
            except ValueError: pass
    return float("inf")


def _load_img(p):
    return np.asarray(Image.open(p).convert("RGB"))


def _find_attacked_strip(run_dir, label, sample_idx, choose_sem, eps_str):
    """Attacked edit strips are saved by _render_attacked_edit with EXP_NAME
    beginning 'attackA-...' or 'attackB-...-eps...-linf-edit_strip'.
    We search the `run_dir` tree because the LOCO 'EXP_NAME' is resolved to a
    different top-level folder by define_argparser. Fall back to any PNG.
    """
    patterns = [
        os.path.join(run_dir, "**", f"{label}-{sample_idx}-{choose_sem}-eps{eps_str}-*edit_strip*.png"),
        os.path.join(run_dir, "..", "..", "..", "**",
                     f"{label}-{sample_idx}-{choose_sem}-eps{eps_str}-*edit_strip*.png"),
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_edit_strip", required=False, default="",
                    help="PNG of clean LOCO edit strip (Phase 1). Optional: if empty, row 2 is blank.")
    ap.add_argument("--attack_sweep_dir", required=True,
                    help="e.g. src/runs/<phase2_exp>/results/sample_idx4729/")
    ap.add_argument("--attack_type", choices=["A", "B"], required=True)
    ap.add_argument("--sample_idx", type=int, default=4729)
    ap.add_argument("--choose_sem", default="l_eye")
    ap.add_argument("--out",        required=True)
    args = ap.parse_args()

    subdirs = sorted(
        glob.glob(os.path.join(args.attack_sweep_dir,
                               "attackA-*" if args.attack_type == "A" else "attackB-*")),
        key=_natural_eps_sort,
    )
    if not subdirs:
        raise SystemExit(f"no attack_{args.attack_type} runs under {args.attack_sweep_dir}")

    n = len(subdirs)
    nrows_per_block = 3 if args.attack_type == "B" else 2
    fig, axes = plt.subplots(nrows_per_block * n, 1,
                             figsize=(10, 3 * nrows_per_block * n))
    if nrows_per_block * n == 1:
        axes = np.array([axes])

    label = "attackB" if args.attack_type == "B" else "attackA"

    for k, run_dir in enumerate(subdirs):
        eps = _natural_eps_sort(run_dir)
        eps_str = f"{eps:g}"
        base = 0 if k == 0 else k * nrows_per_block

        # Row 1 (B only): imperceptibility triptych
        if args.attack_type == "B":
            try:
                clean_px = _load_img(os.path.join(run_dir, "x0_clean.png"))
                adv_px   = _load_img(os.path.join(run_dir, "x0_adv.png"))
                dvis     = _load_img(os.path.join(run_dir, "delta_img_rescaled.png"))
                H = min(clean_px.shape[0], adv_px.shape[0], dvis.shape[0])
                panel = np.concatenate(
                    [clean_px[:H], adv_px[:H], dvis[:H]], axis=1,
                )
                axes[base + 0].imshow(panel)
                axes[base + 0].set_title(f"eps={eps_str} | x0_clean  x0_adv  delta_vis",
                                         fontsize=10)
            except Exception as exc:
                axes[base + 0].text(0.5, 0.5, f"[no imperceptibility imgs: {exc}]",
                                    ha="center", va="center")
            axes[base + 0].axis("off")
            row_clean = base + 1
            row_adv   = base + 2
        else:
            row_clean = base + 0
            row_adv   = base + 1

        # Row 2: clean LOCO strip
        if args.clean_edit_strip and os.path.isfile(args.clean_edit_strip):
            clean_strip = _load_img(args.clean_edit_strip)
            axes[row_clean].imshow(clean_strip)
            axes[row_clean].set_title("clean LOCO edit strip", fontsize=10)
        else:
            axes[row_clean].text(0.5, 0.5,
                                 "(no clean strip provided)",
                                 ha="center", va="center")
        axes[row_clean].axis("off")

        # Row 3: attacked strip
        att_path = _find_attacked_strip(run_dir, label, args.sample_idx,
                                        args.choose_sem, eps_str)
        if att_path:
            att_img = _load_img(att_path)
            axes[row_adv].imshow(att_img)
            axes[row_adv].set_title(f"attacked LOCO edit strip  eps={eps_str}",
                                    fontsize=10)
        else:
            axes[row_adv].text(0.5, 0.5,
                               f"(no attacked strip found for eps={eps_str})",
                               ha="center", va="center")
        axes[row_adv].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[phase2-vis] -> {args.out}")


if __name__ == "__main__":
    main()
