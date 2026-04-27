"""5-row mega figure: edit-transformation under attack and defense.

Stacks five 5-frame LOCO edit strips, one per condition:

    row 1: clean (Phase 1)            -> the gold-standard edit
    row 2: attacked B (eps=0.031)     -> identity destroyed at extreme frames
    row 3: defended bits:4            -> the strongest defense by direction recovery
    row 4: defended jpeg:75           -> mid defense
    row 5: defended blur:1.5          -> weakest defense (over-smooths)

This is the single figure that tells the entire defense story end-to-end:
direction recovery, locality breakdown, qualitative edit quality.

Pure-CPU script. It only consumes already-saved 5-frame strip PNGs
(produced by phase1 / phase2_attack_b / phase3_defense_strip).

Example
-------
RUN=src/runs/CelebA_HQ_HF-CelebA_HQ_mask-phase2_attackB_ms_s4729/results/sample_idx4729/attackB-linf-eps_img0.031-40steps
CLEAN=src/runs/CelebA_HQ_HF-CelebA_HQ_mask-phase1_ms_s4729/results/sample_idx4729/4729-Edit-randomFalse_xt-noise-False_l_eye-edit_0.6T_null_proj_True_rank5_scale_0.5-pc_000.png
python tools/fig_defense_strips_full.py \
    --row "clean (Phase 1)=$CLEAN" \
    --row "attacked  B  (eps=0.031)=$RUN/attackB-4729-l_eye-eps0.031-linf-edit_strip.png" \
    --row "defended bits:4=$RUN/defenseD2-bits-4-edit_strip.png" \
    --row "defended jpeg:75=$RUN/defenseD2-jpeg-75-edit_strip.png" \
    --row "defended blur:1.5=$RUN/defenseD2-blur-1.5-edit_strip.png" \
    --out tools/report_figures/fig_defense_strips_full_eye_eps0.031_4729.png
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def _load(p: str) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def _parse_row(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"--row must be 'label=path', got: {spec!r}")
    label, path = spec.rsplit("=", 1)
    return label.strip(), path.strip()


def main() -> None:
    P = argparse.ArgumentParser(description=__doc__)
    P.add_argument(
        "--row", action="append", required=True,
        help=("Repeatable. Format: 'label=path'. "
              "Order from top (clean / cleanest) to bottom (worst defense). "
              "Each path must be a 5-frame horizontal LOCO edit strip."),
    )
    P.add_argument("--out", required=True)
    P.add_argument(
        "--title",
        default=(r"LOCO edit transformation under attack and defense  "
                 r"($\varepsilon_{\mathrm{img}}=0.031$,  semantic = l_eye,  idx 4729)"),
    )
    P.add_argument("--frame_label_top", default=r"$-2\,\mathrm{d}\lambda \ldots +2\,\mathrm{d}\lambda$",
                   help="Optional small caption above the top row.")
    args = P.parse_args()

    rows: List[Tuple[str, np.ndarray]] = []
    for spec in args.row:
        label, path = _parse_row(spec)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"row '{label}': file not found: {path}")
        rows.append((label, _load(path)))

    n = len(rows)
    plt.rcParams.update({"font.size": 12})
    fig = plt.figure(figsize=(14.5, 1.95 * n + 1.0))
    gs = fig.add_gridspec(
        n, 1, hspace=0.32,
        left=0.18, right=0.99, top=0.93, bottom=0.04,
    )

    for k, (label, im) in enumerate(rows):
        ax = fig.add_subplot(gs[k, 0])
        ax.imshow(im); ax.axis("off")
        ax.set_title(label, fontsize=14, fontweight="bold",
                     loc="left", pad=6)

    fig.suptitle(args.title, fontsize=14.5, y=0.985)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"[def-strips-full] -> {args.out}")


if __name__ == "__main__":
    main()
