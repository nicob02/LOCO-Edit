"""Composite "what does the transferred attack look like?" figure.

Stacks (in order):
    row 0:  source self-attacked LOCO 5-frame strip
    row 1+: each target's transferred-attack LOCO 5-frame strip

So the reader can see, in one figure, that the same delta_img collapses the
source's edit completely yet only partially perturbs each target's edit.

CPU-only: it consumes the strip PNGs produced by the GPU script
``src/tools/phase2_transfer_strip.py``.

Example
-------
SRC_RUN=src/runs/CelebA_HQ_HF-CelebA_HQ_mask-phase2_attackB_ms_s4729/results/sample_idx4729/attackB-linf-eps_img0.031-40steps
python tools/fig_transfer_visual.py \
    --src_strip "$SRC_RUN/transferB-src4729-tgt4729-l_eye-eps0.031-linf-edit_strip.png" \
    --tgt_strip "500=$SRC_RUN/transferB-src4729-tgt500-l_eye-eps0.031-linf-edit_strip.png" \
    --tgt_strip "1000=$SRC_RUN/transferB-src4729-tgt1000-l_eye-eps0.031-linf-edit_strip.png" \
    --tgt_strip "2000=$SRC_RUN/transferB-src4729-tgt2000-l_eye-eps0.031-linf-edit_strip.png" \
    --tgt_strip "5000=$SRC_RUN/transferB-src4729-tgt5000-l_eye-eps0.031-linf-edit_strip.png" \
    --tgt_strip "10000=$SRC_RUN/transferB-src4729-tgt10000-l_eye-eps0.031-linf-edit_strip.png" \
    --transfer_csv "$SRC_RUN/transfer.csv" \
    --source_misalign 0.487 \
    --out tools/report_figures/fig_transfer_visual_eye_eps0.031_4729.png
"""
from __future__ import annotations

import argparse
import glob as _glob
import json as _json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def _load(p: str) -> np.ndarray:
    matches = _glob.glob(p)
    if not matches:
        raise FileNotFoundError(p)
    return np.asarray(Image.open(matches[0]).convert("RGB"))


def _parse_tgt_arg(s: str) -> Tuple[str, str]:
    if "=" not in s:
        raise SystemExit(f"--tgt_strip must be 'idx=path', got: {s}")
    label, path = s.rsplit("=", 1)
    return label.strip(), path.strip()


def _misalign_for(df: pd.DataFrame, idx: int) -> float | None:
    if df is None or df.empty:
        return None
    row = df.loc[df["target_idx"].astype(int) == int(idx)]
    if row.empty:
        return None
    return float(row["misalignment"].iloc[0])


def main() -> None:
    P = argparse.ArgumentParser(description=__doc__)
    P.add_argument("--src_strip", required=True,
                   help="Path to the source self-attack strip PNG.")
    P.add_argument("--tgt_strip", required=True, action="append",
                   help="One '<idx>=<path>' per target, repeated.")
    P.add_argument("--transfer_csv", default="",
                   help="Optional transfer.csv to annotate per-row misalignment.")
    P.add_argument("--source_misalign", type=float, default=None,
                   help="Optional source self-attack misalignment for the title.")
    P.add_argument("--source_idx", type=int, default=None)
    P.add_argument("--out", required=True)
    P.add_argument("--title", default="")
    args = P.parse_args()

    # Optional misalignment lookup.
    df = None
    if args.transfer_csv:
        if args.transfer_csv.lower().endswith(".json"):
            with open(args.transfer_csv) as f:
                blob = _json.load(f)
            rows = blob.get("targets", blob)
            df = pd.DataFrame(rows)
        else:
            df = pd.read_csv(args.transfer_csv)
        if "misalignment" not in df.columns and "cos_true" in df.columns:
            df["misalignment"] = 1.0 - df["cos_true"]

    targets: List[Tuple[str, str]] = [_parse_tgt_arg(s) for s in args.tgt_strip]

    src_im = _load(args.src_strip)
    tgt_ims = []
    for label, path in targets:
        try:
            tgt_ims.append((label, _load(path)))
        except FileNotFoundError:
            print(f"[transfer-vis] WARN: missing strip for target {label}: {path}")

    n_rows = 1 + len(tgt_ims)
    plt.rcParams.update({"font.size": 12})
    fig = plt.figure(figsize=(14, 1.95 * n_rows + 1.0))
    gs = fig.add_gridspec(
        n_rows, 1, hspace=0.35,
        left=0.16, right=0.99, top=0.93, bottom=0.04,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(src_im); ax0.axis("off")
    src_label = (
        rf"source = idx {args.source_idx}, self-attack"
        if args.source_idx is not None else "source self-attack"
    )
    if args.source_misalign is not None:
        src_label += f"   (misalign = {args.source_misalign:.2f})"
    ax0.set_title(src_label, fontsize=14, fontweight="bold", pad=6, loc="left")

    for r, (label, im) in enumerate(tgt_ims, start=1):
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(im); ax.axis("off")
        try:
            idx = int(label)
        except ValueError:
            idx = None
        m = _misalign_for(df, idx) if idx is not None else None
        title = f"target idx {idx}" if idx is not None else label
        if m is not None:
            title += f"   (misalign = {m:.2f})"
        ax.set_title(title, fontsize=13, pad=4, loc="left")

    fig.suptitle(
        args.title or
        r"Cross-sample $\delta_{\mathrm{img}}$ transfer  ($\varepsilon_{\mathrm{img}}=0.031$)",
        fontsize=15, y=0.985,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"[transfer-vis] -> {args.out}")


if __name__ == "__main__":
    main()
