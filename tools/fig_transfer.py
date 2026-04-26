"""Cross-sample transfer figure.

Reads a transfer.csv (or transfer_summary.json) produced by
src/tools/phase2_transfer.py and renders a horizontal bar chart of the
target-side direction misalignment under the source's adversarial
perturbation. Reference lines:

    - the source's own self-attacked misalignment (red dashed)
    - the 45-degree rotation threshold at 0.5 (grey dotted)

Example
-------
SRC_RUN=src/runs/CelebA_HQ_HF-CelebA_HQ_mask-phase2_attackB_ms_s4729/results/sample_idx4729/attackB-linf-eps_img0.031-40steps
python tools/fig_transfer.py \
    --transfer_csv "$SRC_RUN/transfer.csv" \
    --source_misalign 0.487 \
    --out tools/report_figures/fig_transfer_bars_4729.png
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load(path: str) -> pd.DataFrame:
    if path.lower().endswith(".json"):
        with open(path) as f:
            blob = json.load(f)
        rows = blob["targets"] if isinstance(blob, dict) and "targets" in blob else blob
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def main() -> None:
    P = argparse.ArgumentParser(description=__doc__)
    P.add_argument("--transfer_csv", required=True,
                   help="Path to transfer.csv or transfer_summary.json.")
    P.add_argument("--source_misalign", type=float, required=True,
                   help="Undefended source-self misalignment at the same eps "
                        "(read from attackB-linf-40steps-sweep.csv).")
    P.add_argument("--out", required=True)
    P.add_argument("--title", default="")
    args = P.parse_args()

    df = _load(args.transfer_csv)
    if "misalignment" not in df.columns and "cos_true" in df.columns:
        df["misalignment"] = 1.0 - df["cos_true"]
    df = df.sort_values("target_idx").reset_index(drop=True)

    mean = df["misalignment"].mean()
    std = df["misalignment"].std()

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(df) + 1.6))
    bars = ax.barh([str(int(t)) for t in df["target_idx"]],
                   df["misalignment"], color="#1f77b4", edgecolor="black")
    for i, m in enumerate(df["misalignment"]):
        ax.text(m + 0.01, i, f"{m:.2f}", va="center", fontsize=11)
    ax.axvline(args.source_misalign, color="red", lw=1.5, ls="--",
               label=f"source self-attack = {args.source_misalign:.2f}")
    ax.axvline(0.5, color="grey", lw=1.0, ls=":",
               label=r"45$^\circ$ rotation threshold")
    ax.axvline(mean, color="black", lw=1.0, ls="-", alpha=0.6,
               label=f"transfer mean = {mean:.2f} ± {std:.2f}")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"misalignment under transferred $\delta_{\mathrm{img}}$")
    ax.set_ylabel("target sample idx")
    ax.set_title(args.title or "Cross-sample δ transfer (eps_img=0.031, source=4729)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[transfer-fig] -> {args.out}")


if __name__ == "__main__":
    main()
