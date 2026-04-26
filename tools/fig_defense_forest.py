"""Forest (horizontal bar) plot of D2 defense reduction% at one canonical eps.

Reads several `defense_D2.csv` files (one per sample / per semantic), filters to
the chosen eps_img, averages reduction% across files for each plan, and renders
a horizontal bar with std error bars + per-plan PSNR cost.

This is the cleanest single-number summary of "which defense wins at the
canonical attack budget", and pairs naturally with the multisample heatmap.

Example
-------
# 6-sample average of bits:4, jpeg:75, blur:1.5 at eps=0.031
python tools/fig_defense_forest.py \
    --d2_csv_glob 'src/runs/*phase2_attackB_ms_s*/results/sample_idx*/defense_D2.csv' \
    --eps 0.031 \
    --out tools/report_figures/fig_defense_forest_eye_N6.png \
    --title "D2 ranking at eps=0.031 — eye, 6 samples"
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_EPS_RE = re.compile(r"eps(?:_img)?([0-9.]+)")
DEFAULT_PLANS = ["bits:4", "jpeg:75", "blur:1.5"]


def _eps_from_run(run: str) -> float:
    m = _EPS_RE.search(str(run))
    return float(m.group(1)) if m else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d2_csv_glob", required=True)
    p.add_argument("--eps", type=float, default=0.031)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    p.add_argument(
        "--plans",
        default=",".join(DEFAULT_PLANS),
        help=("Comma-separated 'method:param' plans to keep, "
              "or 'all' to render every plan present in the CSVs."),
    )
    args = p.parse_args()

    paths = sorted(glob.glob(args.d2_csv_glob))
    if not paths:
        raise SystemExit(f"[forest] no CSVs match: {args.d2_csv_glob}")
    print(f"[forest] N CSVs = {len(paths)}")

    rows = []
    for path in paths:
        d = pd.read_csv(path)
        d["eps"] = d["run"].apply(_eps_from_run)
        d = d[np.isclose(d["eps"], args.eps)].copy()
        d["red"] = (
            100.0
            * (d["misalign_undefended"] - d["misalign_defended"])
            / d["misalign_undefended"].clip(lower=1e-6)
        )
        d["plan"] = d.apply(lambda r: f"{r.purify_method}:{r.purify_param:g}", axis=1)
        d["src"] = os.path.basename(os.path.dirname(path))
        rows.append(d[["plan", "red", "psnr_purify_clean", "src"]])
    df = pd.concat(rows, ignore_index=True)
    if args.plans.strip().lower() != "all":
        keep = [s.strip() for s in args.plans.split(",") if s.strip()]
        df = df[df["plan"].isin(keep)]
        if df.empty:
            raise SystemExit(f"[forest] no rows match plans={keep}")

    agg = (
        df.groupby("plan")
        .agg(mean=("red", "mean"), std=("red", "std"),
             psnr=("psnr_purify_clean", "mean"), n=("red", "count"))
        .sort_values("mean")
    )
    agg["std"] = agg["std"].fillna(0.0)

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(agg) + 1.6))
    bars = ax.barh(agg.index, agg["mean"], xerr=agg["std"],
                   color="#4878d0", edgecolor="black", capsize=5)
    ax.axvline(0, color="black", lw=1)
    for i, (lbl, row) in enumerate(agg.iterrows()):
        side = "left" if row["mean"] >= 0 else "right"
        x_text = row["mean"] + (5 if row["mean"] >= 0 else -5)
        ax.text(
            x_text, i,
            f"{row['mean']:+.0f}% ±{row['std']:.0f}   PSNR {row['psnr']:.0f} dB   N={int(row['n'])}",
            va="center", ha=side, fontsize=11,
        )
    ax.set_xlabel(f"reduction%   at  $\\varepsilon_\\mathrm{{img}}$={args.eps:g}   (positive = defense helped)")
    ax.set_title(args.title or f"D2 defense ranking at eps={args.eps:g}  (N CSVs={len(paths)})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[forest] -> {args.out}")


if __name__ == "__main__":
    main()
