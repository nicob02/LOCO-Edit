"""Figure 6 (clean) - Pareto: defense reduction vs image-quality cost.

Same data as the original fig6 but with three readability fixes:
  1. Annotate ONLY the Pareto-frontier points (upper-right convex envelope),
     not every (method, param, eps) triple. Avoids label spaghetti.
  2. Drop the in-axis text 'Pareto-good region' which collided with markers.
  3. Plot a dashed Pareto-frontier line connecting the dominant points to
     guide the reader's eye.

Usage:
  python tools/fig6_defense_pareto_clean.py \
      --d2_csv tools/phase2_figures/defense_D2.csv \
      --out    tools/phase3_figures/fig6_defense_pareto_clean.png
"""
from __future__ import annotations
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_EPS_RE = re.compile(r"eps(?:_img)?([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")
MARKER = {"bits": "s", "jpeg": "o", "blur": "^"}
COLOR  = {"bits": "#1f77b4", "jpeg": "#d62728", "blur": "#2ca02c"}


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Return boolean mask of points on the upper-right Pareto frontier
    (maximise both coordinates).
    """
    n = len(points)
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (points[j, 0] >= points[i, 0]
                and points[j, 1] >= points[i, 1]
                and (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                on_front[i] = False
                break
    return on_front


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d2_csv", required=True)
    p.add_argument("--out",    required=True)
    p.add_argument("--y_min",  type=float, default=-100.0,
                   help="clip y-axis at this value to remove unreadable -2000% outliers")
    args = p.parse_args()

    df = pd.read_csv(args.d2_csv)
    df["eps"] = df["run"].apply(lambda s: float(_EPS_RE.search(s).group(1)))
    df["red_pct"] = 100.0 * (df["misalign_undefended"] - df["misalign_defended"]) \
                    / np.maximum(df["misalign_undefended"], 1e-6)

    # Identify Pareto frontier across ALL methods+params+eps.
    pts = df[["psnr_purify_clean", "red_pct"]].values
    df["on_front"] = pareto_front(pts)

    plt.rcParams.update({"font.size": 12,
                         "axes.titlesize": 13,
                         "axes.labelsize": 12,
                         "legend.fontsize": 11})
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    # subtle Pareto-good band
    ax.axhline(0, ls=":", color="grey", alpha=0.6, lw=1)
    ax.axhspan(0, 100, color="green", alpha=0.04, lw=0)

    # scatter all points
    for method, g in df.groupby("purify_method"):
        sizes = 60 + 220 * (g["eps"] / max(g["eps"].max(), 1e-6))
        ax.scatter(g["psnr_purify_clean"], g["red_pct"],
                   s=sizes, marker=MARKER[method], c=COLOR[method],
                   edgecolor="black", lw=0.7, alpha=0.85,
                   label=method)

    # annotate ONLY frontier points (and only those with red_pct > 0,
    # because negative-reduction points on the "Pareto front" are not useful)
    front = df[df["on_front"] & (df["red_pct"] > 0)].copy()
    front = front.sort_values("psnr_purify_clean")
    if not front.empty:
        ax.plot(front["psnr_purify_clean"], front["red_pct"],
                "k--", lw=1.0, alpha=0.55, zorder=0,
                label="Pareto frontier")
        for _, r in front.iterrows():
            label = f"{r['purify_method']}:{r['purify_param']:g}\n$\\varepsilon={r['eps']:g}$"
            ax.annotate(label,
                        xy=(r["psnr_purify_clean"], r["red_pct"]),
                        xytext=(8, 6), textcoords="offset points",
                        fontsize=9.5, color="black",
                        arrowprops=dict(arrowstyle="-", color="grey", lw=0.5, alpha=0.6))

    ax.set_xlabel("PSNR(purify(clean), clean)  (dB)   ---  quality preservation -->")
    ax.set_ylabel("misalignment reduction  (%)        ---  robustness gain -->")
    ax.set_xlim(min(30, df["psnr_purify_clean"].min() - 1),
                max(50, df["psnr_purify_clean"].max() + 1))
    ax.set_ylim(args.y_min, max(80, df["red_pct"].max() + 5))
    ax.grid(alpha=0.3)
    ax.legend(title="method", loc="lower left", framealpha=0.95)
    ax.set_title("D2 Pareto: robustness gain vs quality cost\n"
                 "(marker size = epsilon; only Pareto-front points are labelled)",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[fig6-clean] -> {args.out}  ({int(df['on_front'].sum())} frontier points)")


if __name__ == "__main__":
    main()
