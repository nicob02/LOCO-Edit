"""Figure 6 - Pareto: defense reduction vs image-quality cost.

A good defense has:
  - high reduction% of misalignment (y-axis)
  - low distortion of the clean image, i.e. HIGH psnr_purify_clean (x-axis)

We scatter every (method, param, eps) triple and colour by epsilon. The upper-
right region is Pareto-optimal. bits:4 at large epsilon sits there.
"""
from __future__ import annotations
import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


_EPS_RE = re.compile(r"eps(?:_img)?([0-9.]+)")

MARKER = {"bits": "s", "jpeg": "o", "blur": "^"}
COLOR  = {"bits": "#1f77b4", "jpeg": "#d62728", "blur": "#2ca02c"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d2_csv", required=True)
    p.add_argument("--out",    required=True)
    args = p.parse_args()
    df = pd.read_csv(args.d2_csv)
    df["eps"] = df["run"].apply(lambda s: float(_EPS_RE.search(s).group(1)))
    df["red_pct"] = 100.0 * (df["misalign_undefended"] - df["misalign_defended"]) \
                    / np.maximum(df["misalign_undefended"], 1e-6)

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for method, g in df.groupby("purify_method"):
        sizes = 50 + 140 * (g["eps"] / g["eps"].max())
        sc = ax.scatter(g["psnr_purify_clean"], g["red_pct"],
                        s=sizes, marker=MARKER[method], c=COLOR[method],
                        edgecolor="black", lw=0.6, alpha=0.85,
                        label=method)
        for _, r in g.iterrows():
            ax.annotate(f"{method}:{r['purify_param']:g}\n$\\varepsilon={r['eps']:g}$",
                        xy=(r["psnr_purify_clean"], r["red_pct"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, color="#333")

    ax.axhline(0, ls=":", color="grey", alpha=0.5)
    ax.fill_between([30, 50], 0, 100, color="green", alpha=0.04)
    ax.text(49.5, 55, "Pareto-good region\n(high reduction, low distortion)",
            ha="right", va="center", fontsize=10, color="#1a6a22")

    ax.set_xlabel("PSNR of purify(clean)  vs clean  (dB)   -- quality preservation")
    ax.set_ylabel("misalignment reduction  (%)   -- robustness gain")
    ax.set_xlim(30, 50)
    ax.set_ylim(min(-30, df["red_pct"].min() - 5), max(80, df["red_pct"].max() + 5))
    ax.grid(alpha=0.3)
    ax.legend(title="method", loc="lower left")
    ax.set_title("D2 Pareto frontier: robustness gain vs image-quality cost\n"
                 "(marker size = epsilon; points in upper-right corner are best)",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[fig6] -> {args.out}")


if __name__ == "__main__":
    main()
