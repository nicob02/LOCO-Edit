"""Plot Phase 3 defense effectiveness curves.

For D1 (randomized smoothing) plot misalign_defended vs eps for each sigma.
For D2 (purification)         plot misalign_defended vs eps for each method.
Reference: misalign_undefended curve from the attack sweep itself.
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


_EPS_RE = re.compile(r"eps(?:_img)?([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")


def _eps_from_run(s: str) -> float:
    m = _EPS_RE.search(s)
    return float(m.group(1)) if m else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d1_csv", type=str, default="",
                   help="defense_D1_attack[A/B].csv")
    p.add_argument("--d2_csv", type=str, default="",
                   help="defense_D2.csv")
    p.add_argument("--out",    type=str, default="phase3_figures/defenses.png")
    args = p.parse_args()

    if not (args.d1_csv or args.d2_csv):
        raise SystemExit("provide --d1_csv and/or --d2_csv")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ncols = (1 if args.d1_csv else 0) + (1 if args.d2_csv else 0)
    fig, axes = plt.subplots(1, max(ncols, 1), figsize=(6 * max(ncols, 1), 4.5))
    if ncols == 1:
        axes = [axes]

    col = 0
    if args.d1_csv:
        df = pd.read_csv(args.d1_csv)
        df["eps"] = df["run"].apply(_eps_from_run)
        df = df.sort_values(["sigma", "n_samples", "eps"])
        ax = axes[col]; col += 1
        # undefended baseline
        base = df.drop_duplicates("eps")[["eps", "misalign_undefended"]].sort_values("eps")
        ax.plot(base["eps"], base["misalign_undefended"], "k-", marker="x",
                linewidth=2, label="undefended")
        for (sigma, n), g in df.groupby(["sigma", "n_samples"]):
            ax.plot(g["eps"], g["misalign_defended"],
                    marker="o", label=f"D1  sigma={sigma}  n={n}")
        ax.set_xlabel(r"$\varepsilon$ (L$_\infty$)")
        ax.set_ylabel("direction misalignment")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("D1 randomized smoothing of $x_t$")

    if args.d2_csv:
        df = pd.read_csv(args.d2_csv)
        df["eps"] = df["run"].apply(_eps_from_run)
        df = df.sort_values(["purify_method", "purify_param", "eps"])
        ax = axes[col]; col += 1
        base = df.drop_duplicates("eps")[["eps", "misalign_undefended"]].sort_values("eps")
        ax.plot(base["eps"], base["misalign_undefended"], "k-", marker="x",
                linewidth=2, label="undefended")
        for (m, pv), g in df.groupby(["purify_method", "purify_param"]):
            ax.plot(g["eps"], g["misalign_defended"],
                    marker="s", label=f"D2 {m}={pv}")
        ax.set_xlabel(r"$\varepsilon_{\mathrm{img}}$ (L$_\infty$)")
        ax.set_ylabel("direction misalignment")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("D2 input purification of $x_0$")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[phase3_plot] -> {args.out}")


if __name__ == "__main__":
    main()
