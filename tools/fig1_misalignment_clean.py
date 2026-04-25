"""Figure 1 (clean, single-panel) - direction misalignment vs epsilon.

This is the slide/paper-headline version of Figure 1: only Part A (native
budgets), no secondary top axis, larger fonts, cleaner annotations.

Usage:
  python tools/fig1_misalignment_clean.py \
      --csv_a tools/phase2_figures/attackA_sweep.csv \
      --csv_b tools/phase2_figures/attackB_sweep.csv \
      --out   tools/phase2_figures/fig1_clean.png
"""
from __future__ import annotations
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_a", required=True)
    p.add_argument("--csv_b", required=True)
    p.add_argument("--out",   required=True)
    p.add_argument("--title", default="LOCO direction misalignment under adversarial perturbations")
    args = p.parse_args()

    da = pd.read_csv(args.csv_a).sort_values("eps")
    db = pd.read_csv(args.csv_b).sort_values("eps_img")

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # main curves
    ax.plot(da["eps"], da["misalignment"], "-o",
            color="#1f77b4", lw=2.4, ms=9, mec="white", mew=0.8,
            label=r"Attack A  (latent PGD on $x_t$,  $\varepsilon = \varepsilon_{x_t}$)")
    ax.plot(db["eps_img"], db["misalignment"], "--s",
            color="#d62728", lw=2.4, ms=9, mec="white", mew=0.8,
            label=r"Attack B  (image PGD on $x_0$,  $\varepsilon = \varepsilon_{\mathrm{img}}$)")

    # 45 degree threshold
    ax.axhline(0.5, ls="--", color="#555", lw=1.0, alpha=0.6)
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 0.105, 0.515,
            r"$45^\circ$ rotation threshold",
            ha="right", va="bottom", color="#555", fontsize=10)

    # saturation band (subtle)
    ax.fill_between([-0.005, 0.115], 0.75, 1.02, color="red", alpha=0.05, lw=0)
    ax.text(0.108, 0.94, "saturation plateau",
            ha="right", va="center", color="firebrick", fontsize=10, alpha=0.85)

    ax.set_xlabel(r"adversarial budget  $\varepsilon$  (L$_\infty$, in each attack's native space)")
    ax.set_ylabel(r"direction misalignment   $1 - |\langle v_{\mathrm{clean}}, v_{\mathrm{adv}}\rangle|$")
    ax.set_xlim(-0.005, 0.115)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_title(args.title)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[fig1-clean] -> {args.out}")


if __name__ == "__main__":
    main()
