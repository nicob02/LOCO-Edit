"""Plot direction misalignment vs eps from Attack A and/or Attack B sweep CSVs.

Usage
-----
    # Just Attack A
    python tools/phase2_plot.py --csv_a path/to/attackA-linf-40steps-sweep.csv

    # Just Attack B
    python tools/phase2_plot.py --csv_b path/to/attackB-linf-40steps-sweep.csv

    # Both on the same axes (latent-space epsilon on top axis, image-space on bottom)
    python tools/phase2_plot.py \
        --csv_a path/to/attackA-linf-40steps-sweep.csv \
        --csv_b path/to/attackB-linf-40steps-sweep.csv \
        --out phase2_figures/misalignment_vs_eps.png

Run on CPU, no GPU required. Just needs matplotlib + pandas.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_a", type=str, default="",
                   help="Path to Attack A sweep CSV (latent-space eps on xt).")
    p.add_argument("--csv_b", type=str, default="",
                   help="Path to Attack B sweep CSV (image-space eps on x0).")
    p.add_argument("--out",   type=str, default="phase2_figures/misalignment_vs_eps.png")
    args = p.parse_args()

    if not (args.csv_a or args.csv_b):
        raise SystemExit("Provide at least one of --csv_a / --csv_b.")

    os.makedirs(Path(args.out).parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if args.csv_a:
        df_a = pd.read_csv(args.csv_a).sort_values("eps")
        ax.plot(df_a["eps"], df_a["misalignment"],
                marker="o", label=r"Attack A (latent $\varepsilon_{x_t}$)")

    if args.csv_b:
        df_b = pd.read_csv(args.csv_b).sort_values("eps_img")
        ax.plot(df_b["eps_img"], df_b["misalignment"],
                marker="s", linestyle="--",
                label=r"Attack B (image-space $\varepsilon_{\mathrm{img}}$)")
        sec = ax.secondary_xaxis(
            "top",
            functions=(lambda e: e * 127.5, lambda e255: e255 / 127.5),
        )
        sec.set_xlabel(r"$\varepsilon_{\mathrm{img}}$ in $[0,1]$ units ($\times 255$)")

    ax.axhline(0.5, linestyle=":", color="grey", alpha=0.5,
               label="misalignment = 0.5 (45° rotation)")
    ax.set_xlabel(r"$\varepsilon$ (L$_\infty$)")
    ax.set_ylabel(r"direction misalignment  $1 - |\langle v_{\mathrm{clean}}, v_{\mathrm{adv}}\rangle|$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("LOCO direction instability under L$_\\infty$ perturbations")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[phase2_plot] wrote {args.out}")


if __name__ == "__main__":
    main()
