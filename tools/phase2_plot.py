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

    df_a = pd.read_csv(args.csv_a).sort_values("eps") if args.csv_a else None
    df_b = pd.read_csv(args.csv_b).sort_values("eps_img") if args.csv_b else None

    # Top panel: native epsilons (A on xt, B on img).
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    if df_a is not None:
        ax.plot(df_a["eps"], df_a["misalignment"],
                marker="o", label=r"Attack A (latent $\varepsilon_{x_t}$)")

    if df_b is not None:
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
    ax.set_xlabel(r"$\varepsilon$ (L$_\infty$, native space)")
    ax.set_ylabel(r"direction misalignment  $1 - |\langle v_{\mathrm{clean}}, v_{\mathrm{adv}}\rangle|$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Native budget: $\\varepsilon_{x_t}$ (A) vs $\\varepsilon_{\\mathrm{img}}$ (B)")

    # Right panel: FAIR comparison -- both plotted against effective epsilon in
    # latent-xt space. Attack A uses its own eps; Attack B uses the EFFECTIVE
    # delta_xt produced by the full nonlinear DDIM chain (column added in
    # phase2_attack_b.py). If Attack B sits ABOVE Attack A at the same
    # effective-eps_xt, the image-space attacker is stronger than the
    # latent-space attacker at a matched latent footprint.
    if df_a is not None:
        ax2.plot(df_a["eps"], df_a["misalignment"],
                 marker="o", label=r"Attack A (latent)")
    if df_b is not None and "eps_xt_effective_inf" in df_b.columns:
        ax2.plot(df_b["eps_xt_effective_inf"], df_b["misalignment"],
                 marker="s", linestyle="--",
                 label=r"Attack B (effective $\varepsilon_{x_t}$)")
    elif df_b is not None:
        print("[phase2_plot] WARN: Attack B CSV lacks 'eps_xt_effective_inf'; "
              "re-run attack_b with the updated phase2_attack_b.py to get the "
              "fair-comparison curve.")
    ax2.axhline(0.5, linestyle=":", color="grey", alpha=0.5)
    ax2.set_xlabel(r"$\varepsilon_{x_t}$ (L$_\infty$, latent space)")
    ax2.set_ylabel("direction misalignment")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")
    ax2.set_title("Fair comparison: same latent-space footprint")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[phase2_plot] wrote {args.out}")


if __name__ == "__main__":
    main()
