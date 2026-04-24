"""Figure 1 - direction misalignment vs epsilon.

Produces a single publication-quality figure with two panels:
  (a) Native budgets. Attack A's eps_{x_t} vs Attack B's eps_img on a shared
      x-axis; a secondary top axis shows eps_img in 0..255 scale.
  (b) Fair latent comparison. Both attacks plotted against the effective
      latent-space perturbation they induce at x_t, on a log x-axis.
"""
from __future__ import annotations
import argparse, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_a", required=True)
    p.add_argument("--csv_b", required=True)
    p.add_argument("--out",   required=True)
    args = p.parse_args()

    da = pd.read_csv(args.csv_a).sort_values("eps")
    db = pd.read_csv(args.csv_b).sort_values("eps_img")

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- (a) native budgets ----
    ax1.plot(da["eps"], da["misalignment"], "-o", color="#1f77b4", lw=2, ms=7,
             label=r"Attack A  (latent $\varepsilon_{x_t}$)")
    ax1.plot(db["eps_img"], db["misalignment"], "--s", color="#d62728", lw=2, ms=7,
             label=r"Attack B  (image-space $\varepsilon_{\mathrm{img}}$)")
    ax1.axhline(0.5, ls=":", color="grey", alpha=0.7)
    ax1.text(0.095, 0.52, "45° rotation threshold", color="grey",
             ha="right", fontsize=9)
    ax1.fill_between([0, 0.11], 0.75, 1.02, color="red", alpha=0.05)
    ax1.text(0.085, 0.95, "saturation plateau", color="firebrick",
             ha="right", fontsize=9)

    sec = ax1.secondary_xaxis("top",
        functions=(lambda e: e * 255, lambda e255: e255 / 255))
    sec.set_xlabel(r"$\varepsilon_{\mathrm{img}}$ in [0,255]", labelpad=8)

    ax1.set_xlabel(r"$\varepsilon$   (L$_\infty$, in the attack's native space)")
    ax1.set_ylabel(r"direction misalignment   $1 - |\langle v_{\mathrm{clean}}, v_{\mathrm{adv}}\rangle|$")
    ax1.set_xlim(-0.002, 0.11)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", framealpha=0.95)
    ax1.set_title("(a)  Native budgets")

    # ---- (b) fair latent comparison ----
    ax2.plot(da["eps"], da["misalignment"], "-o", color="#1f77b4", lw=2, ms=7,
             label=r"Attack A  (eps == $\varepsilon_{x_t}$)")
    if "eps_xt_effective_inf" in db.columns:
        ax2.plot(db["eps_xt_effective_inf"], db["misalignment"], "--s",
                 color="#d62728", lw=2, ms=7,
                 label=r"Attack B  (effective $\varepsilon_{x_t}$, post-DDIM)")
    ax2.axhline(0.5, ls=":", color="grey", alpha=0.7)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"effective  $\varepsilon_{x_t}$   (L$_\infty$, latent space; log)")
    ax2.set_ylabel("direction misalignment")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(loc="lower right", framealpha=0.95)
    ax2.set_title("(b)  Fair comparison on matched latent footprint")

    # annotate the chain-gain ratio at eps=0.02
    if "eps_xt_effective_inf" in db.columns:
        a_eps_at_mid = 0.02
        b_eff_at_mid = float(np.interp(0.45, db["misalignment"], db["eps_xt_effective_inf"]))
        ax2.annotate(f"B needs {b_eff_at_mid/a_eps_at_mid:.0f}x larger "
                     r"$\varepsilon_{x_t}$ to match A",
                     xy=(b_eff_at_mid, 0.45),
                     xytext=(b_eff_at_mid*0.3, 0.12),
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                     fontsize=9, color="black")

    fig.suptitle("LOCO edit: direction misalignment under adversarial perturbations",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[fig1] -> {args.out}")


if __name__ == "__main__":
    main()
