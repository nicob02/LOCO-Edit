"""Figure 2 - DDIM chain amplification.

The central mechanistic question of Attack B is: why does a small image-space
perturbation destroy the direction at all? Answer: the nonlinear DDIM inversion
amplifies it many hundred-fold in the latent space at x_t. We visualise this by
plotting:
  (a) effective eps_xt (from the full nonlinear chain, measured in the sweep)
      vs closed-form linear prediction (an ideal Jacobian approximation), both
      as a function of eps_img.
  (b) chain_gain = eff_inf / closedform_inf, in log scale.
"""
from __future__ import annotations
import argparse, os
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_b", required=True)
    p.add_argument("--out",   required=True)
    args = p.parse_args()
    db = pd.read_csv(args.csv_b).sort_values("eps_img")

    plt.rcParams.update({"font.size": 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # ---- (a) predicted vs actual latent footprint ----
    ax1.plot(db["eps_img"], db["eps_xt_effective_inf"], "-s",
             color="#d62728", lw=2, ms=8, label="actual (nonlinear DDIM)")
    ax1.plot(db["eps_img"], db["eps_xt_closedform_inf"], "--o",
             color="#1f77b4", lw=2, ms=7, label="closed-form (linear)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel(r"$\varepsilon_{\mathrm{img}}$  (L$_\infty$ at $x_0$)")
    ax1.set_ylabel(r"latent footprint at $x_t$   ($\|\Delta x_t\|_\infty$)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="lower right")
    ax1.set_title("(a)  Latent footprint predicted vs actual")

    # ---- (b) chain gain ----
    ax2.plot(db["eps_img"], db["chain_gain_inf"], "-D",
             color="#2ca02c", lw=2, ms=8)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\varepsilon_{\mathrm{img}}$  (L$_\infty$)")
    ax2.set_ylabel(r"chain gain  $= \varepsilon_{x_t}^{\mathrm{actual}} / \varepsilon_{x_t}^{\mathrm{closed}}$")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_title("(b)  Nonlinear amplification factor")

    for _, r in db.iterrows():
        ax2.annotate(f"{r['chain_gain_inf']:.0f}x",
                     xy=(r["eps_img"], r["chain_gain_inf"]),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, color="#1a6a22")

    fig.suptitle("Attack B works because the DDIM chain amplifies the image-space perturbation",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[fig2] -> {args.out}")


if __name__ == "__main__":
    main()
