"""Figure 8 - clean defense curves with zoomed inset (D1 + D2).

Produces a 2-panel figure:
  (a) D1 randomized smoothing of x_t vs Attack B: undefended + all (sigma, n).
      Zoomed inset for eps in [0, 0.02] where the ~10% gain lives.
  (b) D2 input purification. Show only the 3 winning curves
      (bits:4, bits:6, jpeg:75) plus undefended.
"""
from __future__ import annotations
import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


_EPS_RE = re.compile(r"eps(?:_img)?([0-9]+(?:\.[0-9]+)?)")

def _eps(s):
    m = _EPS_RE.search(s)
    return float(m.group(1)) if m else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d1_csv", required=True)
    p.add_argument("--d2_csv", required=True)
    p.add_argument("--out",    required=True)
    args = p.parse_args()

    d1 = pd.read_csv(args.d1_csv); d1["eps"] = d1["run"].apply(_eps)
    d2 = pd.read_csv(args.d2_csv); d2["eps"] = d2["run"].apply(_eps)

    plt.rcParams.update({"font.size": 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- (a) D1 ----
    base = d1.drop_duplicates("eps")[["eps", "misalign_undefended"]].sort_values("eps")
    ax1.plot(base["eps"], base["misalign_undefended"], "k-x", lw=2.2, ms=9, label="undefended")
    sigmas = sorted(d1["sigma"].unique())
    cmap = plt.get_cmap("viridis")
    for i, sig in enumerate(sigmas):
        g_small = d1[(d1["sigma"] == sig) & (d1["n_samples"] == d1["n_samples"].max())].sort_values("eps")
        ax1.plot(g_small["eps"], g_small["misalign_defended"],
                 "-o", color=cmap(i / max(1, len(sigmas) - 1)), lw=1.6, ms=6,
                 label=rf"$\sigma={sig}$  n={int(g_small['n_samples'].iloc[0])}")
    ax1.set_xlabel(r"$\varepsilon_{\mathrm{img}}$  (L$_\infty$)")
    ax1.set_ylabel("misalignment")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("(a)  D1 randomized smoothing of $x_t$")
    ax1.grid(alpha=0.3); ax1.legend(loc="lower right", fontsize=9)

    axins = inset_axes(ax1, width="42%", height="45%", loc="upper left",
                       borderpad=2.5)
    axins.plot(base["eps"], base["misalign_undefended"], "k-x", lw=2, ms=7)
    for i, sig in enumerate(sigmas):
        g = d1[(d1["sigma"] == sig) & (d1["n_samples"] == d1["n_samples"].max())].sort_values("eps")
        axins.plot(g["eps"], g["misalign_defended"], "-o",
                   color=cmap(i / max(1, len(sigmas) - 1)), lw=1.3, ms=4)
    axins.set_xlim(0, 0.02); axins.set_ylim(0, 0.55)
    axins.grid(alpha=0.3); axins.set_title("zoom: low $\\varepsilon$", fontsize=9)

    # ---- (b) D2 ----
    base2 = d2.drop_duplicates("eps")[["eps", "misalign_undefended"]].sort_values("eps")
    ax2.plot(base2["eps"], base2["misalign_undefended"], "k-x", lw=2.2, ms=9, label="undefended")
    winners = [("bits", 4.0, "#1f77b4", "s"),
               ("bits", 6.0, "#56b4e9", "s"),
               ("jpeg", 75.0, "#d62728", "o"),
               ("blur", 1.5,  "#2ca02c", "^")]
    for m, param, c, mk in winners:
        g = d2[(d2["purify_method"] == m) & (d2["purify_param"] == param)].sort_values("eps")
        if g.empty: continue
        ax2.plot(g["eps"], g["misalign_defended"], lw=1.8, ms=7, marker=mk,
                 color=c, label=f"{m}:{param:g}")
    ax2.set_xlabel(r"$\varepsilon_{\mathrm{img}}$  (L$_\infty$)")
    ax2.set_ylabel("misalignment")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("(b)  D2 input purification (best plans)")
    ax2.grid(alpha=0.3); ax2.legend(loc="lower right", fontsize=9)

    fig.suptitle("Defense effectiveness against Attack B",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[fig8] -> {args.out}")


if __name__ == "__main__":
    main()
