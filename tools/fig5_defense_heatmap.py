"""Figure 5 - defense heatmap: reduction% across (method x epsilon).

For every (purify_method, purify_param) row and every epsilon, compute
the relative reduction of misalignment:

    reduction% = 100 * (misalign_undef - misalign_def) / max(misalign_undef, 1e-6)

and render as a heatmap (rows = method, cols = eps). Positive values (green)
mean the defense helped; negative values (red) mean the defense introduced
more rotation than the attack did. We annotate each cell with the numeric
value for readability.
"""
from __future__ import annotations
import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_EPS_RE = re.compile(r"eps(?:_img)?([0-9.]+)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d2_csv", required=True)
    p.add_argument("--out",    required=True)
    args = p.parse_args()

    df = pd.read_csv(args.d2_csv)
    df["eps"] = df["run"].apply(lambda s: float(_EPS_RE.search(s).group(1)))
    df["red_pct"] = np.where(
        df["misalign_undefended"] > 1e-6,
        100.0 * (df["misalign_undefended"] - df["misalign_defended"])
               / df["misalign_undefended"],
        0.0,
    )
    df["label"] = df.apply(lambda r: f"{r['purify_method']}:{r['purify_param']:g}", axis=1)

    pivot = df.pivot(index="label", columns="eps", values="red_pct")
    psnr  = df.pivot(index="label", columns="eps", values="psnr_purify_clean")
    pivot = pivot.reindex(
        sorted(pivot.index,
               key=lambda x: ({"bits": 0, "jpeg": 1, "blur": 2}[x.split(":")[0]], x))
    )
    psnr = psnr.reindex(pivot.index)

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    vmax = min(vmax, 100)
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                   aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{e:g}" for e in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(r"$\varepsilon_{\mathrm{img}}$  (L$_\infty$)")
    ax.set_ylabel("purification plan")
    ax.set_title("D2 input purification: relative misalignment reduction (%) per (method, epsilon)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            p_ = psnr.values[i, j]
            ax.text(j, i, f"{v:+.0f}%\n{p_:.0f} dB",
                    ha="center", va="center", fontsize=9,
                    color="black" if abs(v) < vmax * 0.6 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("reduction%   (positive = defense helped)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[fig5] -> {args.out}")


if __name__ == "__main__":
    main()
