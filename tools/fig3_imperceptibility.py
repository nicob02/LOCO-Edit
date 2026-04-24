"""Figure 3 - Attack B imperceptibility grid.

For every epsilon in the Attack B sweep, display one column with three rows:
    row 0 : x_0^clean
    row 1 : x_0^adv
    row 2 : |delta| x 10  (visualised)

The resulting figure lets a reader confirm visually that the attack is
imperceptible at all but the largest budgets.
"""
from __future__ import annotations
import argparse, glob, os, re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


_EPS_RE = re.compile(r"eps_img([0-9.]+)")


def _load(path):
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", required=True,
                   help="phase2_attackB sweep results dir")
    p.add_argument("--out",       required=True)
    args = p.parse_args()

    subdirs = sorted(glob.glob(os.path.join(args.sweep_dir, "attackB-*")),
                     key=lambda s: float(_EPS_RE.search(s).group(1)))
    n = len(subdirs)
    plt.rcParams.update({"font.size": 11})

    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 7.8),
                             gridspec_kw={"hspace": 0.04, "wspace": 0.04})
    if n == 1:
        axes = axes.reshape(3, 1)

    row_titles = ["clean  $x_0^{\\mathrm{clean}}$",
                  "adversarial  $x_0^{\\mathrm{adv}}$",
                  r"$|\delta|\times 10$ (visualised)"]

    for k, d in enumerate(subdirs):
        eps = float(_EPS_RE.search(d).group(1))
        try:
            imgs = [_load(os.path.join(d, "x0_clean.png")),
                    _load(os.path.join(d, "x0_adv.png")),
                    _load(os.path.join(d, "delta_img_rescaled.png"))]
        except FileNotFoundError as e:
            print(f"[fig3] missing file in {d}: {e}")
            continue
        for r, img in enumerate(imgs):
            axes[r, k].imshow(img)
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
            for spine in axes[r, k].spines.values():
                spine.set_edgecolor("#444"); spine.set_linewidth(0.6)
        axes[0, k].set_title(
            f"$\\varepsilon_{{\\mathrm{{img}}}}={eps:g}$  "
            f"(~{eps*255:.1f}/255)", fontsize=10)

    for r in range(3):
        axes[r, 0].set_ylabel(row_titles[r], fontsize=11)
        axes[r, 0].yaxis.set_label_coords(-0.05, 0.5)

    fig.suptitle("Attack B is perceptually invisible at all but the largest budgets",
                 fontsize=13, y=0.995)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"[fig3] -> {args.out}")


if __name__ == "__main__":
    main()
