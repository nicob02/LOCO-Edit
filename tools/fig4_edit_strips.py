"""Figure 4 - clean vs attacked LOCO edit strips (large, readable).

The previous phase2_visual_compare.py stretched the tall canvas so each strip
became ~30 pixels tall. We instead use a gridspec whose row heights match the
native strip aspect ratio, producing a figure where you can actually read the
edits.

Layout (rows, left-to-right for each row):
    row 0         : clean edit strip at the same lambda range
    rows 1..n_eps : attacked edit strips, one per epsilon

Works for both Attack A and Attack B sweeps.
"""
from __future__ import annotations
import argparse, glob, os, re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


_EPS_RE = re.compile(r"eps(?:_img)?([0-9.]+)")


def _load(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _find_strip(run_dir, sample_idx, choose_sem, eps_val, attack_label):
    eps_str = f"{eps_val:g}"
    patterns = [
        os.path.join(run_dir, "**",
            f"{attack_label}-{sample_idx}-{choose_sem}-eps{eps_str}-*edit_strip*.png"),
        os.path.join(run_dir, "..", "..", "..", "**",
            f"{attack_label}-{sample_idx}-{choose_sem}-eps{eps_str}-*edit_strip*.png"),
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir",   required=True)
    p.add_argument("--attack",      choices=["A", "B"], required=True)
    p.add_argument("--clean_strip", default="",
                   help="Phase 1 clean edit strip (optional)")
    p.add_argument("--sample_idx",  type=int, default=4729)
    p.add_argument("--choose_sem",  default="l_eye")
    p.add_argument("--out",         required=True)
    args = p.parse_args()

    prefix = "attackA" if args.attack == "A" else "attackB"
    raw = glob.glob(os.path.join(args.sweep_dir, f"{prefix}-*"))
    raw = [p for p in raw if os.path.isdir(p) and _EPS_RE.search(p)]
    subdirs = sorted(raw, key=lambda s: float(_EPS_RE.search(s).group(1)))
    if not subdirs:
        raise SystemExit(f"no {prefix}-* directories with eps* under {args.sweep_dir}")
    n_eps = len(subdirs)

    strips = []
    if args.clean_strip and os.path.isfile(args.clean_strip):
        strips.append(("clean  (no attack)", _load(args.clean_strip)))

    collected = []
    for d in subdirs:
        eps = float(_EPS_RE.search(d).group(1))
        path = _find_strip(d, args.sample_idx, args.choose_sem, eps, prefix)
        if path is None:
            print(f"[fig4] no edit strip for eps={eps} under {d}")
            continue
        collected.append((eps, path))

    if not strips and collected:
        eps_min, path_min = collected[0]
        strips.append((f"~clean  (smallest $\\varepsilon={eps_min:g}$, misalign$\\approx$0)",
                       _load(path_min)))
        collected = collected[1:]

    for eps, path in collected:
        strips.append((f"attacked  $\\varepsilon={eps:g}$", _load(path)))

    if not strips:
        raise SystemExit("no strips found")

    plt.rcParams.update({"font.size": 12})
    h_per_row = 1.6
    fig, axes = plt.subplots(len(strips), 1,
                             figsize=(14, h_per_row * len(strips)),
                             gridspec_kw={"hspace": 0.25})
    if len(strips) == 1:
        axes = [axes]
    for ax, (label, img) in zip(axes, strips):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center",
                      fontsize=11, labelpad=60)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555"); spine.set_linewidth(0.8)

    attack_word = "latent-space" if args.attack == "A" else "image-space"
    fig.suptitle(
        f"LOCO edit rows under Attack {args.attack} ({attack_word} PGD),  "
        f"sample_idx={args.sample_idx},  semantic = {args.choose_sem}",
        fontsize=13, y=1.0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"[fig4] -> {args.out}  ({len(strips)} rows)")


if __name__ == "__main__":
    main()
