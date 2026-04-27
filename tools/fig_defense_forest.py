"""Forest (horizontal bar) plot of D2 defense reduction% at one canonical eps.

Reads several `defense_D2.csv` files (one per sample / per semantic), filters to
the chosen eps_img, averages reduction% across files for each plan, and renders
a horizontal bar with std error bars + per-plan PSNR cost.

This is the cleanest single-number summary of "which defense wins at the
canonical attack budget", and pairs naturally with the multisample heatmap.

Example
-------
# 6-sample average of bits:4, jpeg:75, blur:1.5 at eps=0.031
python tools/fig_defense_forest.py \
    --d2_csv_glob 'src/runs/*phase2_attackB_ms_s*/results/sample_idx*/defense_D2.csv' \
    --eps 0.031 \
    --out tools/report_figures/fig_defense_forest_eye_N6.png \
    --title "D2 ranking at eps=0.031 — eye, 6 samples"
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_EPS_RE = re.compile(r"eps(?:_img)?([0-9.]+)")
DEFAULT_PLANS = ["bits:4", "jpeg:75", "blur:1.5"]


def _eps_from_run(run: str) -> float:
    m = _EPS_RE.search(str(run))
    return float(m.group(1)) if m else float("nan")


def _expand_globs(specs: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        for pat in spec.split(","):
            pat = pat.strip()
            if not pat:
                continue
            for path in glob.glob(pat):
                if path not in seen:
                    seen.add(path)
                    out.append(path)
    return sorted(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d2_csv_glob", required=True, action="append",
                   help=("Glob matching defense_D2.csv files. May be passed "
                         "multiple times; comma-separated globs are also accepted."))
    p.add_argument("--eps", type=float, default=0.031)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    p.add_argument(
        "--plans",
        default=",".join(DEFAULT_PLANS),
        help=("Comma-separated 'method:param' plans to keep, "
              "or 'all' to render every plan present in the CSVs."),
    )
    args = p.parse_args()

    paths = _expand_globs(args.d2_csv_glob)
    if not paths:
        raise SystemExit(
            f"[forest] no CSVs match: {args.d2_csv_glob}\n"
            "  Note: Python glob does NOT understand bash brace expansion {a,b,c}.\n"
            "  Pass --d2_csv_glob multiple times, or use a wildcard."
        )
    print(f"[forest] N CSVs = {len(paths)}")

    rows = []
    for path in paths:
        d = pd.read_csv(path)
        d["eps"] = d["run"].apply(_eps_from_run)
        d = d[np.isclose(d["eps"], args.eps)].copy()
        d["red"] = (
            100.0
            * (d["misalign_undefended"] - d["misalign_defended"])
            / d["misalign_undefended"].clip(lower=1e-6)
        )
        d["plan"] = d.apply(lambda r: f"{r.purify_method}:{r.purify_param:g}", axis=1)
        d["src"] = os.path.basename(os.path.dirname(path))
        rows.append(d[["plan", "red", "psnr_purify_clean", "src"]])
    df = pd.concat(rows, ignore_index=True)
    if args.plans.strip().lower() != "all":
        keep = [s.strip() for s in args.plans.split(",") if s.strip()]
        df = df[df["plan"].isin(keep)]
        if df.empty:
            raise SystemExit(f"[forest] no rows match plans={keep}")

    agg = (
        df.groupby("plan")
        .agg(mean=("red", "mean"), std=("red", "std"),
             psnr=("psnr_purify_clean", "mean"), n=("red", "count"))
        .sort_values("mean")
    )
    agg["std"] = agg["std"].fillna(0.0)

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(10.0, 0.55 * len(agg) + 1.8))
    ax.barh(agg.index, agg["mean"], xerr=agg["std"],
            color="#4878d0", edgecolor="black",
            error_kw={"capsize": 5, "elinewidth": 1.2})
    ax.axvline(0, color="black", lw=1)

    # Place each label fully outside the error bar so it never collides with
    # the bar fill or the y-axis labels.
    pad = 6.0
    label_x_pos = 0.0
    label_x_neg = 0.0
    for i, (_, row) in enumerate(agg.iterrows()):
        end = row["mean"] + (row["std"] if row["mean"] >= 0 else -row["std"])
        if row["mean"] >= 0:
            x_text, ha = end + pad, "left"
            label_x_pos = max(label_x_pos, x_text + 60)
        else:
            x_text, ha = end - pad, "right"
            label_x_neg = min(label_x_neg, x_text - 60)
        ax.text(
            x_text, i,
            f"{row['mean']:+.0f}% ± {row['std']:.0f}    "
            f"PSNR {row['psnr']:.0f} dB    N={int(row['n'])}",
            va="center", ha=ha, fontsize=11,
        )

    cur_lo, cur_hi = ax.get_xlim()
    ax.set_xlim(min(cur_lo, label_x_neg), max(cur_hi, label_x_pos))
    ax.set_xlabel(
        rf"reduction%   at  $\varepsilon_{{\mathrm{{img}}}}={args.eps:g}$"
        r"    (positive = defense helped)"
    )
    ax.set_title(args.title or f"D2 defense ranking at eps={args.eps:g}  (N CSVs={len(paths)})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[forest] -> {args.out}")


if __name__ == "__main__":
    main()
