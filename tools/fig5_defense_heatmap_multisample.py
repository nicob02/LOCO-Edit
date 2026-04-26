"""Multisample (or cross-semantic) D2 heatmap: mean reduction% per (plan, eps).

Aggregates several `defense_D2.csv` files (one per sample or per semantic),
parses the eps from the `run` string, computes relative reduction% per row,
then groups by (plan, eps) and renders a heatmap of the *mean* reduction%
with std + clean-PSNR annotated in each cell.

For the report we usually only want THREE plans (bits:4, jpeg:75, blur:1.5)
so the heatmap stays compact and the takeaway is unambiguous. Pass --plans
to override (use --plans all to render every plan in the CSVs).

Example
-------
# (a) multisample on the eye semantic, 6 samples, 3 plans
python tools/fig5_defense_heatmap_multisample.py \
    --d2_csv_glob 'src/runs/*phase2_attackB_ms_s*/results/sample_idx*/defense_D2.csv' \
    --out tools/report_figures/fig5_defense_heatmap_multisample_eye_N6.png \
    --title "D2 reduction% (Attack B, eye, mean over 6 samples)"
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


def _label(method: str, param: float) -> str:
    return f"{method}:{param:g}"


def _load_all(paths: list[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        d["eps"] = d["run"].apply(_eps_from_run)
        d["red"] = (
            100.0
            * (d["misalign_undefended"] - d["misalign_defended"])
            / d["misalign_undefended"].clip(lower=1e-6)
        )
        d["plan"] = d.apply(lambda r: _label(r.purify_method, r.purify_param), axis=1)
        d["src"] = os.path.basename(os.path.dirname(p))
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def _expand_globs(specs: list[str]) -> list[str]:
    """Glob-expand each spec. Supports comma-separated globs within one --flag."""
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
                   help=("Quoted glob matching defense_D2.csv files. "
                         "May be passed multiple times; comma-separated globs "
                         "inside one flag are also accepted."))
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="D2 reduction% across samples")
    p.add_argument(
        "--plans",
        default=",".join(DEFAULT_PLANS),
        help=("Comma-separated 'method:param' plans to keep "
              "(default = bits:4,jpeg:75,blur:1.5). Pass 'all' to keep every plan."),
    )
    p.add_argument("--vmin", type=float, default=-100.0)
    p.add_argument("--vmax", type=float, default=100.0)
    args = p.parse_args()

    paths = _expand_globs(args.d2_csv_glob)
    if not paths:
        raise SystemExit(
            f"[heatmap-ms] no CSVs match: {args.d2_csv_glob}\n"
            "  Note: Python glob does NOT understand bash brace expansion {a,b,c}.\n"
            "  Pass --d2_csv_glob multiple times instead, or use a wildcard that\n"
            "  covers all the folders (e.g. '*phase2_attackB_*s4729/.../*.csv')."
        )
    print(f"[heatmap-ms] N CSVs = {len(paths)}")
    for q in paths:
        print(f"            {q}")

    raw = _load_all(paths)
    if args.plans.strip().lower() != "all":
        keep = [s.strip() for s in args.plans.split(",") if s.strip()]
        raw = raw[raw["plan"].isin(keep)]
        if raw.empty:
            raise SystemExit(f"[heatmap-ms] no rows match plans={keep}")

    g = raw.groupby(["plan", "eps"])
    mean = g["red"].mean().unstack("eps")
    if args.plans.strip().lower() != "all":
        keep_order = [s.strip() for s in args.plans.split(",") if s.strip() in mean.index]
        if keep_order:
            mean = mean.reindex(index=keep_order)
    mean = mean.dropna(how="all")
    std = g["red"].std().unstack("eps").reindex_like(mean)
    psnr = g["psnr_purify_clean"].mean().unstack("eps").reindex_like(mean)

    n_rows, n_cols = mean.shape
    fig, ax = plt.subplots(figsize=(1.3 * n_cols + 3.0, 0.85 * n_rows + 1.6))

    im = ax.imshow(mean.values, cmap="RdYlGn", vmin=args.vmin, vmax=args.vmax,
                   aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"{c:g}" for c in mean.columns])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(mean.index)
    ax.set_xlabel(r"$\varepsilon_{\mathrm{img}}$  (L$_\infty$)")
    ax.set_ylabel("purification plan")
    fig.colorbar(im, ax=ax, label="mean reduction%   (positive = defense helped)")

    for i in range(n_rows):
        for j in range(n_cols):
            m = mean.values[i, j]
            s = std.values[i, j]
            ps = psnr.values[i, j]
            txt = f"{m:+.0f}%\n±{s:.0f}\n{ps:.0f} dB"
            color = "white" if abs(m) > 50 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=color)

    ax.set_title(f"{args.title}   (N={len(paths)})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[heatmap-ms] -> {args.out}")


if __name__ == "__main__":
    main()
