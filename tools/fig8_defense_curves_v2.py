"""F8 v2 - Multisample two-panel D2 curves with mean +/- std.

(a) Defended misalignment vs eps_img, with a dashed undefended baseline.
(b) Relative reduction% vs eps_img (positive = defense helped).

Only renders three plans by default - bits:4 (winner), jpeg:75 (median),
blur:1.5 (loser) - so the report figure stays uncluttered. Override with
--plans for sensitivity analysis or appendix figures.

Example
-------
python tools/fig8_defense_curves_v2.py \
    --d2_csv_glob 'src/runs/*phase2_attackB_ms_s*/results/sample_idx*/defense_D2.csv' \
    --out tools/report_figures/fig8_defense_D2_multisample_v2.png \
    --title "D2 - Attack B - eye - 6 samples"
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

# (label, method, param, hex color, marker)
DEFAULT_PLANS = [
    ("bits:4 (winner)",  "bits", 4.0, "#1f77b4", "s"),
    ("jpeg:75 (median)", "jpeg", 75.0, "#d62728", "o"),
    ("blur:1.5 (loser)", "blur", 1.5, "#2ca02c", "^"),
]


def _eps_from_run(run: str) -> float:
    m = _EPS_RE.search(str(run))
    return float(m.group(1)) if m else float("nan")


def _parse_plans(arg: str) -> list[tuple[str, str, float, str, str]]:
    if not arg:
        return DEFAULT_PLANS
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    markers = ["s", "o", "^", "v", "D", "P"]
    out = []
    for i, tok in enumerate(arg.split(",")):
        method, param = tok.split(":")
        out.append((tok.strip(), method.strip(), float(param),
                    palette[i % len(palette)], markers[i % len(markers)]))
    return out


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
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="D2 defenses across samples")
    p.add_argument("--plans", default="", help="Optional 'm:p,m:p' override.")
    args = p.parse_args()

    plans = _parse_plans(args.plans)

    paths = _expand_globs(args.d2_csv_glob)
    if not paths:
        raise SystemExit(
            f"[fig8v2] no CSVs match: {args.d2_csv_glob}\n"
            "  Note: Python glob does NOT understand bash brace expansion {a,b,c}.\n"
            "  Pass --d2_csv_glob multiple times, or use a wildcard."
        )
    print(f"[fig8v2] N CSVs = {len(paths)}  plans = {[p[0] for p in plans]}")

    dfs = []
    for q in paths:
        d = pd.read_csv(q)
        d["eps"] = d["run"].apply(_eps_from_run)
        d["red"] = (
            100.0
            * (d["misalign_undefended"] - d["misalign_defended"])
            / d["misalign_undefended"].clip(lower=1e-6)
        )
        d["src"] = os.path.dirname(q)
        dfs.append(d)
    raw = pd.concat(dfs, ignore_index=True)

    # Undefended baseline: the same value repeats across all plans within a row
    # so deduplicate on (src, eps) before averaging.
    base = (
        raw[["src", "eps", "misalign_undefended"]]
        .drop_duplicates(["src", "eps"])
        .groupby("eps")["misalign_undefended"]
    )
    base_mean = base.mean(); base_std = base.std()

    plt.rcParams.update({"font.size": 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax1.errorbar(base_mean.index, base_mean.values, yerr=base_std.values,
                 fmt="--x", color="black", lw=2, capsize=4, label="undefended")

    for label, method, param, color, marker in plans:
        sub = raw[(raw.purify_method == method) & (np.isclose(raw.purify_param, param))]
        if sub.empty:
            print(f"[fig8v2] WARN: no rows for {label}")
            continue
        gm = sub.groupby("eps")["misalign_defended"]
        gr = sub.groupby("eps")["red"]
        ax1.errorbar(gm.mean().index, gm.mean().values, yerr=gm.std().values,
                     fmt=f"-{marker}", color=color, lw=2, ms=8, capsize=4,
                     label=label)
        ax2.errorbar(gr.mean().index, gr.mean().values, yerr=gr.std().values,
                     fmt=f"-{marker}", color=color, lw=2, ms=8, capsize=4,
                     label=label)

    ax1.axhline(0.5, ls=":", color="grey", alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\varepsilon_{\mathrm{img}}$  (L$_\infty$, log scale)")
    ax1.set_ylabel(r"misalignment  $1-|\langle v_{\mathrm{clean}},v_{\mathrm{adv}}\rangle|$")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("(a) Defended misalignment   (lower = better)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right")

    ax2.axhline(0, color="black", lw=1)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\varepsilon_{\mathrm{img}}$  (L$_\infty$, log scale)")
    ax2.set_ylabel("relative reduction%   (positive = defense helped)")
    ax2.set_title("(b) Defense effect")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right")

    fig.suptitle(f"{args.title}   (N={len(paths)})", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[fig8v2] -> {args.out}")


if __name__ == "__main__":
    main()
