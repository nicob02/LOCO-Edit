"""Tier-1 aggregation: stitch per-sample Attack-A/B and D2 CSVs into
mean +/- std tables and a single 'Figure 1 (multi-sample)' plot.

Usage (on the Fir login node, after all Tier-1 jobs finished):
    python tools/tier1_aggregate.py \
        --samples 4729,500,1000,2000,5000,10000 \
        --note_a phase2_attackA_ms_s \
        --note_b phase2_attackB_ms_s \
        --semantic l_eye \
        --out_dir tools/tier1_figures

Assumes Attack-B sweep CSV name `attackB-linf-40steps-sweep.csv`
under src/runs/CelebA_HQ_HF-CelebA_HQ_mask-<note_b><idx>/results/sample_idx<idx>/ .
Symmetric for Attack A / D2. Missing samples are skipped with a warning.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_RUNS = "src/runs"
ATTACK_A_CSV_GLOB = "attackA-linf-*steps-sweep.csv"
ATTACK_B_CSV_GLOB = "attackB-linf-*steps-sweep.csv"
D2_CSV = "defense_D2.csv"


def load_sample_csv(note_prefix: str, idx: str, csv_glob: str) -> pd.DataFrame | None:
    run = os.path.join(
        REPO_RUNS,
        f"CelebA_HQ_HF-CelebA_HQ_mask-{note_prefix}{idx}",
        "results",
        f"sample_idx{idx}",
    )
    hits = sorted(glob.glob(os.path.join(run, csv_glob)))
    if not hits:
        print(f"  [warn] sample={idx} note={note_prefix} glob='{csv_glob}' -> no CSV found", file=sys.stderr)
        return None
    df = pd.read_csv(hits[0])
    df["sample_idx"] = idx
    return df


def aggregate_attack(samples: list[str], note_prefix: str, csv_glob: str, label: str) -> pd.DataFrame:
    dfs = []
    for idx in samples:
        d = load_sample_csv(note_prefix, idx, csv_glob)
        if d is not None:
            dfs.append(d)
    if not dfs:
        raise SystemExit(f"[{label}] no per-sample CSVs found.")
    raw = pd.concat(dfs, ignore_index=True)
    eps_col = "eps" if "eps" in raw.columns else ("eps_img" if "eps_img" in raw.columns else raw.columns[0])
    value_col = "misalign"
    if value_col not in raw.columns:
        # Handle older CSV schemas
        for alt in ("misalignment", "direction_misalign", "misalign_clean"):
            if alt in raw.columns:
                value_col = alt
                break
    print(f"[{label}] epsilon col = '{eps_col}', value col = '{value_col}', N samples = {raw['sample_idx'].nunique()}")
    agg = raw.groupby(eps_col)[value_col].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["eps", "mean", "std", "n"]
    return agg


def aggregate_d2(samples: list[str], note_b_prefix: str) -> pd.DataFrame:
    dfs = []
    for idx in samples:
        run = os.path.join(
            REPO_RUNS,
            f"CelebA_HQ_HF-CelebA_HQ_mask-{note_b_prefix}{idx}",
            "results",
            f"sample_idx{idx}",
            D2_CSV,
        )
        if os.path.exists(run):
            d = pd.read_csv(run)
            d["sample_idx"] = idx
            dfs.append(d)
        else:
            print(f"  [warn] missing D2 CSV for sample={idx}: {run}", file=sys.stderr)
    if not dfs:
        raise SystemExit("[D2] no defense_D2.csv files found.")
    raw = pd.concat(dfs, ignore_index=True)
    method_col = "purify_method" if "purify_method" in raw.columns else "method"
    param_col = "purify_param" if "purify_param" in raw.columns else "param"
    eps_col = "eps_img" if "eps_img" in raw.columns else "eps"
    agg = raw.groupby([method_col, param_col, eps_col]).agg(
        misalign_mean=("misalign_defended", "mean"),
        misalign_std=("misalign_defended", "std"),
        reduction_mean=("defense_reduction_pct", "mean"),
        reduction_std=("defense_reduction_pct", "std"),
        n=("sample_idx", "nunique"),
    ).reset_index()
    agg.columns = ["method", "param", "eps", "mis_mean", "mis_std", "red_mean", "red_std", "n"]
    return agg


def plot_multisample_fig1(agg_a: pd.DataFrame, agg_b: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(agg_a["eps"], agg_a["mean"], yerr=agg_a["std"],
                fmt="-o", color="tab:blue", capsize=3,
                label=f"Attack A  (latent eps_xt)   N={int(agg_a['n'].min())}")
    ax.errorbar(agg_b["eps"], agg_b["mean"], yerr=agg_b["std"],
                fmt="--s", color="tab:red", capsize=3,
                label=f"Attack B  (image eps_img)   N={int(agg_b['n'].min())}")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.text(agg_b["eps"].min() * 1.1, 0.52, "45 deg rotation threshold",
            fontsize=8, color="gray")
    ax.axhspan(0.75, 1.0, alpha=0.08, color="red")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\varepsilon$  (L$_\infty$, native space of each attack; log)")
    ax.set_ylabel(r"direction misalignment  $1 - |\langle v_{\mathrm{clean}}, v_{\mathrm{adv}}\rangle|$")
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("LOCO direction misalignment across samples (mean +/- std)")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[fig1-multisample] saved -> {out_path}")


def plot_multisample_d2(agg_d2: pd.DataFrame, out_path: str, methods=("bits", "jpeg", "blur")) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    color = {"bits": "tab:blue", "jpeg": "tab:red", "blur": "tab:green"}
    marker = {"bits": "s", "jpeg": "o", "blur": "^"}
    for method in methods:
        sub = agg_d2[agg_d2["method"].astype(str) == method]
        if sub.empty:
            continue
        for param, grp in sub.groupby("param"):
            grp = grp.sort_values("eps")
            ax.errorbar(grp["eps"], grp["mis_mean"], yerr=grp["mis_std"],
                        fmt=f"-{marker[method]}", color=color[method],
                        alpha=0.8, capsize=2,
                        label=f"{method}:{param}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\varepsilon_{\mathrm{img}}$ (L$_\infty$)")
    ax.set_ylabel("defended misalignment  (mean +/- std)")
    ax.set_title("D2 purification across samples")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_ylim(-0.02, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[d2-multisample] saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True,
                        help="comma-separated sample indices, e.g. 4729,500,1000,2000,5000,10000")
    parser.add_argument("--note_a", default="phase2_attackA_ms_s",
                        help="NOTE prefix for Attack A runs (followed by sample_idx)")
    parser.add_argument("--note_b", default="phase2_attackB_ms_s",
                        help="NOTE prefix for Attack B runs (followed by sample_idx)")
    parser.add_argument("--semantic", default="l_eye")
    parser.add_argument("--out_dir", default="tools/tier1_figures")
    args = parser.parse_args()

    samples = [s.strip() for s in args.samples.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"==> Tier-1 aggregation over samples: {samples}")
    agg_a = aggregate_attack(samples, args.note_a, ATTACK_A_CSV_GLOB, label="Attack A")
    agg_b = aggregate_attack(samples, args.note_b, ATTACK_B_CSV_GLOB, label="Attack B")
    agg_a.to_csv(os.path.join(args.out_dir, "attackA_mean_std.csv"), index=False)
    agg_b.to_csv(os.path.join(args.out_dir, "attackB_mean_std.csv"), index=False)
    plot_multisample_fig1(agg_a, agg_b,
                          os.path.join(args.out_dir, "fig1_misalignment_multisample.png"))

    try:
        agg_d2 = aggregate_d2(samples, args.note_b)
        agg_d2.to_csv(os.path.join(args.out_dir, "defense_D2_mean_std.csv"), index=False)
        plot_multisample_d2(agg_d2, os.path.join(args.out_dir, "fig8_defense_D2_multisample.png"))
    except SystemExit as e:
        print(f"[D2] skipped: {e}")

    print("\nAttack A mean +/- std:")
    print(agg_a.to_string(index=False))
    print("\nAttack B mean +/- std:")
    print(agg_b.to_string(index=False))


if __name__ == "__main__":
    main()
