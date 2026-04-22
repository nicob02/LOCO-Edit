"""Phase 3 - Defense D1: randomized smoothing of the LOCO direction.

Motivation
----------
Attacks A/B exploit a sharp ridge in v(x_t) = top-1 right singular vector of
J(x_t): tiny rotations of x_t can flip v by >70 degrees (Phase 2 empirical).
Classical randomized smoothing (Cohen et al., 2019) says: take the expectation
of v under x_t + eta, eta ~ N(0, sigma^2 I), and the smoothed function f_bar
is Lipschitz in x_t with constant ~ 1 / (sigma * sqrt(2*pi)). So for any
attack with ||delta||_2 <= R, the rotation is bounded by R / sigma.

Algorithm (per inference)
-------------------------
  Inputs:  x_t, mask M, n_samples, sigma
  for i in 1..n_samples:
      eta_i ~ N(0, sigma^2 I)
      v_i   = GPM(x_t + eta_i, mask=M)             # top-1 direction
      if i > 1 and <v_1, v_i> < 0: v_i <- -v_i     # sign canonicalisation
  v_bar = mean_i(v_i);  v_smooth = v_bar / ||v_bar||_2

The sign step matters: singular vectors are only defined up to sign, so naive
averaging of n_samples independent GPMs can cancel. We pin the sign of each
v_i to align with v_1 (the first sampled direction) before averaging.

What this script reports
------------------------
For each cached attack result and a grid of (sigma, n_samples):
  - cos_clean_smooth_vs_clean  -- how much does smoothing change the clean dir
                                  (should stay high, ~0.95+, else we lost signal)
  - cos_smooth_vs_smooth        -- cos( v_smooth(x_t), v_smooth(x_t_adv) )
                                  the defended misalignment. Should be >>
                                  cos_true (i.e. defense recovers direction).
  - misalign_undefended         -- 1 - cos_true (reproduces Phase 2 number for ref)
  - misalign_defended           -- 1 - cos_smooth_vs_smooth

A good defense curve shows misalign_defended sitting well below
misalign_undefended at large epsilon, with the gap widening as sigma grows
(up to a saturation where smoothing also destroys clean signal).
"""

from __future__ import annotations

import gc
import glob
import json
import os
import sys
import time

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.define_argparser import parse_args, preset
from modules.edit import EditUncondDiffusion
from tools.phase2_attack_b import _project_to_v_clean_subspace


def _sign_canonical(v_ref: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Flip v if it is anti-aligned with v_ref (singular vectors are sign-ambiguous)."""
    if (v_ref.flatten() * v.flatten()).sum().item() < 0:
        return -v
    return v


def _top_direction(edit, x_t, t, mask, *, pca_rank: int, pca_rank_null: int,
                   vT_null_top: torch.Tensor | None):
    """Run a single GPM at x_t, project into null-complement, return unit v.

    vT_null_top is reused from the first clean call; recomputing null space
    on every smoothing sample would be prohibitively expensive and also
    somewhat circular -- the null space defines the edit subspace, not the
    direction inside it.
    """
    _, _, vT_modify = edit.local_encoder_decoder_pullback_xt(
        x=x_t, t=t, op="mid", block_idx=0,
        pca_rank=pca_rank, chunk_size=1,
        min_iter=10, max_iter=50,
        convergence_threshold=1e-4, mask=mask, noise=False,
    )
    vT_modify = vT_modify.detach().to(edit.dtype)
    if vT_null_top is not None:
        vT = _project_to_v_clean_subspace(vT_modify, vT_null_top.to(edit.dtype))
    else:
        vT = vT_modify
    v = vT[0].view_as(x_t)
    v = v / v.flatten().norm().clamp(min=1e-12)
    gc.collect(); torch.cuda.empty_cache()
    return v


def _smoothed_direction(edit, x_t, t, mask, *, sigma: float, n_samples: int,
                         pca_rank: int, pca_rank_null: int,
                         vT_null_top: torch.Tensor | None):
    vs = []
    for i in range(n_samples):
        if sigma > 0:
            eta = torch.randn_like(x_t) * sigma
        else:
            eta = torch.zeros_like(x_t)
        v = _top_direction(edit, x_t + eta, t, mask,
                           pca_rank=pca_rank, pca_rank_null=pca_rank_null,
                           vT_null_top=vT_null_top)
        if len(vs) > 0:
            v = _sign_canonical(vs[0], v)
        vs.append(v)
    v_mean = torch.stack(vs, dim=0).mean(dim=0)
    v_smooth = v_mean / v_mean.flatten().norm().clamp(min=1e-12)
    return v_smooth


def main():
    args = parse_args()
    args = preset(args)
    assert args.attack_type in ("A", "B")

    edit = EditUncondDiffusion(args)

    sweep_dir = args.sweep_dir or edit.result_folder
    pat = "attackA-*/attackA_result.pt" if args.attack_type == "A" \
          else "attackB-*/attackB_result.pt"
    result_paths = sorted(glob.glob(os.path.join(sweep_dir, pat)))
    if not result_paths:
        raise SystemExit(f"no attack_{args.attack_type} results under {sweep_dir}")

    # Reconstruct clean x_t + v_clean + vT_null_top once (shared across eps).
    xT = edit.run_DDIMinversion(idx=args.sample_idx)
    mask = edit.dataset.getmask(idx=args.sample_idx, choose_sem=args.choose_sem)
    xt_clean, t, _ = edit.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=edit.edit_t_idx)
    xt_clean = xt_clean.to(device=edit.device, dtype=edit.dtype)
    if torch.is_tensor(t):     t = t.to(edit.device)
    if torch.is_tensor(mask):  mask = mask.to(edit.device)

    # Pull vT_null_top from the first attack result (they all share it).
    first = torch.load(result_paths[0], map_location="cpu")
    v_clean = first["v_clean"].to(device=edit.device, dtype=edit.dtype)
    # We don't have vT_null_top serialized; reload basis file from disk.
    basis_rel = os.path.join(
        "basis", f"local_basis-{edit.edit_t}T-select-mask-{args.choose_sem}",
    )
    # Try own run_folder first, fall back to attack run's folder.
    candidates = [
        os.path.join(edit.result_folder, basis_rel,
                     f"vT-null-{args.pca_rank_null}.pt"),
        os.path.join(sweep_dir, "..", basis_rel,
                     f"vT-null-{args.pca_rank_null}.pt"),
    ]
    vT_null_top = None
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isfile(c):
            vT_null_top = torch.load(c, map_location=edit.device).to(edit.dtype)[: args.pca_rank_null]
            print(f"[defense-D1] loaded vT_null_top from {c}")
            break
    if vT_null_top is None:
        print("[defense-D1] WARN: no vT_null_top found -- skipping null-space projection "
              "(numbers will be slightly larger but still comparable).")

    sigmas     = [float(x) for x in args.defense_sigmas.split(",") if x.strip()]
    n_sample_list = [int(x) for x in args.defense_n_samples.split(",") if x.strip()]

    rows = []
    for p in result_paths:
        run_dir = os.path.dirname(p)
        blob = torch.load(p, map_location="cpu")
        cos_true = float(blob["cos_true"])
        if args.attack_type == "A":
            xt_adv = blob["x_adv"].to(device=edit.device, dtype=edit.dtype)
        else:
            xt_adv = blob["xt_adv"].to(device=edit.device, dtype=edit.dtype)

        for sigma in sigmas:
            for n in n_sample_list:
                t0 = time.time()
                print(f"\n[defense-D1] {os.path.basename(run_dir)}  "
                      f"sigma={sigma:g}  n_samples={n}")
                torch.manual_seed(args.seed)  # reproducible smoothing
                v_clean_smooth = _smoothed_direction(
                    edit, xt_clean, t, mask,
                    sigma=sigma, n_samples=n,
                    pca_rank=args.pca_rank, pca_rank_null=args.pca_rank_null,
                    vT_null_top=vT_null_top,
                )
                torch.manual_seed(args.seed + 1)
                v_adv_smooth = _smoothed_direction(
                    edit, xt_adv, t, mask,
                    sigma=sigma, n_samples=n,
                    pca_rank=args.pca_rank, pca_rank_null=args.pca_rank_null,
                    vT_null_top=vT_null_top,
                )
                cos_smooth_vs_smooth = abs(float(
                    (v_clean_smooth.flatten() * v_adv_smooth.flatten()).sum().item()
                ))
                cos_clean_preserved = abs(float(
                    (v_clean.flatten() * v_clean_smooth.flatten()).sum().item()
                ))
                wall = time.time() - t0
                row = {
                    "run":                      os.path.basename(run_dir),
                    "sigma":                    sigma,
                    "n_samples":                n,
                    "cos_true_undefended":      cos_true,
                    "misalign_undefended":      1.0 - cos_true,
                    "cos_smooth_vs_smooth":     cos_smooth_vs_smooth,
                    "misalign_defended":        1.0 - cos_smooth_vs_smooth,
                    "cos_clean_preserved":      cos_clean_preserved,
                    "defense_gain":             (1.0 - cos_true) - (1.0 - cos_smooth_vs_smooth),
                    "wall_seconds":             wall,
                }
                rows.append(row)
                print(f"[defense-D1]   misalign_undef={1-cos_true:.3f}  "
                      f"misalign_def={1-cos_smooth_vs_smooth:.3f}  "
                      f"clean_preserved={cos_clean_preserved:.3f}  "
                      f"(wall {wall:.1f}s)")

    out_csv = os.path.join(sweep_dir,
                           f"defense_D1_attack{args.attack_type}.csv")
    import csv as _csv
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[defense-D1] summary -> {out_csv}")


if __name__ == "__main__":
    main()
