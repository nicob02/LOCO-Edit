"""Phase 2 - Locality / mask-leakage audit.

Phase 1's central claim was that null-space projection keeps the LOCO edit
localized inside the user-specified mask M (the "eye" edit changes eyes, not
hair). Phase 2 attacks rotate the editing direction; a compelling attack
should also *break locality*, i.e. the attacked edit bleeds outside M.

Metric (per attack run):

               || (x0_edit_adv - x0_clean)  *  (1 - M) ||_2
    leakage = ---------------------------------------------------------
               || (x0_edit_adv - x0_clean)  *  (1 - M) ||_2
             + || (x0_edit_adv - x0_clean)  *       M  ||_2

where x0_clean is the original image and x0_edit_adv is the decoded x_0 after
applying the attacked direction v_adv with the same x-space guidance chain
LOCO uses (num_step iterations from xt_adv). A clean baseline is also
reported, using v_clean at xt_clean (same iteration count).

This tool auto-discovers Attack A and/or Attack B result .pt files under a
given sweep directory and emits one CSV + a small side-by-side PNG per run.

Usage (single-GPU SLURM job):
    python src/tools/phase2_locality.py \
        --attack_type B \
        --config base_eye_null \
        --note phase2_attackB_eye_sweep \
        --sample_idx 4729 --choose_sem eye \
        --sweep_dir src/runs/<exp>/results/sample_idx4729
"""

from __future__ import annotations

import gc
import glob
import json
import os
import sys

import torch
import torchvision.utils as tvu

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.define_argparser import parse_args, preset
from modules.edit import EditUncondDiffusion


def _ddim_decode_from_xt(edit, xt_start: torch.Tensor) -> torch.Tensor:
    """Decode x_t all the way to x_0 without any x-space guidance (the 'no edit'
    reconstruction). Needed as the locality baseline so DDIM round-trip noise
    cancels out of the leakage metric.
    """
    xt = xt_start.clone().to(device=edit.device, dtype=edit.dtype)
    out = edit.DDIMforwardsteps(
        xt, t_start_idx=edit.edit_t_idx, t_end_idx=-1,
        save_image=False, performance_boosting=True,
    )
    x0 = out[0] if isinstance(out, tuple) else out
    return x0.to(device=edit.device, dtype=edit.dtype)


def _render_edit_x0(edit, xt_start: torch.Tensor, v_dir: torch.Tensor) -> torch.Tensor:
    """Apply x-space guidance from `xt_start` along direction `v_dir`, decode to x_0.

    Mirrors exactly what run_edit_null_space_projection does for one direction
    at the maximum lambda (final iteration of the guidance chain). Decoding is
    via DDIMforwardsteps (performance_boosting=True -> no KV cache inflation).

    NOTE on return conventions: DDIMforwardsteps returns a 3-tuple
    (latents, t, t_idx) only when `t_end_idx` is hit mid-loop. When we denoise
    all the way to t=0 (t_end_idx=-1), the function falls through to the final
    `return xt` at the end of the method -> a single tensor. Handle both.
    """
    xt = xt_start.clone().to(device=edit.device, dtype=edit.dtype)
    vk = v_dir.view(-1, *xt.shape[1:]).to(device=edit.device, dtype=edit.dtype)
    for _ in range(edit.x_space_guidance_num_step):
        xt = edit.x_space_guidance_direct(
            xt, t_idx=edit.edit_t_idx, vk=vk,
            single_edit_step=edit.x_space_guidance_edit_step,
        )
    out = edit.DDIMforwardsteps(
        xt, t_start_idx=edit.edit_t_idx, t_end_idx=-1,
        save_image=False, performance_boosting=True,
    )
    x0 = out[0] if isinstance(out, tuple) else out
    return x0.to(device=edit.device, dtype=edit.dtype)


def _mask_to_image_shape(mask: torch.Tensor, shape) -> torch.Tensor:
    """Take a [*,1,H,W] mask in {0,1} or [0,1] and broadcast to `shape` (C-channel)."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    m = (mask > 0.5).float()
    if m.shape[1] == 1 and shape[1] > 1:
        m = m.expand(-1, shape[1], -1, -1)
    return m


def _leakage(diff: torch.Tensor, mask_bin: torch.Tensor) -> dict:
    """Leakage of `diff` w.r.t. the mask (1 = inside ROI, 0 = outside).

    We report three metrics:
      * `leakage`            : outside_L2 / (inside_L2 + outside_L2)
                               -- the traditional metric, but biased by mask area
                               (if the mask covers <<50% of pixels, this metric is
                               pinned toward 1.0 even for a perfectly local edit).
      * `leakage_area_norm`  : outside_RMS / (inside_RMS + outside_RMS), where
                               RMS = L2 / sqrt(#elements). Area-normalized; the
                               direct locality analogue, robust to mask size.
      * `intensity_ratio`    : outside_mean_abs / inside_mean_abs. Unbounded.
                               <1 -> edit more intense inside mask (locality holds).
                               =1 -> uniform.  >1 -> locality broken.
    """
    d = diff.abs()
    m = mask_bin
    inv_m = 1.0 - mask_bin

    inside_L2 = (diff * m).flatten().norm().item()
    outside_L2 = (diff * inv_m).flatten().norm().item()
    total_L2 = inside_L2 + outside_L2

    inside_n = m.sum().clamp_min(1).item()
    outside_n = inv_m.sum().clamp_min(1).item()

    inside_mean_abs = (d * m).sum().item() / inside_n
    outside_mean_abs = (d * inv_m).sum().item() / outside_n
    inside_rms = inside_L2 / (inside_n ** 0.5)
    outside_rms = outside_L2 / (outside_n ** 0.5)

    return {
        "leakage":           outside_L2 / max(total_L2, 1e-8),
        "leakage_area_norm": outside_rms / max(inside_rms + outside_rms, 1e-12),
        "intensity_ratio":   outside_mean_abs / max(inside_mean_abs, 1e-12),
        "norm_inside":       inside_L2,
        "norm_outside":      outside_L2,
        "mean_abs_inside":   inside_mean_abs,
        "mean_abs_outside":  outside_mean_abs,
    }


def _extract_eps_tag(dirname: str) -> str:
    """Parse 'attackA-linf-eps0.02-40steps' -> '0.02' (also works for eps_img)."""
    for tok in dirname.split("-"):
        if tok.startswith("eps_img"):
            return tok[len("eps_img"):]
        if tok.startswith("eps"):
            return tok[len("eps"):]
    return "?"


def _load_clean_xt(edit, sample_idx: int) -> torch.Tensor:
    """Recompute x_t_clean by running the standard inversion+forward on the sample.

    This is the same path Phase 1 and both attacks use to get x_t. We do NOT
    trust saved xt_clean copies (shape/dtype may differ across runs); we do
    it fresh so x_t is deterministically consistent with the edit pipeline.
    """
    xT = edit.run_DDIMinversion(idx=sample_idx)
    xt, t, t_idx = edit.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=edit.edit_t_idx)
    assert t_idx == edit.edit_t_idx
    return xt.to(device=edit.device, dtype=edit.dtype), t


def main():
    args = parse_args()
    args = preset(args)

    assert args.attack_type in ("A", "B"), "pass --attack_type A or B"
    edit = EditUncondDiffusion(args)

    # Resolve sweep_dir -- either the user gave one, or we default to the
    # Phase 2 run's results/sample_idx<sidx> folder.
    sweep_dir = args.sweep_dir or os.path.join(
        edit.result_folder, f"sample_idx{args.sample_idx}",
    )
    if not os.path.isdir(sweep_dir):
        sweep_dir = edit.result_folder
    if args.attack_type == "A":
        result_paths = sorted(glob.glob(os.path.join(sweep_dir, "attackA-*/attackA_result.pt"))) \
                    + sorted(glob.glob(os.path.join(sweep_dir, "attackA-*/attack_result.pt")))
    else:
        result_paths = sorted(glob.glob(os.path.join(sweep_dir, "attackB-*/attackB_result.pt")))
    if not result_paths:
        raise SystemExit(
            f"[locality] no attack_{args.attack_type} result .pt files found under {sweep_dir}\n"
            "  Did you point --sweep_dir at the right place?"
        )
    print(f"[locality] found {len(result_paths)} attack runs under {sweep_dir}")

    mask = edit.dataset.getmask(idx=args.sample_idx, choose_sem=args.choose_sem)
    if torch.is_tensor(mask):
        mask = mask.to(device=edit.device)
    xt_clean, t = _load_clean_xt(edit, args.sample_idx)

    rows = []
    # ---- Baseline 1: "no edit" DDIM reconstruction from xt_clean ---------------
    # This is the correct reference frame for measuring leakage: compared to the
    # dataset image, the DDIM round-trip reconstruction adds globally-distributed
    # encode/decode noise that dominates the outside-mask L2. Using the no-edit
    # reconstruction cancels that noise to first order.
    print("[locality] computing DDIM no-edit reconstruction (reference) ...")
    with torch.no_grad():
        x0_recon_clean = _ddim_decode_from_xt(edit, xt_clean)

    # ---- Baseline 2: clean LOCO edit (v_clean applied at xt_clean) -------------
    print("[locality] computing clean-baseline edit (reused for all eps) ...")
    ref = torch.load(result_paths[0], map_location="cpu")
    v_clean = ref["v_clean"].to(device=edit.device, dtype=edit.dtype)
    with torch.no_grad():
        x0_edit_clean = _render_edit_x0(edit, xt_clean, v_clean)
    diff_clean = (x0_edit_clean - x0_recon_clean).detach()
    mask_bin = _mask_to_image_shape(mask, diff_clean.shape).to(diff_clean.device)
    leak_clean = _leakage(diff_clean, mask_bin)
    print(f"[locality] clean baseline (area-normalised): "
          f"leakage_area={leak_clean['leakage_area_norm']:.3f}, "
          f"intensity_ratio(out/in)={leak_clean['intensity_ratio']:.2f}")
    print(f"[locality] clean baseline (raw L2, area-biased): "
          f"leakage={leak_clean['leakage']:.3f} "
          f"(inside_L2={leak_clean['norm_inside']:.3f}, outside_L2={leak_clean['norm_outside']:.3f})")
    del ref
    gc.collect(); torch.cuda.empty_cache()

    for p in result_paths:
        run_dir = os.path.dirname(p)
        eps_tag = _extract_eps_tag(os.path.basename(run_dir))
        print(f"\n[locality] run {os.path.basename(run_dir)} (eps={eps_tag}) ...")

        blob = torch.load(p, map_location="cpu")
        v_adv = blob["v_adv"].to(device=edit.device, dtype=edit.dtype)
        if args.attack_type == "A":
            # x_adv = xt + delta ; xt_adv = x_adv.
            xt_adv = blob["x_adv"].to(device=edit.device, dtype=edit.dtype)
        else:
            xt_adv = blob["xt_adv"].to(device=edit.device, dtype=edit.dtype)

        # Full attack effect at x_0: run the adversarial direction from the
        # adversarial latent, compare against the clean no-edit reconstruction
        # (stable reference). For Attack B xt_adv is a function of x_0_adv, so
        # this also captures any non-local change induced by the image-space
        # perturbation itself, not just the rotated direction.
        with torch.no_grad():
            x0_edit_adv = _render_edit_x0(edit, xt_adv, v_adv)
        diff_adv = (x0_edit_adv - x0_recon_clean).detach()
        leak_adv = _leakage(diff_adv, mask_bin)

        # Also isolate the pure "direction effect": apply v_adv from xt_clean.
        # This tells us how much of the locality break is due to the rotated
        # direction alone, versus the latent-space perturbation of xt.
        with torch.no_grad():
            x0_edit_adv_dironly = _render_edit_x0(edit, xt_clean, v_adv)
        diff_adv_dironly = (x0_edit_adv_dironly - x0_recon_clean).detach()
        leak_adv_dironly = _leakage(diff_adv_dironly, mask_bin)

        # Side-by-side dump for visual QA (rescale diffs to [0,1] for saving).
        vis = torch.cat([
            (x0_recon_clean / 2 + 0.5).clamp(0, 1),
            (x0_edit_clean / 2 + 0.5).clamp(0, 1),
            (x0_edit_adv / 2 + 0.5).clamp(0, 1),
        ], dim=0)
        tvu.save_image(vis, os.path.join(run_dir, "locality_triptych.png"),
                       nrow=3)

        d_adv_img = diff_adv.abs().mean(dim=1, keepdim=True)
        d_adv_img = (d_adv_img - d_adv_img.min()) / (d_adv_img.max() - d_adv_img.min() + 1e-8)
        tvu.save_image(d_adv_img, os.path.join(run_dir, "locality_diff_adv.png"))

        row = {
            "run":                       os.path.basename(run_dir),
            "eps":                       eps_tag,
            # area-normalised (preferred): use these in the paper
            "leakage_area_clean":        leak_clean["leakage_area_norm"],
            "leakage_area_adv":          leak_adv["leakage_area_norm"],
            "leakage_area_adv_dironly":  leak_adv_dironly["leakage_area_norm"],
            "intensity_ratio_clean":     leak_clean["intensity_ratio"],
            "intensity_ratio_adv":       leak_adv["intensity_ratio"],
            "intensity_ratio_adv_dironly": leak_adv_dironly["intensity_ratio"],
            "leakage_area_increase":     leak_adv["leakage_area_norm"] - leak_clean["leakage_area_norm"],
            "leakage_area_increase_dir": leak_adv_dironly["leakage_area_norm"] - leak_clean["leakage_area_norm"],
            # raw L2 (area-biased, kept for backward compat)
            "leakage_clean":             leak_clean["leakage"],
            "leakage_adv":               leak_adv["leakage"],
            "leakage_adv_dironly":       leak_adv_dironly["leakage"],
            "norm_inside_clean":         leak_clean["norm_inside"],
            "norm_outside_clean":        leak_clean["norm_outside"],
            "norm_inside_adv":           leak_adv["norm_inside"],
            "norm_outside_adv":          leak_adv["norm_outside"],
            "mean_abs_inside_clean":     leak_clean["mean_abs_inside"],
            "mean_abs_outside_clean":    leak_clean["mean_abs_outside"],
            "mean_abs_inside_adv":       leak_adv["mean_abs_inside"],
            "mean_abs_outside_adv":      leak_adv["mean_abs_outside"],
            "leakage_increase":          leak_adv["leakage"] - leak_clean["leakage"],
            "leakage_increase_dir":      leak_adv_dironly["leakage"] - leak_clean["leakage"],
        }
        rows.append(row)
        with open(os.path.join(run_dir, "locality_summary.json"), "w") as f:
            json.dump(row, f, indent=2)
        print(f"[locality]   area-norm leak: clean={leak_clean['leakage_area_norm']:.3f}  "
              f"adv={leak_adv['leakage_area_norm']:.3f}  "
              f"(Δ full = {row['leakage_area_increase']:+.3f}, "
              f"Δ dir-only = {row['leakage_area_increase_dir']:+.3f}) | "
              f"intensity_ratio: clean={leak_clean['intensity_ratio']:.2f} "
              f"adv={leak_adv['intensity_ratio']:.2f}")

    csv_path = os.path.join(sweep_dir,
                            f"locality_attack{args.attack_type}.csv")
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[locality] summary -> {csv_path}")


if __name__ == "__main__":
    main()
