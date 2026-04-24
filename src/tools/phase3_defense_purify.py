"""Phase 3 - Defense D2: input purification against Attack B.

Motivation
----------
Attack B's delta_img is high-frequency and carefully aligned with the
direction of the gradient of cos(J(x_t).v_clean, J(x_t+delta_xt).v_clean)
through the DDIM inversion. Any lossy preprocessing of x_0 that destroys
high-frequency structure should kill delta_img without hurting the image's
semantic content much. Standard choices from the adversarial-robustness
literature:

  - JPEG re-encoding at quality q   (destroys high-frequency adversarial noise)
  - Gaussian blur with kernel k     (low-pass filter, same idea)
  - Bit-depth reduction (quantization) to k bits per channel

We evaluate each purification on the attacker's x_0_adv and report the
defended LOCO-direction misalignment after the full DDIM pipeline runs on
purify(x_0_adv).

What this script reports (per purify method, per epsilon)
---------------------------------------------------------
  cos_true_undefended   : Phase 2 direction misalignment at x_0_adv
  cos_true_defended     : direction misalignment at purify(x_0_adv)
  psnr_purify_clean     : PSNR( purify(x_0_clean), x_0_clean ) -- how much
                           signal the defense itself destroys
  defense_gain          : 1-cos_true_undef - 1-cos_true_def

A good defense has large gain AND small clean-side PSNR degradation.
"""

from __future__ import annotations

import gc
import glob
import io
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
from tools.phase2_attack_b import (
    _invert_from_tensor,
    _project_to_v_clean_subspace,
    verify_misalignment_b,
)


# ---------------------------------------------------------------------------
# purification primitives
# ---------------------------------------------------------------------------

def _jpeg_purify(x0_adv: torch.Tensor, quality: int) -> torch.Tensor:
    """x_0 in [-1, 1] -> JPEG encode/decode at `quality` -> back to [-1, 1]."""
    from PIL import Image
    # Note: we need to move to CPU + uint8 for PIL. Shape [1, 3, H, W].
    x = (x0_adv.detach() / 2 + 0.5).clamp(0, 1) * 255.0
    x_u8 = x[0].permute(1, 2, 0).round().to(torch.uint8).cpu().numpy()
    img = Image.fromarray(x_u8)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    img_dec = Image.open(buf).convert("RGB")
    import numpy as np
    arr = torch.from_numpy(np.asarray(img_dec)).permute(2, 0, 1).float() / 255.0
    arr = arr.unsqueeze(0).to(device=x0_adv.device, dtype=x0_adv.dtype)
    return (arr * 2.0 - 1.0)


def _gaussian_blur(x0_adv: torch.Tensor, sigma: float, ksize: int = 7) -> torch.Tensor:
    """Separable 2D Gaussian blur via depthwise conv."""
    import torch.nn.functional as F
    C = x0_adv.shape[1]
    half = ksize // 2
    coords = torch.arange(ksize, device=x0_adv.device, dtype=x0_adv.dtype) - half
    k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k1d = k1d / k1d.sum()
    k2d = k1d[:, None] * k1d[None, :]
    weight = k2d.view(1, 1, ksize, ksize).expand(C, 1, ksize, ksize).contiguous()
    pad = (half, half, half, half)
    return F.conv2d(F.pad(x0_adv, pad, mode="reflect"), weight, groups=C)


def _bit_depth_reduce(x0_adv: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize x_0 in [-1, 1] to `bits` bits per channel."""
    levels = 2 ** int(bits) - 1
    x01 = (x0_adv / 2 + 0.5).clamp(0, 1)
    xq = (x01 * levels).round() / levels
    return xq * 2.0 - 1.0


def purify(x0_adv: torch.Tensor, method: str, param: float) -> torch.Tensor:
    if method == "jpeg":
        return _jpeg_purify(x0_adv, quality=int(param))
    if method == "blur":
        return _gaussian_blur(x0_adv, sigma=float(param))
    if method == "bits":
        return _bit_depth_reduce(x0_adv, bits=int(param))
    raise ValueError(f"unknown purify method: {method}")


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    if mse < 1e-12:
        return 99.0
    return float(10.0 * torch.log10(torch.tensor(4.0 / mse)).item())  # range [-1,1] -> peak 2


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args = preset(args)
    assert args.attack_type == "B", "D2 only makes sense against image-space attacks (B)."
    edit = EditUncondDiffusion(args)

    sweep_dir = args.sweep_dir or edit.result_folder
    result_paths = sorted(glob.glob(os.path.join(sweep_dir,
                                                  "attackB-*/attackB_result.pt")))
    if not result_paths:
        raise SystemExit(f"no attackB results under {sweep_dir}")

    # Reconstruct clean xt + v_clean + vT_null_top the same way Attack B does.
    xT = edit.run_DDIMinversion(idx=args.sample_idx)
    mask = edit.dataset.getmask(idx=args.sample_idx, choose_sem=args.choose_sem)
    if torch.is_tensor(mask):
        mask = mask.to(edit.device)
    xt_clean, t, _ = edit.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=edit.edit_t_idx)
    xt_clean = xt_clean.to(device=edit.device, dtype=edit.dtype)
    if torch.is_tensor(t):
        t = t.to(edit.device)

    first = torch.load(result_paths[0], map_location="cpu")
    v_clean = first["v_clean"].to(device=edit.device, dtype=edit.dtype)
    x0_clean = first["x0_clean"].to(device=edit.device, dtype=edit.dtype)
    if x0_clean.ndim == 3:
        x0_clean = x0_clean.unsqueeze(0)

    # We need vT_null_top for verify_misalignment_b. Load from the basis file.
    # The basis is produced by a Phase 1 run, so it might live under a
    # different run folder (e.g. phase1_with_null) than the current --note.
    # We therefore try several candidate paths in order, then fall back to a
    # repo-wide glob across all CelebA_HQ_HF runs for the same sample_idx.
    basis_rel = os.path.join(
        "basis", f"local_basis-{edit.edit_t}T-select-mask-{args.choose_sem}",
    )
    vT_null_file = f"vT-null-{args.pca_rank_null}.pt"
    candidates = [
        os.path.join(sweep_dir, "..", basis_rel, vT_null_file),
        os.path.join(edit.result_folder, basis_rel, vT_null_file),
        os.path.join(edit.result_folder, f"sample_idx{args.sample_idx}", basis_rel, vT_null_file),
    ]
    import glob as _glob
    # Repo-wide fallback: any phase1 / phase2 / ... run that happened to
    # compute the same-sample, same-mask, same-edit_t basis.
    runs_root = os.path.join(os.path.dirname(__file__), "..", "runs")
    candidates += _glob.glob(os.path.join(
        runs_root, "CelebA_HQ_HF-CelebA_HQ_mask-*", "results",
        f"sample_idx{args.sample_idx}", basis_rel, vT_null_file,
    ))

    vT_null_top = None
    for c in candidates:
        c = os.path.abspath(c) if c else ""
        if c and os.path.isfile(c):
            vT_null_top = torch.load(c, map_location=edit.device).to(edit.dtype)[: args.pca_rank_null]
            print(f"[defense-D2] loaded vT_null_top from {c}")
            break
    if vT_null_top is None:
        raise SystemExit(
            f"[defense-D2] cannot find {vT_null_file} under any of:\n  "
            + "\n  ".join(candidates)
            + "\nRun Phase 1 for the same sample_idx, choose_sem and edit_t first."
        )

    # parse purify plan: e.g. "jpeg:75,90 blur:0.5,1.0 bits:4,6"
    plans = []
    for tok in args.purify_plan.split():
        method, pvals = tok.split(":")
        for v in pvals.split(","):
            plans.append((method, float(v)))
    print(f"[defense-D2] purification plan = {plans}")

    rows = []
    for p in result_paths:
        run_dir = os.path.dirname(p)
        blob = torch.load(p, map_location="cpu")
        cos_true_undef = float(blob["cos_true"])
        x0_adv = blob["x0_adv"].to(device=edit.device, dtype=edit.dtype)

        for method, param in plans:
            t0 = time.time()
            # Purify the attacker's image
            x0_purified = purify(x0_adv, method, param)
            x0_purified_clean = purify(x0_clean, method, param)
            psnr_clean = _psnr(x0_purified_clean, x0_clean)
            psnr_adv_vs_clean = _psnr(x0_purified, x0_clean)

            # Run the full DDIM inv+fwd on the purified image
            print(f"\n[defense-D2] {os.path.basename(run_dir)}  "
                  f"purify={method}({param})  PSNR_clean={psnr_clean:.2f} dB")
            xt_def, v_adv_def, cos_true_def = verify_misalignment_b(
                edit, x0_purified, v_clean, vT_null_top, mask,
                pca_rank=args.pca_rank,
            )
            wall = time.time() - t0
            row = {
                "run":                   os.path.basename(run_dir),
                "purify_method":         method,
                "purify_param":          param,
                "cos_true_undefended":   cos_true_undef,
                "misalign_undefended":   1.0 - cos_true_undef,
                "cos_true_defended":     cos_true_def,
                "misalign_defended":     1.0 - cos_true_def,
                "defense_gain":          (1.0 - cos_true_undef) - (1.0 - cos_true_def),
                "psnr_purify_clean":     psnr_clean,
                "psnr_adv_vs_clean":     psnr_adv_vs_clean,
                "wall_seconds":          wall,
            }
            rows.append(row)
            del xt_def, v_adv_def
            gc.collect(); torch.cuda.empty_cache()
            print(f"[defense-D2]   misalign_undef={1-cos_true_undef:.3f}  "
                  f"misalign_def={1-cos_true_def:.3f}  "
                  f"gain={row['defense_gain']:+.3f}  (wall {wall:.1f}s)")

    out_csv = os.path.join(sweep_dir, "defense_D2.csv")
    import csv as _csv
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[defense-D2] summary -> {out_csv}")


if __name__ == "__main__":
    main()
