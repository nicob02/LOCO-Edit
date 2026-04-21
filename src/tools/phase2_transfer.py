"""Phase 2 - Attack B transfer audit.

Does the adversarial image perturbation delta_img_adv, optimised on the source
sample, also break the LOCO direction of *other* (unseen) samples?

Threat-model reading:
- YES -> universal / sample-agnostic attack ('one-size-fits-all' perturbation).
  This is the strongest attacker claim: a single tiny pattern sabotages
  any LOCO edit regardless of the victim image.
- PARTIAL -> the attack carries some transfer, bounded by image statistics.
- NO -> attacker must re-optimise per sample; threat is narrower but still
  realistic for white-box attackers who know the target image in advance.

Transferring Attack A (latent-space delta on x_t) across samples is
ill-defined because each sample has its own x_t; we only transfer Attack B.

Pipeline per target sample:
    x_0_tgt_adv  = clamp(x_0_tgt + delta_img_adv,  [-1, 1])
    x_T_tgt_adv  = DDIM_inverse(x_0_tgt_adv)
    x_t_tgt_adv  = DDIM_forward(x_T_tgt_adv, edit_t_idx)
    v_clean_tgt  = GPM(x_t_tgt,   mask_tgt)   # fresh basis for this sample
    v_adv_tgt    = GPM(x_t_tgt_adv, mask_tgt)
    cos_tgt      = |<v_clean_tgt, v_adv_tgt>|
    misalign_tgt = 1 - cos_tgt
"""

from __future__ import annotations

import gc
import json
import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.define_argparser import parse_args, preset
from modules.edit import EditUncondDiffusion

# reuse the verified pipeline helpers from Attack B
from tools.phase2_attack_b import (
    verify_misalignment_b,  # not used here but useful reference
    _invert_from_tensor,
    _project_to_v_clean_subspace,
)


def _gpm_modify_and_null(edit, x_t, t, mask, *, pca_rank, pca_rank_null):
    """Compute the modify (in-mask) and null (out-of-mask) top directions at x_t.

    Same two calls LOCO Phase 1 makes, run sequentially with aggressive mem
    cleanup between them (otherwise the second GPM accumulates activation graph
    and OOMs on 80 GB H100s at 256x256 resolution).
    """
    _, _, vT_modify = edit.local_encoder_decoder_pullback_xt(
        x=x_t, t=t, op="mid", block_idx=0,
        pca_rank=pca_rank, chunk_size=1,
        min_iter=10, max_iter=50,
        convergence_threshold=1e-4, mask=mask, noise=False,
    )
    vT_modify = vT_modify.detach()
    gc.collect(); torch.cuda.empty_cache()

    _, _, vT_null = edit.local_encoder_decoder_pullback_xt(
        x=x_t, t=t, op="mid", block_idx=0,
        pca_rank=pca_rank_null, chunk_size=1,
        min_iter=10, max_iter=50,
        convergence_threshold=1e-4, mask=~mask, noise=False,
    )
    vT_null = vT_null.detach()
    gc.collect(); torch.cuda.empty_cache()

    v_in_null = _project_to_v_clean_subspace(vT_modify.to(edit.dtype),
                                             vT_null.to(edit.dtype))
    return v_in_null[0].view_as(x_t), vT_null


def _run_target(edit, tgt_idx: int, delta_img: torch.Tensor, *,
                pca_rank: int, pca_rank_null: int, choose_sem: str) -> dict:
    """Full transfer evaluation on one target sample. Returns a summary row."""
    print(f"\n[transfer] target sample_idx = {tgt_idx}")
    # Target's x_0, mask.
    x0_tgt = edit.dataset[tgt_idx].to(device=edit.device, dtype=edit.dtype)
    if x0_tgt.ndim == 3:
        x0_tgt = x0_tgt.unsqueeze(0)
    mask = edit.dataset.getmask(idx=tgt_idx, choose_sem=choose_sem)
    if torch.is_tensor(mask):
        mask = mask.to(device=edit.device)

    # Target's clean x_t via full DDIM inv+fwd.
    xT_tgt = edit.run_DDIMinversion(idx=tgt_idx)
    xt_tgt, t_tgt, _ = edit.DDIMforwardsteps(xT_tgt, t_start_idx=0,
                                             t_end_idx=edit.edit_t_idx)
    xt_tgt = xt_tgt.to(device=edit.device, dtype=edit.dtype)
    if torch.is_tensor(t_tgt):
        t_tgt = t_tgt.to(edit.device)

    # Clean direction at target.
    print(f"[transfer]   GPM on clean x_t_tgt ...")
    v_clean_tgt, vT_null_tgt = _gpm_modify_and_null(
        edit, xt_tgt, t_tgt, mask,
        pca_rank=pca_rank, pca_rank_null=pca_rank_null,
    )

    # Apply the SOURCE's delta_img to the target image, push through full DDIM.
    x0_tgt_adv = (x0_tgt + delta_img.to(device=edit.device, dtype=edit.dtype)
                  ).clamp(-1.0, 1.0)
    print(f"[transfer]   DDIM inv+fwd on x_0_tgt_adv ...")
    xT_tgt_adv = _invert_from_tensor(edit, x0_tgt_adv)
    xt_tgt_adv, t_tgt_adv, _ = edit.DDIMforwardsteps(
        xT_tgt_adv, t_start_idx=0, t_end_idx=edit.edit_t_idx, save_image=False,
    )
    xt_tgt_adv = xt_tgt_adv.to(device=edit.device, dtype=edit.dtype)
    if torch.is_tensor(t_tgt_adv):
        t_tgt_adv = t_tgt_adv.to(edit.device)

    # Adversarial direction at target, projected into the same null-complement.
    print(f"[transfer]   GPM on adv x_t_tgt_adv ...")
    _, _, vT_modify_adv = edit.local_encoder_decoder_pullback_xt(
        x=xt_tgt_adv, t=t_tgt_adv, op="mid", block_idx=0,
        pca_rank=pca_rank, chunk_size=1,
        min_iter=10, max_iter=50,
        convergence_threshold=1e-4, mask=mask, noise=False,
    )
    v_adv_tgt = _project_to_v_clean_subspace(vT_modify_adv.to(edit.dtype),
                                             vT_null_tgt.to(edit.dtype))[0].view_as(xt_tgt_adv)

    cos_tgt = abs(float((v_clean_tgt.flatten() * v_adv_tgt.flatten()).sum().item()))
    eps_xt_eff_inf = float((xt_tgt_adv - xt_tgt).abs().max().item())
    print(f"[transfer]   cos_true = {cos_tgt:.4f}  "
          f"misalign = {1 - cos_tgt:.4f}  "
          f"(||delta_xt||_inf on target = {eps_xt_eff_inf:.4f})")

    del vT_modify_adv, vT_null_tgt
    gc.collect(); torch.cuda.empty_cache()

    return {
        "target_idx":          tgt_idx,
        "cos_true":            cos_tgt,
        "misalignment":        1.0 - cos_tgt,
        "eps_xt_effective_inf": eps_xt_eff_inf,
    }


def main():
    args = parse_args()
    args = preset(args)

    if not args.attack_b_result:
        raise SystemExit("pass --attack_b_result <path to attackB_result.pt>")
    if not args.transfer_targets:
        raise SystemExit('pass --transfer_targets "1000,2000,3000"')

    tgt_ids = [int(x) for x in args.transfer_targets.split(",") if x.strip()]
    blob = torch.load(args.attack_b_result, map_location="cpu")
    delta_img = blob["delta_img"]
    src_idx = blob.get("args", {}).get("sample_idx", -1)
    eps_img = blob.get("args", {}).get("attack_b_eps_img", None)
    print(f"[transfer] loaded delta_img from {args.attack_b_result}")
    print(f"[transfer] source sample = {src_idx}  eps_img = {eps_img}  "
          f"shape = {tuple(delta_img.shape)}  "
          f"||delta_img||_inf = {delta_img.abs().max().item():.4f}")

    edit = EditUncondDiffusion(args)

    # Reference: cos_true on the source (should recover the attack's reported number).
    # We skip this unless --sample_idx == source_idx AND the user wants it;
    # running one more full transfer eval on the source is optional and cheap.

    rows = []
    for tgt in tgt_ids:
        row = _run_target(edit, tgt, delta_img,
                          pca_rank=args.pca_rank,
                          pca_rank_null=args.pca_rank_null,
                          choose_sem=args.choose_sem)
        row["source_idx"] = src_idx
        row["eps_img"]    = eps_img
        rows.append(row)

    # Emit CSV beside the source attack result.
    out_dir = os.path.dirname(args.attack_b_result)
    csv_path = os.path.join(out_dir, "transfer.csv")
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "transfer_summary.json"), "w") as f:
        json.dump({
            "source_idx":  src_idx,
            "eps_img":     eps_img,
            "delta_inf":   float(delta_img.abs().max().item()),
            "targets":     rows,
        }, f, indent=2)
    print(f"\n[transfer] summary -> {csv_path}")
    for r in rows:
        print(f"  target {r['target_idx']:>5d}: "
              f"misalign = {r['misalignment']:.4f}  "
              f"(eps_xt_eff = {r['eps_xt_effective_inf']:.4f})")


if __name__ == "__main__":
    main()
