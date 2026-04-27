"""Phase 2 - Attack B transfer: render edit strips for each target sample.

Companion to ``phase2_transfer.py``. While ``phase2_transfer`` records the
target-side misalignment number for each (source -> target), this script
produces the actual edit strip the reader sees, so the transfer story can
be told *visually* in the paper:

    x_0_tgt_adv = clamp(x_0_tgt + delta_img_src, [-1, 1])
    x_t_tgt_adv = DDIM_inv -> DDIM_fwd to edit_t_idx
    v_in_null   = GPM_modify(x_t_tgt_adv, mask_tgt) projected into nullspace
                  computed at x_t_tgt_adv (out-of-mask GPM)
    edit strip rendered at x_t_tgt_adv along v_in_null

Outputs (next to the source's ``attackB_result.pt``):
    transferB-src<src>-tgt<tgt>-<sem>-eps<eps>-<norm>-edit_strip.png    (one per target)
    transfer_strips_summary.csv                                         (target_idx, label, misalignment)

Run via ``scripts/nibi/phase2_transfer_strip.sh`` (positional args).
"""
from __future__ import annotations

import csv as _csv
import gc
import os
import sys
import time

import torch
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.define_argparser import parse_args, preset
from modules.edit import EditUncondDiffusion
from tools.phase2_attack_b import (
    _invert_from_tensor,
    _project_to_v_clean_subspace,
)


def _render_strip_at(edit, xt, v, *, args, label):
    """Render a 5-frame edit strip at xt along v, saved to edit.result_folder."""
    xts = {1: None, -1: None}
    v = v.view(-1, *xt.shape[1:])
    for direction in [1, -1]:
        vk = direction * v
        chain = [xt.clone()]
        for _ in tqdm(range(edit.x_space_guidance_num_step),
                      desc=f"transfer edit ({'+' if direction == 1 else '-'})"):
            x_next = edit.x_space_guidance_direct(
                chain[-1], t_idx=edit.edit_t_idx, vk=vk,
                single_edit_step=edit.x_space_guidance_edit_step,
            )
            chain.append(x_next)
        xt_dir = torch.cat(chain, dim=0)
        if args.vis_num == 1:
            xt_dir = xt_dir[[0, -1], :]
        else:
            xt_dir = xt_dir[:: max(xt_dir.size(0) // args.vis_num, 1)]
        xts[direction] = xt_dir
    xt_full = torch.cat([xts[-1].flip(dims=[0])[:-1], xts[1]], dim=0)
    edit.EXP_NAME = label
    edit.DDIMforwardsteps(
        xt_full,
        t_start_idx=edit.edit_t_idx,
        t_end_idx=-1,
        performance_boosting=True,
    )
    print(f"[transfer-strip] saved strip with EXP_NAME={edit.EXP_NAME}")


def _gpm_at_xt(edit, xt, t, mask, *, pca_rank, pca_rank_null):
    _, _, vT_modify = edit.local_encoder_decoder_pullback_xt(
        x=xt, t=t, op="mid", block_idx=0,
        pca_rank=pca_rank, chunk_size=1,
        min_iter=10, max_iter=50,
        convergence_threshold=1e-4, mask=mask, noise=False,
    )
    vT_modify = vT_modify.detach()
    gc.collect(); torch.cuda.empty_cache()
    _, _, vT_null = edit.local_encoder_decoder_pullback_xt(
        x=xt, t=t, op="mid", block_idx=0,
        pca_rank=pca_rank_null, chunk_size=1,
        min_iter=10, max_iter=50,
        convergence_threshold=1e-4, mask=~mask, noise=False,
    )
    vT_null = vT_null.detach()
    gc.collect(); torch.cuda.empty_cache()
    v_in_null = _project_to_v_clean_subspace(vT_modify.to(edit.dtype),
                                             vT_null.to(edit.dtype))
    return v_in_null[0].view_as(xt), vT_modify, vT_null


def main():
    args = parse_args()
    args = preset(args)

    if not args.attack_b_result.strip():
        raise SystemExit("--attack_b_result <path/attackB_result.pt> is required.")
    if not args.transfer_targets.strip():
        raise SystemExit('--transfer_targets "1000,2000,..." is required.')

    tgt_ids = [int(x) for x in args.transfer_targets.split(",") if x.strip()]
    blob = torch.load(args.attack_b_result, map_location="cpu")

    edit = EditUncondDiffusion(args)
    delta_img = blob["delta_img"].to(device=edit.device, dtype=edit.dtype)
    src_idx = int(blob.get("args", {}).get("sample_idx", -1))
    b_args = blob.get("args", {}) or {}
    eps_tag = float(b_args.get("attack_b_eps_img", b_args.get("attack_eps", 0.0)))
    norm_tag = b_args.get("attack_norm", b_args.get("attack_b_norm", "linf"))
    print(f"[transfer-strip] source idx = {src_idx}, eps_img = {eps_tag:g}, "
          f"||delta_img||_inf = {delta_img.abs().max().item():.4f}")

    run_dir = os.path.dirname(os.path.abspath(args.attack_b_result))
    print(f"[transfer-strip] writing strips next to {run_dir}")

    # Render the *source self-attacked* strip first (reuses xt_adv / v_adv from
    # the blob), so the visual figure can stack source + targets in a single
    # call without depending on phase3_defense_strip.py having been run.
    if args.also_render_source:
        try:
            xt_adv_blob = blob["xt_adv"].to(device=edit.device, dtype=edit.dtype)
            v_adv_blob = blob["v_adv"].to(device=edit.device, dtype=edit.dtype)
            v_adv_blob = v_adv_blob.view_as(xt_adv_blob)
            label = (
                f"transferB-src{src_idx}-tgt{src_idx}-{args.choose_sem}"
                f"-eps{eps_tag:g}-{norm_tag}-edit_strip"
            )
            print(f"[transfer-strip] rendering source self-attack -> {label}.png")
            prev = edit.result_folder
            edit.result_folder = run_dir
            try:
                _render_strip_at(edit, xt_adv_blob, v_adv_blob,
                                 args=args, label=label)
            finally:
                edit.result_folder = prev
            del xt_adv_blob, v_adv_blob
            gc.collect(); torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[transfer-strip] WARN: source self render failed: {exc}")

    summary = []
    for tgt in tgt_ids:
        t0 = time.time()
        print(f"\n[transfer-strip] target = {tgt}")
        x0_tgt = edit.dataset[tgt].to(device=edit.device, dtype=edit.dtype)
        if x0_tgt.ndim == 3:
            x0_tgt = x0_tgt.unsqueeze(0)
        mask = edit.dataset.getmask(idx=tgt, choose_sem=args.choose_sem)
        if torch.is_tensor(mask):
            mask = mask.to(edit.device)

        x0_tgt_adv = (x0_tgt + delta_img).clamp(-1.0, 1.0)
        xT_tgt_adv = _invert_from_tensor(edit, x0_tgt_adv)
        xt_tgt_adv, t_tgt_adv, _ = edit.DDIMforwardsteps(
            xT_tgt_adv, t_start_idx=0, t_end_idx=edit.edit_t_idx, save_image=False,
        )
        xt_tgt_adv = xt_tgt_adv.to(device=edit.device, dtype=edit.dtype)
        if torch.is_tensor(t_tgt_adv):
            t_tgt_adv = t_tgt_adv.to(edit.device)

        v_tgt_adv, vT_modify, vT_null = _gpm_at_xt(
            edit, xt_tgt_adv, t_tgt_adv, mask,
            pca_rank=args.pca_rank, pca_rank_null=args.pca_rank_null,
        )

        # Optional: also recover the target's CLEAN basis to report the
        # target-side misalignment (matches phase2_transfer.py output).
        misalign_tgt = None
        if args.also_compute_misalign:
            xT_tgt = edit.run_DDIMinversion(idx=tgt)
            xt_tgt, t_tgt, _ = edit.DDIMforwardsteps(
                xT_tgt, t_start_idx=0, t_end_idx=edit.edit_t_idx, save_image=False,
            )
            xt_tgt = xt_tgt.to(device=edit.device, dtype=edit.dtype)
            if torch.is_tensor(t_tgt):
                t_tgt = t_tgt.to(edit.device)
            v_tgt_clean, _, _ = _gpm_at_xt(
                edit, xt_tgt, t_tgt, mask,
                pca_rank=args.pca_rank, pca_rank_null=args.pca_rank_null,
            )
            cos_true = abs(float(
                (v_tgt_clean.flatten() * v_tgt_adv.flatten()).sum().item()))
            misalign_tgt = 1.0 - cos_true
            del xt_tgt, v_tgt_clean
            gc.collect(); torch.cuda.empty_cache()

        label = (
            f"transferB-src{src_idx}-tgt{tgt}-{args.choose_sem}"
            f"-eps{eps_tag:g}-{norm_tag}-edit_strip"
        )
        prev = edit.result_folder
        edit.result_folder = run_dir
        try:
            _render_strip_at(edit, xt_tgt_adv, v_tgt_adv,
                             args=args, label=label)
        finally:
            edit.result_folder = prev

        wall = time.time() - t0
        print(f"[transfer-strip] target {tgt}: "
              f"misalign={misalign_tgt if misalign_tgt is not None else 'n/a'}  "
              f"wall={wall:.1f}s")
        summary.append({
            "target_idx":   tgt,
            "label":        label,
            "misalignment": misalign_tgt if misalign_tgt is not None else "",
            "wall_seconds": round(wall, 1),
        })
        del vT_modify, vT_null, v_tgt_adv, xt_tgt_adv
        gc.collect(); torch.cuda.empty_cache()

    out_csv = os.path.join(run_dir, "transfer_strips_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"\n[transfer-strip] summary -> {out_csv}")


if __name__ == "__main__":
    main()
