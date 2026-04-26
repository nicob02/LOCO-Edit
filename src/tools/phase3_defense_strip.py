"""Phase 3 - Defense D2: render the *defended* LOCO edit strip.

Companion to ``phase3_defense_purify.py``. While ``phase3_defense_purify``
records the misalignment number per (purify_method, purify_param), this script
produces the actual edit strip the reader sees:

    purify(x_0_adv) -> DDIM inversion + forward -> x_t_def -> GPM -> v_def
    edit strip rendered at x_t_def along v_def

So we get a like-for-like 5-frame strip directly comparable to the clean
Phase-1 strip and the attacked Phase-2 strip.

Inputs (positional via argparse / preset)
----------------------------------------
The script must run inside the SAME --note family as the attack (so the
basis files resolve), and you point it at a single ``attackB_result.pt``
via --attack_b_result. ``--purify_plan`` selects which purifications to
render. By default we render the three plans we use in the report:

    bits:4   jpeg:75   blur:1.5

Outputs
-------
For each (method, param) we write a dedicated subfolder

    <run_dir>/defenseD2-<method>-<param>-edit_strip/
        defenseD2_strip.pt          (xt_full + metadata for repro)

and rely on edit.DDIMforwardsteps()'s built-in PNG saver to drop the
canonical strip image as

    <run_dir>/defenseD2-<method>-<param>-edit_strip-pc_0.png

so the file naming is grep-friendly for the eval-locality script.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch
import torchvision.utils as tvu
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.define_argparser import parse_args, preset
from modules.edit import EditUncondDiffusion
from tools.phase3_defense_purify import purify, _psnr
from tools.phase2_attack_b import (
    _invert_from_tensor,
    _project_to_v_clean_subspace,
)


def _render_strip_at(edit, xt_def, v_def, t, *, args, label):
    """Reuse Phase-1's x_space_guidance to render a 5-frame strip and save it.

    Mirrors phase2_attack_a._render_attacked_edit but starts from the
    DEFENDED latent and uses the DEFENDED direction.
    """
    xts = {1: None, -1: None}
    v = v_def.view(-1, *xt_def.shape[1:])
    for direction in [1, -1]:
        vk = direction * v
        chain = [xt_def.clone()]
        for _ in tqdm(range(edit.x_space_guidance_num_step),
                      desc=f"defended edit ({'+' if direction == 1 else '-'})"):
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
    print(f"[defense-strip] saved strip with EXP_NAME={edit.EXP_NAME}")


def main():
    args = parse_args()
    args = preset(args)
    assert args.attack_type == "B", "Defense D2 strip is only meaningful against B."

    # Where is the attacker's blob?
    if not args.attack_b_result.strip():
        raise SystemExit("--attack_b_result <path/attackB_result.pt> is required.")
    blob = torch.load(args.attack_b_result, map_location="cpu")

    edit = EditUncondDiffusion(args)

    # Reconstruct clean LOCO state for sample_idx (gives xt_clean, v_clean basis,
    # mask, t, vT_null_top -- the same way phase3_defense_purify does).
    xT = edit.run_DDIMinversion(idx=args.sample_idx)
    mask = edit.dataset.getmask(idx=args.sample_idx, choose_sem=args.choose_sem)
    if torch.is_tensor(mask):
        mask = mask.to(edit.device)
    xt_clean, t, _ = edit.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=edit.edit_t_idx)
    xt_clean = xt_clean.to(device=edit.device, dtype=edit.dtype)
    if torch.is_tensor(t):
        t = t.to(edit.device)

    v_clean = blob["v_clean"].to(device=edit.device, dtype=edit.dtype)
    x0_clean = blob["x0_clean"].to(device=edit.device, dtype=edit.dtype)
    x0_adv = blob["x0_adv"].to(device=edit.device, dtype=edit.dtype)
    if x0_clean.ndim == 3:
        x0_clean = x0_clean.unsqueeze(0)
    if x0_adv.ndim == 3:
        x0_adv = x0_adv.unsqueeze(0)

    # vT_null_top: same look-up as phase3_defense_purify (handles the case
    # where the basis lives under a different --note than this run).
    import glob as _glob
    basis_rel = os.path.join(
        "basis", f"local_basis-{edit.edit_t}T-select-mask-{args.choose_sem}",
    )
    vT_null_file = f"vT-null-{args.pca_rank_null}.pt"
    runs_root = os.path.join(_THIS_DIR, "..", "runs")
    candidates = [
        os.path.join(edit.result_folder, basis_rel, vT_null_file),
        os.path.join(edit.result_folder, f"sample_idx{args.sample_idx}",
                     basis_rel, vT_null_file),
    ]
    candidates += _glob.glob(os.path.join(
        runs_root, "CelebA_HQ_HF-CelebA_HQ_mask-*", "results",
        f"sample_idx{args.sample_idx}", basis_rel, vT_null_file,
    ))
    vT_null_top = None
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isfile(c):
            vT_null_top = torch.load(c, map_location=edit.device)\
                .to(edit.dtype)[: args.pca_rank_null]
            print(f"[defense-strip] loaded vT_null_top from {c}")
            break
    if vT_null_top is None:
        raise SystemExit(
            "[defense-strip] vT-null-*.pt not found; run Phase 1 first.\n"
            "Tried:\n  " + "\n  ".join(candidates)
        )

    # parse the purify plan (default = bits:4 jpeg:75 blur:1.5)
    plans: list[tuple[str, float]] = []
    plan_str = args.purify_plan or "bits:4 jpeg:75 blur:1.5"
    for tok in plan_str.split():
        method, pvals = tok.split(":")
        for v in pvals.split(","):
            plans.append((method, float(v)))
    print(f"[defense-strip] plans = {plans}")

    # The blob lives in <run_dir>/attackB_result.pt; we save defended strips
    # into <run_dir>/<EXP_NAME>-pc_0.png next to it.
    run_dir = os.path.dirname(os.path.abspath(args.attack_b_result))
    print(f"[defense-strip] writing strips next to {run_dir}")

    summary = []
    for (method, param) in plans:
        t0 = time.time()
        x0_adv_pur = purify(x0_adv, method, param)
        x0_clean_pur = purify(x0_clean, method, param)
        psnr_clean = _psnr(x0_clean_pur, x0_clean)

        print(f"\n[defense-strip] purify={method}({param})  "
              f"PSNR(purify(x_0_clean), x_0_clean) = {psnr_clean:.2f} dB")

        xT_def = _invert_from_tensor(edit, x0_adv_pur)
        xt_def, t_def, t_idx_def = edit.DDIMforwardsteps(
            xT_def, t_start_idx=0, t_end_idx=edit.edit_t_idx, save_image=False,
        )
        assert t_idx_def == edit.edit_t_idx
        xt_def = xt_def.to(device=edit.device, dtype=edit.dtype)
        if torch.is_tensor(t_def):
            t_def = t_def.to(device=edit.device)

        _, _, vT_modify_def = edit.local_encoder_decoder_pullback_xt(
            x=xt_def, t=t_def, op="mid", block_idx=0,
            pca_rank=args.pca_rank, chunk_size=1,
            min_iter=10, max_iter=50,
            convergence_threshold=1e-4, mask=mask, noise=False,
        )
        vT_def = _project_to_v_clean_subspace(vT_modify_def.to(edit.dtype), vT_null_top)
        v_def = vT_def[0].view_as(xt_def).detach()
        cos_true_def = abs(float((v_clean.flatten() * v_def.flatten()).sum().item()))

        label = f"defenseD2-{method}-{param:g}-edit_strip"
        # Direct save of x0_adv_pur for the qualitative figure ("the input the
        # defender sees after purification").
        tvu.save_image(
            (x0_adv_pur / 2 + 0.5).clamp(0, 1),
            os.path.join(run_dir, f"x0_adv_purified_{method}_{param:g}.png"),
        )

        # Render strip via the existing utility (it writes the PNG under
        # edit.result_folder using EXP_NAME). After writing, we move it next
        # to attackB_result.pt for grep-friendly co-location.
        prev_result_folder = edit.result_folder
        edit.result_folder = run_dir  # write directly next to the attacker blob
        try:
            _render_strip_at(edit, xt_def, v_def, t_def, args=args, label=label)
        finally:
            edit.result_folder = prev_result_folder

        # Save tiny metadata json for repro.
        with open(os.path.join(run_dir, f"{label}.json"), "w") as f:
            json.dump({
                "method": method, "param": param,
                "psnr_purify_clean": psnr_clean,
                "cos_true_defended": cos_true_def,
                "misalign_defended": 1.0 - cos_true_def,
                "src_attackB_result": args.attack_b_result,
            }, f, indent=2)

        wall = time.time() - t0
        print(f"[defense-strip] {method}:{param:g}  "
              f"misalign_def={1 - cos_true_def:.3f}  "
              f"PSNR_clean={psnr_clean:.1f} dB  wall={wall:.1f}s")
        summary.append({"method": method, "param": param,
                        "misalign_defended": 1.0 - cos_true_def,
                        "psnr_purify_clean": psnr_clean,
                        "wall_seconds": wall})
        del xt_def, vT_modify_def, vT_def, v_def, x0_adv_pur, x0_clean_pur
        gc.collect()
        torch.cuda.empty_cache()

    out_csv = os.path.join(run_dir, "defenseD2_strip_summary.csv")
    import csv as _csv
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"\n[defense-strip] summary -> {out_csv}")


if __name__ == "__main__":
    main()
