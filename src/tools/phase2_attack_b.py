"""Phase 2 - Attack B: image-space PGD on x_0.

Threat model
------------
The defender's pipeline is identical to Phase 1: given an uploaded image `x_0`,
it runs DDIM inversion to obtain `x_T`, forward-diffuses to the editing latent
`x_t`, and discovers a rank-1 editing direction `v_clean = f(x_t)` via GPM +
null-space projection. The attacker perturbs the *uploaded image* with

    x_0_adv = x_0 + delta_img,   ||delta_img||_inf <= eps_img

and attack success is the direction misalignment after the full pipeline:

    M(delta_img) := 1 - | <f(x_t), f(x_t_adv)> |

where `x_t_adv` is the latent produced by running DDIM inversion + forward on
the *perturbed* image (no shortcut at verification time).

Differentiable vs. verification paths
-------------------------------------
We use two different x_0 -> x_t maps:

- DIFFERENTIABLE (during PGD):
  the closed-form forward-noising identity
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps_ref
  with `eps_ref = (x_t,clean - sqrt(alpha_bar_t) * x_0) / sqrt(1-alpha_bar_t)`
  extracted once from the clean DDIM trajectory. At delta_img=0 this is an
  exact identity, so the linearisation around the clean point is correct to
  first order. Differentiating this is a single differentiable linear map, so
  one PGD step costs exactly one JVP through the UNet - identical to Attack A.

- VERIFICATION (once per eps):
  the actual LOCO pipeline, no-grad: `x_0 + delta_img_adv` -> DDIM inversion
  (100 steps) -> DDIM forward (40 steps) -> x_t_adv -> full GPM -> v_adv.
  This is what a real deployment would do, so the reported misalignment is
  honest.

Why this is a stronger experiment than Attack A
-----------------------------------------------
Attack A perturbs x_t (an internal state the user never touches). It's a
sensitivity analysis, not a threat model. Attack B's perturbation lives in
the *same modality the user trusts* (the uploaded image), with a budget
expressed in the standard adversarial L_inf convention. Exactly analogous to
"imperceptible adversarial perturbations on classifiers", but targeting the
LOCO steering vector extractor.
"""

from __future__ import annotations

import gc
import glob
import json
import os
import sys
import time

import torch
import torchvision.utils as tvu

# Make `python -m tools.phase2_attack_b` and direct invocation both work.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.define_argparser import parse_args, preset
from utils.utils import extract
from modules.edit import EditUncondDiffusion

# Reuse Attack A helpers so there's one source of truth for projection,
# null-space handling, basis resolution and the edit-strip renderer.
from tools.phase2_attack_a import (
    _project,
    _step,
    _jvp_x0,
    _project_to_v_clean_subspace,
    _find_basis_dir,
    _render_attacked_edit,
)


# ---------------------------------------------------------------------------
# differentiable x_0 -> x_t map (closed-form forward noising)
# ---------------------------------------------------------------------------

def _alpha_bar_at(edit, t, shape):
    """Scheduler's cumulative alpha at timestep `t`, broadcast to `shape`."""
    ab = edit.scheduler.return_alphas_cumprod()
    return extract(ab, t, shape)


def _extract_eps_ref(edit, x0, xt_clean, t):
    """Infer the implicit noise that takes x0 to xt_clean under DDIM with eta=0.

    This satisfies exactly:
        xt_clean = sqrt(a_t) * x0 + sqrt(1 - a_t) * eps_ref
    """
    at = _alpha_bar_at(edit, t, x0.shape)
    eps_ref = (xt_clean - at.sqrt() * x0) / (1.0 - at).sqrt().clamp(min=1e-8)
    return eps_ref.detach()


def _forward_closed_form(edit, x0, eps_ref, t):
    """Closed-form x_0 -> x_t using the fixed eps_ref (differentiable in x_0)."""
    at = _alpha_bar_at(edit, t, x0.shape)
    return at.sqrt() * x0 + (1.0 - at).sqrt() * eps_ref


# ---------------------------------------------------------------------------
# attack loop: optimise delta_img (image space) via closed-form x_0 -> x_t
# ---------------------------------------------------------------------------

def pgd_attack_b(edit, x0, xt_clean, eps_ref, t, mask, v_clean, *,
                 eps_img, alpha_img, steps, init="zero"):
    """PGD on delta_img with proxy loss = cos( J(x_t)*v_clean , J(x_t_adv)*v_clean ).

    The only difference from pgd_attack_a is the variable we optimise:
    delta_img lives in image space; x_t_adv is obtained via the closed-form
    forward-noising identity so the chain rule is trivial.
    """
    g_clean = _jvp_x0(edit, t, xt_clean, v_clean, mask).detach()
    g_clean_flat = g_clean.flatten()
    g_clean_norm = g_clean_flat.norm().clamp(min=1e-12)

    if init == "rand":
        delta_img = (torch.rand_like(x0) * 2 - 1) * eps_img
        delta_img = _project(delta_img, eps_img, "linf")
    else:
        delta_img = torch.zeros_like(x0)
    delta_img = delta_img.detach().requires_grad_(True)

    # Freeze model params: only delta_img receives gradient.
    for p in edit.unet.parameters():
        p.requires_grad_(False)

    history = []
    for step_idx in range(steps):
        x0_adv = x0 + delta_img
        xt_adv = _forward_closed_form(edit, x0_adv, eps_ref, t)

        g_adv = _jvp_x0(edit, t, xt_adv, v_clean, mask)
        g_adv_flat = g_adv.flatten()
        g_adv_norm = g_adv_flat.norm().clamp(min=1e-12)
        cos = (g_adv_flat * g_clean_flat).sum() / (g_clean_norm * g_adv_norm)

        loss = cos
        grad = torch.autograd.grad(loss, delta_img, retain_graph=False)[0]

        with torch.no_grad():
            delta_img_new = _step(delta_img, grad, alpha_img, "linf")
            delta_img_new = _project(delta_img_new, eps_img, "linf")
        delta_img.data.copy_(delta_img_new)

        history.append({
            "step":          step_idx,
            "cos_proxy":     float(cos.item()),
            "delta_img_inf": float(delta_img.detach().abs().max().item()),
            "delta_img_l2":  float(delta_img.detach().flatten().norm().item()),
        })
        if (step_idx % 5 == 0) or (step_idx == steps - 1):
            print(f"[attack-b] step {step_idx:3d}  "
                  f"cos_proxy={cos.item():+.4f}  "
                  f"||d_img||_inf={delta_img.detach().abs().max().item():.4f}  "
                  f"||d_img||_2={delta_img.detach().flatten().norm().item():.3f}")

    return delta_img.detach(), history


# ---------------------------------------------------------------------------
# verification: run the full (non-diff) LOCO pipeline on x_0 + delta_img
# ---------------------------------------------------------------------------

@torch.no_grad()
def _invert_from_tensor(edit, x0):
    """Reimplementation of edit.run_DDIMinversion that starts from a tensor
    instead of a dataset index. Only the CelebA_HQ_HF branch is needed here
    because that's what Attack B targets.
    """
    num_inference_steps = edit.inv_steps
    edit.scheduler.set_timesteps(num_inference_steps, device=edit.device, is_inversion=True)
    timesteps = edit.scheduler.timesteps

    xt = x0.to(edit.device, dtype=edit.dtype)
    for i, t in enumerate(timesteps):
        if i == len(timesteps) - 1:
            break
        et = edit.unet(xt, t)
        if not isinstance(et, torch.Tensor):
            et = et.sample
        xt = edit.scheduler.step(et, t, xt, eta=0, use_clipped_model_output=None).prev_sample
    return xt


def verify_misalignment_b(edit, x0_adv, v_clean, vT_null_top, mask, *, pca_rank):
    """Run the full LOCO pipeline on x_0 + delta_img_adv and report TRUE cosine."""
    print("\n[verify-b] running full DDIM inversion + forward on x_0 + delta_adv ...")
    xT_adv = _invert_from_tensor(edit, x0_adv)
    xt_adv, t_adv, t_idx_adv = edit.DDIMforwardsteps(
        xT_adv, t_start_idx=0, t_end_idx=edit.edit_t_idx,
        save_image=False,
    )
    assert t_idx_adv == edit.edit_t_idx
    xt_adv = xt_adv.to(device=edit.device, dtype=edit.dtype)
    if torch.is_tensor(t_adv):
        t_adv = t_adv.to(device=edit.device)

    print("[verify-b] running GPM at the pipeline-produced x_t_adv ...")
    _, _, vT_modify_adv = edit.local_encoder_decoder_pullback_xt(
        x=xt_adv, t=t_adv, op="mid", block_idx=0,
        pca_rank=pca_rank, chunk_size=1,
        min_iter=10, max_iter=50,
        convergence_threshold=1e-4, mask=mask, noise=False,
    )
    vT_adv = _project_to_v_clean_subspace(vT_modify_adv.to(edit.dtype), vT_null_top)
    v_adv = vT_adv[0].view_as(xt_adv)
    cos_true = abs(float((v_clean.flatten() * v_adv.flatten()).sum().item()))
    print(f"[verify-b] |cos(v_clean, v_adv)| = {cos_true:.4f}")
    print(f"[verify-b] direction misalignment = {1 - cos_true:.4f}")
    return xt_adv.detach(), v_adv.detach(), cos_true


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args = preset(args)

    print(f"[attack-b] sample_idx={args.sample_idx}  sem={args.choose_sem}  "
          f"eps_img={args.attack_b_eps_img}  alpha_img={args.attack_b_alpha_img}  "
          f"steps={args.attack_b_steps}")

    edit = EditUncondDiffusion(args)

    # -- Step 1: latent + mask (same as Attack A).
    xT = edit.run_DDIMinversion(idx=args.sample_idx)
    mask = edit.dataset.getmask(idx=args.sample_idx, choose_sem=args.choose_sem)
    xt, t, t_idx = edit.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=edit.edit_t_idx)
    assert t_idx == edit.edit_t_idx

    xt = xt.to(device=edit.device, dtype=edit.dtype)
    if torch.is_tensor(t):
        t = t.to(device=edit.device)
    if torch.is_tensor(mask):
        mask = mask.to(device=edit.device)

    # -- Step 1b: load x_0 from dataset (needed for image-space perturbation
    # and to build eps_ref for the closed-form forward).
    x0 = edit.dataset[args.sample_idx].to(device=edit.device, dtype=edit.dtype)
    print(f"[attack-b] x_0 shape={tuple(x0.shape)} range=[{x0.min():.3f},{x0.max():.3f}] "
          f"(expected ~[-1, 1])")

    eps_ref = _extract_eps_ref(edit, x0, xt, t)
    # Sanity: re-applying the closed form at delta_img = 0 must reproduce xt_clean.
    xt_rt = _forward_closed_form(edit, x0, eps_ref, t)
    rt_err = (xt_rt - xt).flatten().norm().item()
    print(f"[attack-b] closed-form round-trip error at delta_img=0: {rt_err:.2e} "
          f"(should be ~0)")

    # -- Step 2: resolve basis (identical logic to Attack A).
    basis_rel = os.path.join(
        "basis", f"local_basis-{edit.edit_t}T-select-mask-{args.choose_sem}",
    )
    save_dir_own = os.path.join(edit.result_folder, basis_rel)
    vT_modify_name = f"vT-modify-pca-rank-{args.pca_rank}.pt"
    vT_null_name   = f"vT-null-{args.pca_rank_null}.pt"

    cached_dir = _find_basis_dir(args, edit, basis_rel)
    vT_modify_path = None
    vT_null_path = None
    if cached_dir is not None:
        cand_m = os.path.join(cached_dir, vT_modify_name)
        cand_n = os.path.join(cached_dir, vT_null_name)
        if os.path.exists(cand_m) and os.path.exists(cand_n):
            vT_modify_path = cand_m
            vT_null_path = cand_n
            print(f"[attack-b] re-using cached basis from {cached_dir!r}")

    if vT_modify_path is None:
        print(f"[attack-b] no cached basis found. Candidates tried: "
              f"(1) {save_dir_own!r}  "
              f"(2) --attack_basis_src={getattr(args, 'attack_basis_src', '')!r}  "
              f"(3) glob src/runs/CelebA_HQ_HF-CelebA_HQ_mask*/results/sample_idx{args.sample_idx}/{basis_rel}")
        print(f"[attack-b] falling back to on-the-fly GPM (chunk_size=1).")
        os.makedirs(save_dir_own, exist_ok=True)
        vT_modify_path = os.path.join(save_dir_own, vT_modify_name)
        vT_null_path   = os.path.join(save_dir_own, vT_null_name)

        _, _, vT_modify_t = edit.local_encoder_decoder_pullback_xt(
            x=xt, t=t, op="mid", block_idx=0,
            pca_rank=args.pca_rank, chunk_size=1,
            min_iter=10, max_iter=50,
            convergence_threshold=1e-4, mask=mask, noise=False,
        )
        torch.save(vT_modify_t.detach().cpu(), vT_modify_path)
        del vT_modify_t
        gc.collect()
        torch.cuda.empty_cache()

        _, _, vT_null_t = edit.local_encoder_decoder_pullback_xt(
            x=xt, t=t, op="mid", block_idx=0,
            pca_rank=args.pca_rank_null, chunk_size=1,
            min_iter=10, max_iter=50,
            convergence_threshold=1e-4, mask=~mask, noise=False,
        )
        torch.save(vT_null_t.detach().cpu(), vT_null_path)
        del vT_null_t
        gc.collect()
        torch.cuda.empty_cache()

    vT_modify = torch.load(vT_modify_path, map_location=edit.device).to(edit.dtype)
    vT_null   = torch.load(vT_null_path,   map_location=edit.device).to(edit.dtype)
    vT_null_top = vT_null[: args.pca_rank_null, :]

    vT_clean = _project_to_v_clean_subspace(vT_modify, vT_null_top)
    v_clean = vT_clean[0].view_as(xt).detach()
    print(f"[attack-b] v_clean ready: shape={tuple(v_clean.shape)}")

    # -- Step 3-5: PGD over a single eps_img OR a sweep.
    if args.attack_b_eps_sweep.strip():
        eps_list = [float(s) for s in args.attack_b_eps_sweep.split(",") if s.strip()]
    else:
        eps_list = [args.attack_b_eps_img]

    summary_rows = []
    for eps_img in eps_list:
        t0 = time.time()
        delta_img, history = pgd_attack_b(
            edit, x0, xt, eps_ref, t, mask, v_clean,
            eps_img=eps_img, alpha_img=args.attack_b_alpha_img,
            steps=args.attack_b_steps, init=args.attack_init,
        )

        x0_adv = (x0 + delta_img).detach().clamp(-1.0, 1.0)
        xt_adv, v_adv, cos_true = verify_misalignment_b(
            edit, x0_adv, v_clean, vT_null_top, mask, pca_rank=args.pca_rank,
        )
        wall = time.time() - t0

        out_dir = os.path.join(
            edit.result_folder,
            f"attackB-linf-eps_img{eps_img:g}-{args.attack_b_steps}steps",
        )
        os.makedirs(out_dir, exist_ok=True)

        # Save imperceptibility artefacts: x_0, x_0_adv, delta_img (rescaled).
        tvu.save_image((x0 / 2 + 0.5).clamp(0, 1),
                       os.path.join(out_dir, "x0_clean.png"))
        tvu.save_image((x0_adv / 2 + 0.5).clamp(0, 1),
                       os.path.join(out_dir, "x0_adv.png"))
        # rescale delta to [0,1] for visual inspection
        d = delta_img.detach()
        d_vis = (d - d.min()) / (d.max() - d.min() + 1e-8)
        tvu.save_image(d_vis, os.path.join(out_dir, "delta_img_rescaled.png"))

        torch.save(
            {
                "delta_img": delta_img.cpu(),
                "x0_clean":  x0.cpu(),
                "x0_adv":    x0_adv.cpu(),
                "xt_clean":  xt.cpu(),
                "xt_adv":    xt_adv.cpu(),
                "v_clean":   v_clean.cpu(),
                "v_adv":     v_adv.cpu(),
                "cos_true":  cos_true,
                "history":   history,
                "args":      vars(args),
            },
            os.path.join(out_dir, "attackB_result.pt"),
        )
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(
                {
                    "sample_idx":         args.sample_idx,
                    "choose_sem":         args.choose_sem,
                    "edit_t":             args.edit_t,
                    "pca_rank":           args.pca_rank,
                    "pca_rank_null":      args.pca_rank_null,
                    "eps_img":            eps_img,
                    "eps_img_255":        eps_img * 127.5,
                    "alpha_img":          args.attack_b_alpha_img,
                    "steps":              args.attack_b_steps,
                    "init":               args.attack_init,
                    "cos_proxy_init":     history[0]["cos_proxy"],
                    "cos_proxy_final":    history[-1]["cos_proxy"],
                    "cos_true":           cos_true,
                    "misalignment":       1.0 - cos_true,
                    "delta_img_inf_final": history[-1]["delta_img_inf"],
                    "delta_img_l2_final":  history[-1]["delta_img_l2"],
                    "wall_seconds":       wall,
                },
                f, indent=2,
            )
        summary_rows.append({
            "eps_img":            eps_img,
            "eps_img_255":        eps_img * 127.5,
            "cos_true":           cos_true,
            "misalignment":       1.0 - cos_true,
            "cos_proxy_final":    history[-1]["cos_proxy"],
        })
        print(f"[attack-b] eps_img={eps_img:g} (~{eps_img*127.5:.2f}/255)  "
              f"misalign={1 - cos_true:.4f}  "
              f"(proxy final={history[-1]['cos_proxy']:+.4f})  "
              f"wall={wall:.1f}s  saved -> {out_dir}")

        # Optional: render the post-attack LOCO edit strip at xt_adv.
        if args.attack_render_edit:
            print(f"[attack-b] rendering post-attack edit strip at eps_img={eps_img:g} ...")
            try:
                delta_xt = (xt_adv - xt).detach()
                # _render_attacked_edit expects (edit, xt_clean, delta_xt, t, mask, v_adv,...)
                _render_attacked_edit(edit, xt, delta_xt, t, mask, v_adv,
                                      out_dir=out_dir, args=args)
            except Exception as exc:
                print(f"[attack-b] WARN: render_attacked_edit failed: {exc}")

    if len(summary_rows) > 1:
        sweep_csv = os.path.join(
            edit.result_folder,
            f"attackB-linf-{args.attack_b_steps}steps-sweep.csv",
        )
        import csv as _csv
        with open(sweep_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[attack-b] sweep summary -> {sweep_csv}")


if __name__ == "__main__":
    main()
