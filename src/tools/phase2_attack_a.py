"""Phase 2 - Attack A: direction instability via PGD on x_t.

Threat model
------------
The defender's pipeline is the LOCO direction extractor `f`:

    f(x_t) = top-1 right singular vector of the mask-restricted PMP Jacobian
             at x_t, after orthogonal projection onto the null-space spanned
             by the top `pca_rank_null` outside-mask singular vectors.

The attacker has white-box access to the same UNet, scheduler, mask M and the
pre-computed null-basis V_null. The attacker may add a perturbation delta to
the editing latent x_t with ||delta||_p <= eps. Attack success is measured as

    M(delta) := 1 - | <f(x_t), f(x_t + delta)> |

i.e. directional misalignment of the discovered editing direction.

Why a proxy loss
----------------
f(.) is the output of a generalised power method (GPM) - many UNet forwards
to converge a singular vector. Differentiating through GPM is expensive.
Instead we use:

    L_proxy(delta) = cos( J(x_t).v_clean ,  J(x_t + delta).v_clean )

where v_clean = f(x_t) is computed once and frozen. Each PGD step is then
two forward-mode JVPs through the UNet. After PGD converges we VERIFY by
re-running the full GPM at x_t + delta_adv to recover v_adv and reporting
the true cosine.

Run after Phase 1 (which produces the cached vT-modify and vT-null files
under src/runs/<exp>/results/sample_idx<idx>/basis/local_basis-...).
"""

from __future__ import annotations

import gc
import glob
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

# Make `python -m tools.phase2_attack_a` and direct invocation both work.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.define_argparser import parse_args, preset
from modules.edit import EditUncondDiffusion


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _project(delta: torch.Tensor, eps: float, norm: str) -> torch.Tensor:
    """Project `delta` back onto the eps-ball of the given norm."""
    if norm == "linf":
        return delta.clamp(-eps, eps)
    if norm == "l2":
        flat = delta.view(delta.size(0), -1)
        n = flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
        scale = (eps / n).clamp(max=1.0)
        return (flat * scale).view_as(delta)
    raise ValueError(f"unknown norm {norm!r}")


def _step(delta: torch.Tensor, grad: torch.Tensor, alpha: float, norm: str) -> torch.Tensor:
    """One PGD step: descend on `grad` (we MINIMIZE the proxy loss)."""
    if norm == "linf":
        return delta - alpha * grad.sign()
    if norm == "l2":
        gn = grad.flatten().norm().clamp(min=1e-12)
        return delta - alpha * grad / gn
    raise ValueError(f"unknown norm {norm!r}")


def _jvp_x0(edit, t, x, v, mask):
    """Compute J(x) @ v where J is the Jacobian of get_x0(t, .) at x."""
    f = lambda xx: edit.get_x0(t, xx, mask=mask)
    _out, jv = torch.func.jvp(f, (x,), (v,))
    return jv


def _project_to_v_clean_subspace(vT_modify, vT_null_top):
    """Apply the same null-space projection used in run_edit_null_space_projection."""
    proj = (vT_null_top.T @ (vT_null_top @ vT_modify.T)).T
    vT = vT_modify - proj
    vT = vT / vT.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return vT


def _find_basis_dir(args, edit, basis_rel: str):
    """Find an existing local_basis folder to re-use instead of recomputing.

    Priority:
      1. this run's own result_folder (default, matches on first resubmit)
      2. --attack_basis_src, which may point at any ancestor of the basis dir
      3. glob: any sibling run under the CelebA_HQ_HF-CelebA_HQ_mask* tree with
         a matching sample_idx + mask basis folder already written to disk.
    Returns an absolute folder path, or None if nothing matches.
    """
    candidates = []

    primary = os.path.join(edit.result_folder, basis_rel)
    candidates.append(primary)

    src = getattr(args, "attack_basis_src", "")
    if src:
        if os.path.isdir(src) and os.path.basename(src.rstrip("/")).startswith("local_basis-"):
            candidates.append(src)
        else:
            candidates.append(os.path.join(src, basis_rel))
            candidates.append(os.path.join(src, "basis", os.path.basename(basis_rel)))

    pattern = os.path.join(
        "src", "runs",
        "CelebA_HQ_HF-CelebA_HQ_mask*",
        "results",
        f"sample_idx{args.sample_idx}",
        basis_rel,
    )
    candidates.extend(sorted(glob.glob(pattern)))

    for cand in candidates:
        if cand and os.path.isdir(cand):
            return cand
    return None


# ---------------------------------------------------------------------------
# attack loop (returns delta + history; does NOT verify via GPM)
# ---------------------------------------------------------------------------

def pgd_attack_a(edit, xt, t, mask, v_clean, *, eps, alpha, steps, norm, init):
    # Anchor: image-space tangent at the clean latent.
    g_clean = _jvp_x0(edit, t, xt, v_clean, mask).detach()
    g_clean_flat = g_clean.flatten()
    g_clean_norm = g_clean_flat.norm().clamp(min=1e-12)

    # Initialise delta inside the eps-ball.
    if init == "rand":
        delta = (torch.rand_like(xt) * 2 - 1) * eps
        delta = _project(delta, eps, norm)
    else:
        delta = torch.zeros_like(xt)
    delta = delta.detach().requires_grad_(True)

    # Freeze model params; we only differentiate w.r.t. delta.
    for p in edit.unet.parameters():
        p.requires_grad_(False)

    history = []
    for step in range(steps):
        # Single PGD iteration.
        x_adv = xt + delta
        g_adv = _jvp_x0(edit, t, x_adv, v_clean, mask)
        g_adv_flat = g_adv.flatten()
        g_adv_norm = g_adv_flat.norm().clamp(min=1e-12)
        cos = (g_adv_flat * g_clean_flat).sum() / (g_clean_norm * g_adv_norm)

        # Loss to MINIMIZE = +cos (we want g_adv to rotate away from g_clean).
        loss = cos
        grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]

        with torch.no_grad():
            delta_new = _step(delta, grad, alpha, norm)
            delta_new = _project(delta_new, eps, norm)
        delta.data.copy_(delta_new)

        history.append({
            "step":       step,
            "cos_proxy":  float(cos.item()),
            "delta_norm": float(delta.detach().flatten().norm().item()),
        })
        if (step % 5 == 0) or (step == steps - 1):
            print(f"[attack] step {step:3d}  cos_proxy={cos.item():+.4f}  "
                  f"||delta||_2={delta.detach().flatten().norm().item():.3f}")

    return delta.detach(), history


# ---------------------------------------------------------------------------
# verification: recompute the full direction at x_adv and report TRUE cosine
# ---------------------------------------------------------------------------

def verify_misalignment(edit, xt, delta, t, mask, v_clean, vT_null_top, *, pca_rank):
    print("\n[verify] running GPM at x_t + delta_adv ...")
    x_adv = xt + delta
    u_adv, s_adv, vT_modify_adv = edit.local_encoder_decoder_pullback_xt(
        x=x_adv, t=t, op="mid", block_idx=0,
        pca_rank=pca_rank, min_iter=10, max_iter=50,
        convergence_threshold=1e-4, mask=mask, noise=False,
    )
    vT_adv = _project_to_v_clean_subspace(vT_modify_adv.to(edit.dtype), vT_null_top)
    v_adv = vT_adv[0].view_as(xt)
    cos_true = abs(float((v_clean.flatten() * v_adv.flatten()).sum().item()))
    print(f"[verify] |cos(v_clean, v_adv)| = {cos_true:.4f}")
    print(f"[verify] direction misalignment = {1 - cos_true:.4f}")
    return v_adv.detach(), cos_true


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args = preset(args)

    print(f"[attack-a] sample_idx={args.sample_idx}  sem={args.choose_sem}  "
          f"eps={args.attack_eps}  alpha={args.attack_alpha}  "
          f"steps={args.attack_steps}  norm={args.attack_norm}")

    edit = EditUncondDiffusion(args)

    # Step 1: latent + mask + clean direction.
    xT = edit.run_DDIMinversion(idx=args.sample_idx)
    mask = edit.dataset.getmask(idx=args.sample_idx, choose_sem=args.choose_sem)
    xt, t, t_idx = edit.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=edit.edit_t_idx)
    assert t_idx == edit.edit_t_idx, f"DDIM forward landed at idx {t_idx}, expected {edit.edit_t_idx}"

    # DDIMforwardsteps returns xt on self.buffer_device (CPU); t lives on GPU.
    # local_encoder_decoder_pullback_xt builds its random basis on x.device, so if
    # xt is on CPU we end up calling the GPU UNet with CPU inputs -> the
    # `mat1 is on cpu, ... cuda:0` crash inside time_embedding.linear_1.
    # Mirror what run_edit_null_space_projection does: move xt + mask to GPU,
    # and make sure t is on the same device too. Done once, here.
    xt = xt.to(device=edit.device, dtype=edit.dtype)
    if torch.is_tensor(t):
        t = t.to(device=edit.device)
    if torch.is_tensor(mask):
        mask = mask.to(device=edit.device)

    # Step 2: load the cached Phase-1 basis if at all possible; recompute only
    # as a last resort (re-running both GPMs back-to-back in the same process
    # has repeatedly OOM'd the 80 GB H100 because
    # `torch.autograd.functional.jacobian` holds `pca_rank` full UNet
    # activation graphs and the first GPM's allocator fragments never release).
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
            print(f"[attack-a] re-using cached basis from {cached_dir!r}")

    if vT_modify_path is None:
        print(f"[attack-a] no cached basis found. Candidates tried: "
              f"(1) {save_dir_own!r}  "
              f"(2) --attack_basis_src={getattr(args, 'attack_basis_src', '')!r}  "
              f"(3) glob src/runs/CelebA_HQ_HF-CelebA_HQ_mask*/results/sample_idx{args.sample_idx}/{basis_rel}")
        print(f"[attack-a] falling back to on-the-fly GPM. "
              f"Using chunk_size=1 and aggressive cache clearing to avoid OOM.")
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
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        print(f"[attack-a] after modify GPM: CUDA mem allocated = "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB, reserved = "
              f"{torch.cuda.memory_reserved() / 1e9:.2f} GB")

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
    print(f"[attack-a] v_clean ready: shape={tuple(v_clean.shape)}, "
          f"norm={v_clean.flatten().norm().item():.4f}")

    # Step 3-5: PGD over a single eps OR a sweep.
    if args.attack_eps_sweep.strip():
        eps_list = [float(s) for s in args.attack_eps_sweep.split(",") if s.strip()]
    else:
        eps_list = [args.attack_eps]

    summary_rows = []
    for eps in eps_list:
        t0 = time.time()
        delta, history = pgd_attack_a(
            edit, xt, t, mask, v_clean,
            eps=eps, alpha=args.attack_alpha, steps=args.attack_steps,
            norm=args.attack_norm, init=args.attack_init,
        )
        v_adv, cos_true = verify_misalignment(
            edit, xt, delta, t, mask, v_clean, vT_null_top, pca_rank=args.pca_rank,
        )
        wall = time.time() - t0

        out_dir = os.path.join(
            edit.result_folder,
            f"attackA-{args.attack_norm}-eps{eps:g}-{args.attack_steps}steps",
        )
        os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {
                "delta":     delta.cpu(),
                "x_adv":     (xt + delta).detach().cpu(),
                "v_clean":   v_clean.cpu(),
                "v_adv":     v_adv.cpu(),
                "cos_true":  cos_true,
                "history":   history,
                "args":      vars(args),
            },
            os.path.join(out_dir, "attackA_result.pt"),
        )
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(
                {
                    "sample_idx":      args.sample_idx,
                    "choose_sem":      args.choose_sem,
                    "edit_t":          args.edit_t,
                    "pca_rank":        args.pca_rank,
                    "pca_rank_null":   args.pca_rank_null,
                    "norm":            args.attack_norm,
                    "eps":             eps,
                    "alpha":           args.attack_alpha,
                    "steps":           args.attack_steps,
                    "init":            args.attack_init,
                    "cos_proxy_init":  history[0]["cos_proxy"],
                    "cos_proxy_final": history[-1]["cos_proxy"],
                    "cos_true":        cos_true,
                    "misalignment":    1.0 - cos_true,
                    "wall_seconds":    wall,
                },
                f, indent=2,
            )
        summary_rows.append({
            "eps": eps,
            "cos_true": cos_true,
            "misalignment": 1.0 - cos_true,
            "cos_proxy_final": history[-1]["cos_proxy"],
        })
        print(f"[attack-a] eps={eps:g}  misalign={1 - cos_true:.4f}  "
              f"(proxy final={history[-1]['cos_proxy']:+.4f})  "
              f"wall={wall:.1f}s  saved -> {out_dir}")

        # Step 6: optionally render the post-attack edit strip for the report.
        if args.attack_render_edit:
            print(f"[attack-a] rendering post-attack edit strip at eps={eps:g} ...")
            try:
                _render_attacked_edit(edit, xt, delta, t, mask, v_adv,
                                      out_dir=out_dir, args=args)
            except Exception as exc:
                print(f"[attack-a] WARN: render_attacked_edit failed: {exc}")

    # Sweep summary CSV (handy for the lambda-vs-eps style plot).
    if len(summary_rows) > 1:
        sweep_csv = os.path.join(
            edit.result_folder,
            f"attackA-{args.attack_norm}-{args.attack_steps}steps-sweep.csv",
        )
        import csv as _csv
        with open(sweep_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[attack-a] sweep summary -> {sweep_csv}")


def _render_attacked_edit(edit, xt, delta, t, mask, v_adv, *, out_dir, args):
    """Apply the (post-attack) editing direction at x_t + delta and save a strip.

    This reuses edit.x_space_guidance_direct + DDIMforwardsteps the same way
    run_edit_null_space_projection does. The only differences are:
      - we start from the attacked latent x_t + delta, not from x_t,
      - we use v_adv (the rotated direction) instead of the clean direction.
    """
    import torchvision.utils as tvu
    from tqdm import tqdm

    x_adv = (xt + delta).detach()
    v = v_adv.view(-1, *xt.shape[1:])
    xts = {1: None, -1: None}

    for direction in [1, -1]:
        vk = direction * v
        chain = [x_adv.clone()]
        for _ in tqdm(range(edit.x_space_guidance_num_step),
                      desc=f"attacked edit ({'+' if direction == 1 else '-'})"):
            x_next = edit.x_space_guidance_direct(
                chain[-1], t_idx=edit.edit_t_idx, vk=vk,
                single_edit_step=edit.x_space_guidance_edit_step,
            )
            chain.append(x_next)
        xt_dir = torch.cat(chain, dim=0)
        if args.vis_num == 1:
            xt_dir = xt_dir[[0, -1], :]
        else:
            xt_dir = xt_dir[:: (xt_dir.size(0) // args.vis_num)]
        xts[direction] = xt_dir

    xt_full = torch.cat([xts[-1].flip(dims=[0])[:-1], xts[1]], dim=0)
    edit.EXP_NAME = (
        f"attackA-{args.sample_idx}-{args.choose_sem}-eps{args.attack_eps:g}"
        f"-{args.attack_norm}-edit_strip"
    )
    edit.DDIMforwardsteps(xt_full, t_start_idx=edit.edit_t_idx,
                          t_end_idx=-1, performance_boosting=True)
    print(f"[attack-a] saved attacked edit strip under {edit.result_folder} "
          f"with EXP_NAME={edit.EXP_NAME}")


if __name__ == "__main__":
    main()
