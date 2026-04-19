#!/bin/bash
#SBATCH --account=def-dennisg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=loco_p2_attackA
#SBATCH --output=logs/loco_p2_attackA_%j.out
#SBATCH --error=logs/loco_p2_attackA_%j.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Phase 2 - Attack A (PGD on x_t to rotate the discovered editing direction).
#
# Usage:
#   sbatch scripts/nibi/phase2_attack_a.sh \
#       /project/.../CelebAMask-HQ \
#       4729 l_eye 0.02 linf 40 phase2_attackA_eye
#
# Or as an eps sweep:
#   sbatch scripts/nibi/phase2_attack_a.sh \
#       /project/.../CelebAMask-HQ 4729 l_eye \
#       SWEEP linf 40 phase2_attackA_eye_sweep "0.005,0.01,0.02,0.05,0.1"
#
# Args (positional):
#   $1  CELEBA_ROOT  - dataset root
#   $2  SAMPLE_IDX   - integer image index, e.g. 4729
#   $3  CHOOSE_SEM   - semantic to attack (l_eye, nose, hair, ...)
#   $4  EPS          - eps for single-eps run, or literal "SWEEP" for sweep
#   $5  NORM         - "linf" or "l2"
#   $6  STEPS        - number of PGD iterations
#   $7  NOTE         - exp folder suffix (used by define_argparser to separate runs)
#   $8  EPS_SWEEP    - comma-separated list when $4=="SWEEP", e.g. "0.005,0.01,0.02"
# -----------------------------------------------------------------------------

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

CELEBA_ROOT="${1:?Usage: sbatch [...] phase2_attack_a.sh CELEBA_ROOT SAMPLE_IDX CHOOSE_SEM EPS NORM STEPS NOTE [EPS_SWEEP]}"
SAMPLE_IDX="${2:-4729}"
CHOOSE_SEM="${3:-l_eye}"
EPS="${4:-0.02}"
NORM="${5:-linf}"
STEPS="${6:-40}"
NOTE="${7:-phase2_attackA}"
EPS_SWEEP="${8:-}"

source ~/venvs/ditcap/bin/activate

export HF_HOME="$REPO_ROOT/.hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TORCH_HUB="$REPO_ROOT/.torch_hub"
mkdir -p "$HF_HOME" "$TORCH_HUB"

# Step size: 25% of eps is a sane default for PGD.
if [[ "$EPS" == "SWEEP" ]]; then
  EPS_FOR_ALPHA="$(echo "$EPS_SWEEP" | awk -F',' '{print $1}')"
  ALPHA="$(python -c "print(${EPS_FOR_ALPHA}*0.25)")"
  EPS_FLAG=""
  EPS_SWEEP_FLAG="--attack_eps_sweep $EPS_SWEEP"
else
  ALPHA="$(python -c "print(${EPS}*0.25)")"
  EPS_FLAG="--attack_eps $EPS"
  EPS_SWEEP_FLAG=""
fi

echo "REPO_ROOT     : $REPO_ROOT"
echo "SAMPLE_IDX    : $SAMPLE_IDX"
echo "CHOOSE_SEM    : $CHOOSE_SEM"
echo "EPS           : $EPS"
echo "EPS_SWEEP     : ${EPS_SWEEP:-<single>}"
echo "NORM          : $NORM"
echo "STEPS         : $STEPS"
echo "ALPHA         : $ALPHA"
echo "NOTE          : $NOTE"

cd src

python tools/phase2_attack_a.py \
  --sh_file_name phase2_attack_a.sh \
  --sample_idx "$SAMPLE_IDX" \
  --device cuda:0 \
  --dtype fp32 \
  --seed 0 \
  --model_name CelebA_HQ_HF \
  --dataset_name CelebA_HQ_mask \
  --dataset_root "$CELEBA_ROOT" \
  --for_steps 100 \
  --inv_steps 100 \
  --use_yh_custom_scheduler True \
  --performance_boosting_t 0.2 \
  --edit_t 0.6 \
  --choose_sem "$CHOOSE_SEM" \
  --null_space_projection True \
  --use_mask True \
  --pca_rank_null 5 \
  --pca_rank 1 \
  --vis_num 2 \
  --x_space_guidance_edit_step 1 \
  --x_space_guidance_scale 0.5 \
  --x_space_guidance_num_step 16 \
  --note "$NOTE" \
  --cache_folder "$HF_HOME" \
  --run_attack_a True \
  $EPS_FLAG \
  $EPS_SWEEP_FLAG \
  --attack_alpha "$ALPHA" \
  --attack_steps "$STEPS" \
  --attack_norm "$NORM" \
  --attack_init zero \
  --attack_render_edit True
