#!/bin/bash
#SBATCH --job-name=loco_p2_transfer
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=def-dennisg

set -euo pipefail

# -----------------------------------------------------------------------------
# Phase 2 - Attack B transfer audit.
#
# Takes delta_img_adv from ONE attackB_result.pt and evaluates its effect on a
# list of *different* target sample indices (never seen during PGD).
#
# Usage:
#   sbatch scripts/nibi/phase2_transfer.sh \
#       /project/.../CelebAMask-HQ l_eye phase2_attackB_eye_transfer \
#       /project/.../attackB-linf-eps_img0.031-40steps/attackB_result.pt \
#       "1000,2000,3000"
# -----------------------------------------------------------------------------

module --force purge
module load StdEnv/2023
module load python/3.11.5

VENV="${LOCO_VENV:-$HOME/venvs/ditcap}"
source "${VENV}/bin/activate"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

CELEBA_ROOT="${1:?Usage: sbatch phase2_transfer.sh CELEBA_ROOT CHOOSE_SEM NOTE ATTACK_B_RESULT TARGETS}"
CHOOSE_SEM="${2:-l_eye}"
NOTE="${3:-phase2_attackB_transfer}"
ATTACK_B_RESULT="${4:?missing attackB_result.pt path}"
TARGETS="${5:-1000,2000,3000}"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "REPO_ROOT       : $REPO_ROOT"
echo "ATTACK_B_RESULT : $ATTACK_B_RESULT"
echo "TARGETS         : $TARGETS"

cd "$REPO_ROOT/src"

# sample_idx here is just a placeholder required by preset() -- the tool
# ignores it and uses --transfer_targets. Keep it small/valid.
python tools/phase2_transfer.py \
  --sh_file_name phase2_transfer.sh \
  --sample_idx 0 \
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
  --attack_b_result "$ATTACK_B_RESULT" \
  --transfer_targets "$TARGETS"
