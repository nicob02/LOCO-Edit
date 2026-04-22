#!/bin/bash
#SBATCH --job-name=loco_p3_D2
#SBATCH --time=03:00:00
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
# Phase 3 - Defense D2: input purification against Attack B.
#
# Usage:
#   sbatch scripts/nibi/phase3_defense_purify.sh \
#       /project/.../CelebAMask-HQ 4729 l_eye \
#       phase2_attackB_eye_sweep \
#       /project/.../sample_idx4729 \
#       "jpeg:50,75,90 blur:0.5,1.0,1.5 bits:4,6"
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

CELEBA_ROOT="${1:?need CELEBA_ROOT SAMPLE_IDX CHOOSE_SEM NOTE SWEEP_DIR [PLAN]}"
SAMPLE_IDX="${2:-4729}"
CHOOSE_SEM="${3:-l_eye}"
NOTE="${4:-phase2_attackB_eye_sweep}"
SWEEP_DIR="${5:-}"
PLAN="${6:-jpeg:50,75,90 blur:0.5,1.0,1.5 bits:4,6}"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "SWEEP_DIR  : $SWEEP_DIR"
echo "PURIFY     : $PLAN"

cd "$REPO_ROOT/src"

python tools/phase3_defense_purify.py \
  --sh_file_name phase3_defense_purify.sh \
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
  --attack_type B \
  --sweep_dir "$SWEEP_DIR" \
  --purify_plan "$PLAN"
