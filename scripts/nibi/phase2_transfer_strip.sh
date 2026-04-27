#!/bin/bash
#SBATCH --job-name=loco_p2_transfer_strip
#SBATCH --time=01:00:00
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
# Phase 2 - Attack B transfer: render edit strips (one per target sample) for
# the source's optimal delta_img applied to each target.
#
# Usage:
#   sbatch scripts/nibi/phase2_transfer_strip.sh \
#     /project/.../CelebAMask-HQ \
#     4729 \
#     l_eye \
#     phase2_attackB_ms_s4729 \
#     /full/path/to/attackB_result.pt \
#     "500,1000,2000,5000,10000"
#
# Args (positional):
#   $1  CELEBA_ROOT       - dataset root
#   $2  SAMPLE_IDX        - source sample index (used only for arg-parser)
#   $3  CHOOSE_SEM        - semantic (must match source)
#   $4  NOTE              - same NOTE as the Attack B run (for basis lookup)
#   $5  ATTACK_B_RESULT   - absolute or repo-relative path to attackB_result.pt
#   $6  TRANSFER_TARGETS  - comma-separated target sample idx list
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

CELEBA_ROOT="${1:?need CELEBA_ROOT}"
SAMPLE_IDX="${2:?need SAMPLE_IDX}"
CHOOSE_SEM="${3:?need CHOOSE_SEM}"
NOTE="${4:?need NOTE}"
ATTACK_B_RESULT="${5:?need ATTACK_B_RESULT path}"
TRANSFER_TARGETS="${6:?need TRANSFER_TARGETS comma list}"

# Absolutise paths since the python script runs with CWD=$REPO_ROOT/src.
if [[ "$ATTACK_B_RESULT" != /* ]]; then
  ATTACK_B_RESULT="$REPO_ROOT/$ATTACK_B_RESULT"
fi
if [[ ! -f "$ATTACK_B_RESULT" ]]; then
  echo "ERROR: ATTACK_B_RESULT does not exist: $ATTACK_B_RESULT" >&2
  exit 2
fi
if [[ "$CELEBA_ROOT" != /* ]]; then
  CELEBA_ROOT="$REPO_ROOT/$CELEBA_ROOT"
fi

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "REPO_ROOT       : $REPO_ROOT"
echo "SAMPLE_IDX      : $SAMPLE_IDX"
echo "CHOOSE_SEM      : $CHOOSE_SEM"
echo "NOTE            : $NOTE"
echo "ATTACK_B_RESULT : $ATTACK_B_RESULT"
echo "TRANSFER_TARGETS: $TRANSFER_TARGETS"

cd "$REPO_ROOT/src"

python tools/phase2_transfer_strip.py \
  --sh_file_name phase2_transfer_strip.sh \
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
  --attack_b_result "$ATTACK_B_RESULT" \
  --transfer_targets "$TRANSFER_TARGETS" \
  --also_render_source True \
  --also_compute_misalign True
