#!/bin/bash
#SBATCH --job-name=loco_p2_locality
#SBATCH --time=01:30:00
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
# Phase 2 - Locality / mask-leakage audit.
#
# Re-uses cached attack_*_result.pt files (does NOT re-run PGD). For each run
# it renders clean and attacked LOCO edits, decodes to x_0, and computes a
# leakage ratio (outside-mask change / total change). A compelling Phase 2
# attack should increase leakage by a large margin over the Phase 1 baseline.
#
# Usage:
#   sbatch scripts/nibi/phase2_locality.sh \
#       /project/.../CelebAMask-HQ 4729 l_eye B phase2_attackB_eye_sweep \
#       /project/.../sample_idx4729
#
# Args:
#   $1  CELEBA_ROOT   - dataset root
#   $2  SAMPLE_IDX    - integer, e.g. 4729
#   $3  CHOOSE_SEM    - same semantic as the attack run, e.g. l_eye
#   $4  ATTACK_TYPE   - "A" or "B"
#   $5  NOTE          - exp folder suffix (MUST match the attack run's --note)
#   $6  SWEEP_DIR     - absolute path to the results/sample_idx<N>/ folder from
#                       the attack run (contains attack*-eps*-*/attack*_result.pt)
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

CELEBA_ROOT="${1:?Usage: sbatch phase2_locality.sh CELEBA_ROOT SAMPLE_IDX CHOOSE_SEM ATTACK_TYPE NOTE SWEEP_DIR}"
SAMPLE_IDX="${2:-4729}"
CHOOSE_SEM="${3:-l_eye}"
ATTACK_TYPE="${4:-B}"
NOTE="${5:-phase2_attackB_eye_sweep}"
SWEEP_DIR="${6:-}"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "REPO_ROOT  : $REPO_ROOT"
echo "ATTACK_TYPE: $ATTACK_TYPE"
echo "NOTE       : $NOTE"
echo "SWEEP_DIR  : ${SWEEP_DIR:-<auto>}"

cd "$REPO_ROOT/src"

python tools/phase2_locality.py \
  --sh_file_name phase2_locality.sh \
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
  --attack_type "$ATTACK_TYPE" \
  --sweep_dir "$SWEEP_DIR"
