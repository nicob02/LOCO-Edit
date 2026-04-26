#!/bin/bash
#SBATCH --job-name=loco_p3_defstrip
#SBATCH --time=00:30:00
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
# Phase 3 - Defense D2: render the *defended* LOCO edit strip(s) for one
# attacker blob (one (sample, semantic, eps) configuration).
#
# Usage:
#   sbatch scripts/nibi/phase3_defense_strip.sh \
#     /project/.../CelebAMask-HQ \
#     4729 \
#     l_eye \
#     phase2_attackB_ms_s4729 \
#     /full/path/to/attackB_result.pt \
#     "bits:4 jpeg:75 blur:1.5"
#
# Args (positional):
#   $1  CELEBA_ROOT     - dataset root
#   $2  SAMPLE_IDX      - integer image index used by the attack
#   $3  CHOOSE_SEM      - semantic of the basis (l_eye, nose, hair, ...)
#   $4  NOTE            - same NOTE as the Attack B run (for basis lookup)
#   $5  ATTACK_B_RESULT - absolute path to attackB_result.pt
#   $6  PURIFY_PLAN     - space-separated 'method:p1,p2 ...'  (default 3 plans)
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
PURIFY_PLAN="${6:-bits:4 jpeg:75 blur:1.5}"

# The Python script gets executed with CWD=$REPO_ROOT/src (see line below),
# but the user typically passes ATTACK_B_RESULT as a path relative to the
# repo root (e.g. 'src/runs/...'). Absolutise it now so torch.load() finds
# it regardless of where Python is run from.
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

echo "REPO_ROOT      : $REPO_ROOT"
echo "SAMPLE_IDX     : $SAMPLE_IDX"
echo "CHOOSE_SEM     : $CHOOSE_SEM"
echo "NOTE           : $NOTE"
echo "ATTACK_B_RESULT: $ATTACK_B_RESULT"
echo "PURIFY_PLAN    : $PURIFY_PLAN"

cd "$REPO_ROOT/src"

python tools/phase3_defense_strip.py \
  --sh_file_name phase3_defense_strip.sh \
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
  --purify_plan "$PURIFY_PLAN"
