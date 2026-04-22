#!/bin/bash
#SBATCH --job-name=loco_p2_attackB_hijack
#SBATCH --time=02:00:00
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
# Phase 2 - Attack B (image-space PGD) with TARGETED HIJACK.
#
# Usage:
#   sbatch scripts/nibi/phase2_attack_b_hijack.sh \
#       /project/.../CelebAMask-HQ 4729 l_eye hair \
#       SWEEP 40 phase2_attackB_hijack_eye_to_hair \
#       "0.016,0.031,0.063" 1.0 \
#       /project/.../basis/local_basis-0.6T-select-mask-l_eye
#
# Args:
#   $1  CELEBA_ROOT  - dataset root
#   $2  SAMPLE_IDX   - integer, must match the sample used for BOTH Phase 1 bases
#   $3  SOURCE_SEM   - semantic to destroy (e.g. l_eye)
#   $4  TARGET_SEM   - semantic to hijack into (e.g. hair); requires Phase 1
#                      basis under the same sample_idx/edit_t
#   $5  EPS          - single eps_img or "SWEEP"
#   $6  STEPS        - PGD iterations
#   $7  NOTE         - exp folder suffix
#   $8  EPS_SWEEP    - comma-separated sweep if $5 == "SWEEP"
#   $9  BETA         - hijack weight (default 1.0)
#   $10 BASIS_SRC    - optional explicit path to source-sem basis folder
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

CELEBA_ROOT="${1:?need CELEBA_ROOT SAMPLE_IDX SOURCE_SEM TARGET_SEM EPS STEPS NOTE [EPS_SWEEP] [BETA] [BASIS_SRC]}"
SAMPLE_IDX="${2:-4729}"
SOURCE_SEM="${3:-l_eye}"
TARGET_SEM="${4:-hair}"
EPS="${5:-SWEEP}"
STEPS="${6:-40}"
NOTE="${7:-phase2_attackB_hijack}"
EPS_SWEEP="${8:-0.016,0.031,0.063}"
BETA="${9:-1.0}"
BASIS_SRC="${10:-}"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$EPS" == "SWEEP" ]]; then
  EPS_FOR_ALPHA="$(echo "$EPS_SWEEP" | awk -F',' '{print $1}')"
  ALPHA="$(python -c "print(${EPS_FOR_ALPHA}*0.25)")"
  EPS_FLAG=""
  EPS_SWEEP_FLAG="--attack_b_eps_sweep $EPS_SWEEP"
else
  ALPHA="$(python -c "print(${EPS}*0.25)")"
  EPS_FLAG="--attack_b_eps_img $EPS"
  EPS_SWEEP_FLAG=""
fi

echo "SOURCE_SEM : $SOURCE_SEM"
echo "TARGET_SEM : $TARGET_SEM"
echo "EPS        : $EPS  (sweep=$EPS_SWEEP)"
echo "BETA       : $BETA"
echo "BASIS_SRC  : ${BASIS_SRC:-<auto>}"

cd "$REPO_ROOT/src"

python tools/phase2_attack_b.py \
  --sh_file_name phase2_attack_b_hijack.sh \
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
  --choose_sem "$SOURCE_SEM" \
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
  --run_attack_b True \
  $EPS_FLAG \
  $EPS_SWEEP_FLAG \
  --attack_b_alpha_img "$ALPHA" \
  --attack_b_steps "$STEPS" \
  --attack_b_target_sem "$TARGET_SEM" \
  --attack_b_target_beta "$BETA" \
  --attack_init zero \
  --attack_render_edit True \
  --attack_basis_src "$BASIS_SRC"
