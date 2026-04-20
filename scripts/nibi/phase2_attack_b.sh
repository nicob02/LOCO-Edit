#!/bin/bash
#SBATCH --job-name=loco_p2_attackB
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
# Phase 2 - Attack B: image-space PGD on x_0.
#
# Usage (single eps):
#   sbatch scripts/nibi/phase2_attack_b.sh \
#       /project/.../CelebAMask-HQ 4729 l_eye 0.031 40 phase2_attackB_eye "" /path/to/local_basis-0.6T-select-mask-l_eye
#
# Usage (eps sweep):
#   sbatch scripts/nibi/phase2_attack_b.sh \
#       /project/.../CelebAMask-HQ 4729 l_eye SWEEP 40 phase2_attackB_eye_sweep \
#       "0.004,0.008,0.016,0.031,0.063" /path/to/local_basis-0.6T-select-mask-l_eye
#
# Args (positional):
#   $1  CELEBA_ROOT  - dataset root
#   $2  SAMPLE_IDX   - integer image index
#   $3  CHOOSE_SEM   - semantic to attack (l_eye, nose, hair, ...)
#   $4  EPS_IMG      - L_inf radius in [-1,1] units, or literal "SWEEP" for a sweep.
#                      Canonical values: 0.008 = 1/255*2, 0.031 = 4/255*2, 0.063 = 8/255*2.
#   $5  STEPS        - number of PGD iterations
#   $6  NOTE         - exp folder suffix (used by define_argparser)
#   $7  EPS_SWEEP    - comma-separated list when $4=="SWEEP", e.g. "0.004,0.008,0.016,0.031,0.063"
#   $8  BASIS_SRC    - (recommended) absolute path to a Phase-1 `local_basis-*`
#                      folder; skips the in-process GPM and the associated OOM risk.
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

CELEBA_ROOT="${1:?Usage: sbatch [...] phase2_attack_b.sh CELEBA_ROOT SAMPLE_IDX CHOOSE_SEM EPS_IMG STEPS NOTE [EPS_SWEEP] [BASIS_SRC]}"
SAMPLE_IDX="${2:-4729}"
CHOOSE_SEM="${3:-l_eye}"
EPS_IMG="${4:-0.031}"
STEPS="${5:-40}"
NOTE="${6:-phase2_attackB}"
EPS_SWEEP="${7:-}"
BASIS_SRC="${8:-}"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Default alpha = 25% of eps for L_inf PGD.
if [[ "$EPS_IMG" == "SWEEP" ]]; then
  EPS_FOR_ALPHA="$(echo "$EPS_SWEEP" | awk -F',' '{print $1}')"
  ALPHA="$(python -c "print(${EPS_FOR_ALPHA}*0.25)")"
  EPS_FLAG=""
  EPS_SWEEP_FLAG="--attack_b_eps_sweep $EPS_SWEEP"
else
  ALPHA="$(python -c "print(${EPS_IMG}*0.25)")"
  EPS_FLAG="--attack_b_eps_img $EPS_IMG"
  EPS_SWEEP_FLAG=""
fi

echo "REPO_ROOT     : $REPO_ROOT"
echo "SAMPLE_IDX    : $SAMPLE_IDX"
echo "CHOOSE_SEM    : $CHOOSE_SEM"
echo "EPS_IMG       : $EPS_IMG"
echo "EPS_SWEEP     : ${EPS_SWEEP:-<single>}"
echo "STEPS         : $STEPS"
echo "ALPHA_IMG     : $ALPHA"
echo "NOTE          : $NOTE"
echo "BASIS_SRC     : ${BASIS_SRC:-<none>}"

cd "$REPO_ROOT/src"

python tools/phase2_attack_b.py \
  --sh_file_name phase2_attack_b.sh \
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
  --run_attack_b True \
  $EPS_FLAG \
  $EPS_SWEEP_FLAG \
  --attack_b_alpha_img "$ALPHA" \
  --attack_b_steps "$STEPS" \
  --attack_init zero \
  --attack_render_edit True \
  --attack_basis_src "$BASIS_SRC"
