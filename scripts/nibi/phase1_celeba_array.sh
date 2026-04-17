#!/bin/bash
#SBATCH --job-name=loco_p1_arr
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-19
#SBATCH --account=def-dennisg

set -euo pipefail

module --force purge
module load StdEnv/2023
module load python/3.11.5

VENV="${LOCO_VENV:-$HOME/venvs/ditcap}"
source "${VENV}/bin/activate"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

CELEBA_ROOT="${1:?Usage: sbatch [...] phase1_celeba_array.sh /path/to/CelebAMask-HQ}"

# Edit this list to your 20–50 image IDs (must have chosen semantic in annotations)
IDS=(4729 3456 2984 3638)
if [[ "$SLURM_ARRAY_TASK_ID" -ge "${#IDS[@]}" ]]; then
  echo "Array task id $SLURM_ARRAY_TASK_ID out of range; exiting."
  exit 0
fi
SAMPLE_IDX="${IDS[$SLURM_ARRAY_TASK_ID]}"
CHOOSE_SEM="${2:-l_eye}"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

cd "$REPO_ROOT/src"

python main.py \
  --sh_file_name phase1_celeba_array.sh \
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
  --run_edit_null_space_projection True \
  --choose_sem "$CHOOSE_SEM" \
  --null_space_projection True \
  --use_mask True \
  --pca_rank_null 5 \
  --pca_rank 1 \
  --vis_num 2 \
  --x_space_guidance_edit_step 1 \
  --x_space_guidance_scale 0.5 \
  --x_space_guidance_num_step 16 \
  --note "array_${SLURM_ARRAY_TASK_ID}" \
  --cache_folder "$HF_HOME"

echo "Done sample_idx=$SAMPLE_IDX"
