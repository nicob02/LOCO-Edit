#!/bin/bash
#SBATCH --job-name=unzip_celeba_fast
#SBATCH --account=def-dennisg
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Extract CelebAMask-HQ to node-local NVMe ($SLURM_TMPDIR) for speed,
# then rsync the final directory to $DATA_DIR on the project filesystem.

set -euo pipefail

DATA_DIR="${1:-$HOME/projects/def-dennisg/nicob0/data}"
ZIP_NAME="${2:-CelebAMask-HQ.zip}"
OUT_NAME="${3:-CelebAMask-HQ}"

ZIP_PATH="$DATA_DIR/$ZIP_NAME"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "ERROR: zip not found at $ZIP_PATH" >&2
  exit 1
fi

echo "Node:           $SLURM_JOB_NODELIST"
echo "Local scratch:  $SLURM_TMPDIR"
echo "Project dir:    $DATA_DIR"
echo "Zip:            $ZIP_PATH"
echo "Start:          $(date)"

cd "$SLURM_TMPDIR"

echo "Unzipping to local NVMe..."
unzip -o -q "$ZIP_PATH"

if [[ ! -d "$SLURM_TMPDIR/$OUT_NAME/CelebA-HQ-img" \
   || ! -d "$SLURM_TMPDIR/$OUT_NAME/CelebAMask-HQ-mask-anno" ]]; then
  echo "ERROR: local extraction missing expected subfolders" >&2
  exit 2
fi

IMG_COUNT=$(ls "$SLURM_TMPDIR/$OUT_NAME/CelebA-HQ-img" | wc -l)
MASK_SUBDIRS=$(ls "$SLURM_TMPDIR/$OUT_NAME/CelebAMask-HQ-mask-anno" | wc -l)
echo "Local image count: $IMG_COUNT  (expect 30000)"
echo "Local mask subdirs: $MASK_SUBDIRS  (expect ~15)"

mkdir -p "$DATA_DIR"
echo "Rsync to project filesystem..."
rsync -a --info=progress2 "$SLURM_TMPDIR/$OUT_NAME/" "$DATA_DIR/$OUT_NAME/"

echo "Verification on project:"
test -d "$DATA_DIR/$OUT_NAME/CelebA-HQ-img" \
 && test -d "$DATA_DIR/$OUT_NAME/CelebAMask-HQ-mask-anno" \
 && echo OK || { echo MISSING; exit 3; }

echo "Image count: $(ls "$DATA_DIR/$OUT_NAME/CelebA-HQ-img" | wc -l)  (expect 30000)"
echo "Mask subdirs: $(ls "$DATA_DIR/$OUT_NAME/CelebAMask-HQ-mask-anno" | wc -l)  (expect ~15)"
echo "End: $(date)"
