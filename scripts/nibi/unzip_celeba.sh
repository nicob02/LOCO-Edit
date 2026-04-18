#!/bin/bash
#SBATCH --job-name=unzip_celeba
#SBATCH --account=def-dennisg
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

DATA_DIR="${1:-$HOME/projects/def-dennisg/nicob0/data}"
ZIP_NAME="${2:-CelebAMask-HQ.zip}"

echo "Working in: $DATA_DIR"
cd "$DATA_DIR"

if [[ ! -f "$ZIP_NAME" ]]; then
  echo "ERROR: $DATA_DIR/$ZIP_NAME not found" >&2
  exit 1
fi

echo "Starting unzip at: $(date)"
unzip -o "$ZIP_NAME"
echo "Unzip finished at: $(date)"

echo "Contents:"
ls CelebAMask-HQ

echo "Sanity check:"
test -d CelebAMask-HQ/CelebA-HQ-img \
 && test -d CelebAMask-HQ/CelebAMask-HQ-mask-anno \
 && echo OK || { echo MISSING; exit 1; }

echo "Image count: $(ls CelebAMask-HQ/CelebA-HQ-img | wc -l)  (expect 30000)"
echo "Mask subdirs: $(ls CelebAMask-HQ/CelebAMask-HQ-mask-anno | wc -l) (expect ~15)"
