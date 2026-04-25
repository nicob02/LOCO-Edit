#!/bin/bash
# =============================================================================
# Per-semantic launcher.
#
# For ONE sample index and a list of semantics (default: nose hair), submits
# the full Phase1 -> (Attack A + Attack B) -> D2 pipeline per semantic with
# SLURM afterok dependencies.
#
# Goal: check whether the "bits:4 dominates D2" finding is l_eye-specific or a
# general property of LOCO across editing semantics.
#
# Usage:
#   bash scripts/nibi/per_semantic_launcher.sh \
#        /project/6001170/nicob0/data/CelebAMask-HQ
#
#   # custom sample / semantics:
#   IDX=4729 SEMANTICS="nose hair mouth" bash scripts/nibi/per_semantic_launcher.sh \
#        /project/6001170/nicob0/data/CelebAMask-HQ
#
# Environment overrides:
#   IDX         sample index                 (default: 4729)
#   SEMANTICS   space-separated list         (default: "nose hair")
#   EPS_A       Attack-A sweep list          (default: 0.005,0.01,0.02,0.05,0.1)
#   EPS_B       Attack-B sweep list          (default: 0.004,0.008,0.016,0.031,0.063)
#   PURIFY_PLAN D2 plan                      (default: jpeg:50,75,90 blur:0.5,1.0,1.5 bits:4,6)
#   STEPS       PGD iters                    (default: 40)
# =============================================================================

set -euo pipefail

CELEBA_ROOT="${1:?Usage: per_semantic_launcher.sh /path/to/CelebAMask-HQ}"

IDX="${IDX:-4729}"
SEMANTICS="${SEMANTICS:-nose hair}"
EPS_A="${EPS_A:-0.005,0.01,0.02,0.05,0.1}"
EPS_B="${EPS_B:-0.004,0.008,0.016,0.031,0.063}"
PURIFY_PLAN="${PURIFY_PLAN:-jpeg:50,75,90 blur:0.5,1.0,1.5 bits:4,6}"
STEPS="${STEPS:-40}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

SCRIPT_P1="$REPO_ROOT/scripts/nibi/phase1_celeba_single.sh"
SCRIPT_A="$REPO_ROOT/scripts/nibi/phase2_attack_a.sh"
SCRIPT_B="$REPO_ROOT/scripts/nibi/phase2_attack_b.sh"
SCRIPT_D2="$REPO_ROOT/scripts/nibi/phase3_defense_purify.sh"

SUMMARY="$REPO_ROOT/logs/per_semantic_launcher_$(date +%Y%m%d_%H%M%S).txt"
echo "# Per-semantic launch  IDX=$IDX  SEMANTICS=\"$SEMANTICS\"  ($(date))" > "$SUMMARY"
printf "%-10s  %-10s  %-10s  %-10s  %-10s\n" "SEM" "p1_jid" "A_jid" "B_jid" "D2_jid" >> "$SUMMARY"

echo "================================================================="
echo "Launching per-semantic pipeline"
echo "  sample_idx  : $IDX"
echo "  semantics   : $SEMANTICS"
echo "  EPS_A sweep : $EPS_A"
echo "  EPS_B sweep : $EPS_B"
echo "  D2 plan     : $PURIFY_PLAN"
echo "================================================================="

for SEM in $SEMANTICS; do
  P1_NOTE="phase1_sem_${SEM}_s${IDX}"
  A_NOTE="phase2_attackA_sem_${SEM}_s${IDX}"
  B_NOTE="phase2_attackB_sem_${SEM}_s${IDX}"

  BASIS_DIR="$REPO_ROOT/src/runs/CelebA_HQ_HF-CelebA_HQ_mask-${P1_NOTE}/results/sample_idx${IDX}/basis/local_basis-0.6T-select-mask-${SEM}"
  B_SWEEP_DIR="$REPO_ROOT/src/runs/CelebA_HQ_HF-CelebA_HQ_mask-${B_NOTE}/results/sample_idx${IDX}"

  echo ""
  echo "------------------------------ semantic=${SEM} ------------------"

  P1_JID=$(sbatch --parsable \
    --job-name="loco_p1_${SEM}_s${IDX}" --time=01:00:00 \
    "$SCRIPT_P1" "$CELEBA_ROOT" "$IDX" "$SEM" 0.5 "$P1_NOTE" True)
  echo "  phase1            : JobID $P1_JID"

  A_JID=$(sbatch --parsable \
    --dependency=afterok:${P1_JID} --kill-on-invalid-dep=yes \
    --job-name="loco_p2A_${SEM}_s${IDX}" --time=02:00:00 \
    "$SCRIPT_A" "$CELEBA_ROOT" "$IDX" "$SEM" SWEEP linf "$STEPS" "$A_NOTE" "$EPS_A" "$BASIS_DIR")
  echo "  attack A (sweep)  : JobID $A_JID  (afterok:$P1_JID)"

  B_JID=$(sbatch --parsable \
    --dependency=afterok:${P1_JID} --kill-on-invalid-dep=yes \
    --job-name="loco_p2B_${SEM}_s${IDX}" --time=01:30:00 \
    "$SCRIPT_B" "$CELEBA_ROOT" "$IDX" "$SEM" SWEEP "$STEPS" "$B_NOTE" "$EPS_B" "$BASIS_DIR")
  echo "  attack B (sweep)  : JobID $B_JID  (afterok:$P1_JID)"

  D2_JID=$(sbatch --parsable \
    --dependency=afterok:${B_JID} --kill-on-invalid-dep=yes \
    --job-name="loco_p3D2_${SEM}_s${IDX}" --time=03:00:00 \
    "$SCRIPT_D2" "$CELEBA_ROOT" "$IDX" "$SEM" "$B_NOTE" "$B_SWEEP_DIR" "$PURIFY_PLAN")
  echo "  defense D2        : JobID $D2_JID  (afterok:$B_JID)"

  printf "%-10s  %-10s  %-10s  %-10s  %-10s\n" "$SEM" "$P1_JID" "$A_JID" "$B_JID" "$D2_JID" >> "$SUMMARY"
done

echo ""
echo "================================================================="
echo "All semantics submitted.   Summary -> $SUMMARY"
echo "================================================================="
