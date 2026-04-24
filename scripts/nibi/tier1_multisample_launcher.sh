#!/bin/bash
# =============================================================================
# Tier-1 multi-sample launcher.
#
# For EACH sample ID, submits a 4-job SLURM pipeline with dependencies:
#
#     phase1  (generates local_basis + vT_null_top)
#        |
#        +-- afterok -->  phase2_attack_a  (latent PGD sweep, 5 eps)
#        |
#        +-- afterok -->  phase2_attack_b  (image PGD sweep, 5 eps)
#                                 |
#                                 +-- afterok --> phase3_defense_purify  (D2)
#
# This reproduces Figure-1 / 5 / 6 / 8 across N samples so we can report
# mean +/- std across samples (Tier-1 upgrade required by reviewers).
#
# Usage:
#   bash scripts/nibi/tier1_multisample_launcher.sh \
#        /project/6001170/nicob0/data/CelebAMask-HQ
#
#   # or with a custom id list:
#   IDS="500 1000 2000 5000 10000" bash scripts/nibi/tier1_multisample_launcher.sh \
#        /project/6001170/nicob0/data/CelebAMask-HQ
#
#   # or with a custom semantic:
#   CHOOSE_SEM=l_eye bash scripts/nibi/tier1_multisample_launcher.sh \
#        /project/6001170/nicob0/data/CelebAMask-HQ
#
# Environment overrides:
#   IDS         space-separated sample indices (default: 5 new samples)
#   CHOOSE_SEM  semantic to edit (default: l_eye)
#   EPS_A       attack-A sweep list  (default: 0.005,0.01,0.02,0.05,0.1)
#   EPS_B       attack-B sweep list  (default: 0.004,0.008,0.016,0.031,0.063)
#   PURIFY_PLAN D2 purification plan (default: jpeg:50,75,90 blur:0.5,1.0,1.5 bits:4,6)
#   STEPS       PGD iterations       (default: 40)
#
# The launcher PRINTS one line per sample with the 4 job IDs so you can
# monitor them with `sacct -j <id>` or `SQ`.
# =============================================================================

set -euo pipefail

CELEBA_ROOT="${1:?Usage: tier1_multisample_launcher.sh /path/to/CelebAMask-HQ}"

IDS="${IDS:-500 1000 2000 5000 10000}"
CHOOSE_SEM="${CHOOSE_SEM:-l_eye}"
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

for f in "$SCRIPT_P1" "$SCRIPT_A" "$SCRIPT_B" "$SCRIPT_D2"; do
  [[ -f "$f" ]] || { echo "missing: $f"; exit 1; }
done

SUMMARY_FILE="$REPO_ROOT/logs/tier1_launcher_$(date +%Y%m%d_%H%M%S).txt"
echo "# Tier-1 multisample launch summary ($(date))"                   >  "$SUMMARY_FILE"
echo "# CELEBA_ROOT=$CELEBA_ROOT  IDS=$IDS  SEM=$CHOOSE_SEM"             >> "$SUMMARY_FILE"
echo "# EPS_A=$EPS_A  EPS_B=$EPS_B  STEPS=$STEPS"                        >> "$SUMMARY_FILE"
echo "# PURIFY_PLAN='$PURIFY_PLAN'"                                      >> "$SUMMARY_FILE"
printf "%-8s  %-12s  %-12s  %-12s  %-12s\n" "SAMPLE" "p1_jobid" "A_jobid" "B_jobid" "D2_jobid" >> "$SUMMARY_FILE"

echo "=============================================================="
echo "Launching Tier-1 multisample pipeline"
echo "  IDS         : $IDS"
echo "  CHOOSE_SEM  : $CHOOSE_SEM"
echo "  EPS_A sweep : $EPS_A"
echo "  EPS_B sweep : $EPS_B"
echo "  Purify plan : $PURIFY_PLAN"
echo "  Summary     : $SUMMARY_FILE"
echo "=============================================================="

for IDX in $IDS; do
  P1_NOTE="phase1_ms_s${IDX}"
  A_NOTE="phase2_attackA_ms_s${IDX}"
  B_NOTE="phase2_attackB_ms_s${IDX}"

  BASIS_DIR="$REPO_ROOT/src/runs/CelebA_HQ_HF-CelebA_HQ_mask-${P1_NOTE}/results/sample_idx${IDX}/basis/local_basis-0.6T-select-mask-${CHOOSE_SEM}"
  B_SWEEP_DIR="$REPO_ROOT/src/runs/CelebA_HQ_HF-CelebA_HQ_mask-${B_NOTE}/results/sample_idx${IDX}"

  echo ""
  echo "------------------------------ sample_idx=$IDX ----------------------"

  # ------------- Phase 1 -------------
  P1_JID=$(sbatch --parsable \
    --job-name="loco_p1_s${IDX}" \
    --time=01:00:00 \
    "$SCRIPT_P1" "$CELEBA_ROOT" "$IDX" "$CHOOSE_SEM" 0.5 "$P1_NOTE" True)
  echo "  phase1            : JobID $P1_JID  ->  $BASIS_DIR"

  # ------------- Attack A (depends on P1) -------------
  A_JID=$(sbatch --parsable \
    --dependency=afterok:${P1_JID} --kill-on-invalid-dep=yes \
    --job-name="loco_p2A_s${IDX}" \
    --time=02:00:00 \
    "$SCRIPT_A" "$CELEBA_ROOT" "$IDX" "$CHOOSE_SEM" SWEEP linf "$STEPS" "$A_NOTE" "$EPS_A" "$BASIS_DIR")
  echo "  attack A (sweep)  : JobID $A_JID  (afterok:$P1_JID)"

  # ------------- Attack B (depends on P1) -------------
  B_JID=$(sbatch --parsable \
    --dependency=afterok:${P1_JID} --kill-on-invalid-dep=yes \
    --job-name="loco_p2B_s${IDX}" \
    --time=01:30:00 \
    "$SCRIPT_B" "$CELEBA_ROOT" "$IDX" "$CHOOSE_SEM" SWEEP "$STEPS" "$B_NOTE" "$EPS_B" "$BASIS_DIR")
  echo "  attack B (sweep)  : JobID $B_JID  (afterok:$P1_JID)"

  # ------------- Defense D2 (depends on Attack B) -------------
  D2_JID=$(sbatch --parsable \
    --dependency=afterok:${B_JID} --kill-on-invalid-dep=yes \
    --job-name="loco_p3D2_s${IDX}" \
    --time=03:00:00 \
    "$SCRIPT_D2" "$CELEBA_ROOT" "$IDX" "$CHOOSE_SEM" "$B_NOTE" "$B_SWEEP_DIR" "$PURIFY_PLAN")
  echo "  defense D2        : JobID $D2_JID  (afterok:$B_JID)"

  printf "%-8s  %-12s  %-12s  %-12s  %-12s\n" "$IDX" "$P1_JID" "$A_JID" "$B_JID" "$D2_JID" >> "$SUMMARY_FILE"
done

echo ""
echo "=============================================================="
echo "All samples submitted."
echo "Summary written to : $SUMMARY_FILE"
echo ""
echo "Estimated wall-clock (per sample, sequential deps):"
echo "  phase1 1h + max(attackA 2h, attackB 1.5h) + D2 3h  ~= 6 h"
echo "Estimated GPU-hours total (5 samples in parallel on queue):"
echo "  5 x (1 + 2 + 1.5 + 3) GPU-h = 37.5 GPU-h"
echo ""
echo "Monitor with:"
echo "  SQ                                 # your queue"
echo "  sacct -u \$USER --starttime=today   # today's job history"
echo "=============================================================="
