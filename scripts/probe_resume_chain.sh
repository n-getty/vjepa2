#!/usr/bin/env bash
# Self-healing RESUME CHAIN for one resumable probe run on a short-walltime queue
# (debug / debug-scaling: 1h wall, max_run=1). Runs the probe for up to ~1h, then
# — if it hasn't reached the target epoch count — RESUBMITS ITSELF to continue
# from latest.pt (resume_checkpoint:true in the YAML). Repeats until done.
#
# Why: capacity is backed up; debug-scaling has short waits but a 1h cap. The
# probe checkpoints every epoch (latest.pt) so a 1h chunk loses at most the
# in-flight epoch. At 4 nodes ~6.4 min/epoch -> ~9 epochs/chunk -> ~3 chunks for
# 20 epochs.
#
# Usage:
#   qsub -q debug-scaling -A ModCon -l select=4 -l walltime=01:00:00 \
#        -l filesystems=home:flare \
#        -v PROBE_CFG=<abs yaml>,TARGET_EPOCHS=20,CHAIN_Q=debug-scaling,CHAIN_SELECT=4 \
#        scripts/probe_resume_chain.sh
#
# Env:
#   PROBE_CFG      (req) absolute path to the probe yaml (resume_checkpoint: true)
#   TARGET_EPOCHS  (req) stop resubmitting once log_r0.csv has this many epochs
#   CHAIN_Q        queue to resubmit into (default debug-scaling)
#   CHAIN_SELECT   nodes per chunk (default = this job's node count)
#   CHAIN_WALL     walltime per chunk (default 01:00:00)
#   CHAIN_MAX      safety cap on chunks (default 8) -> never infinite-loop
#
#PBS -N probe_chain
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
export PATH=$PATH:/opt/pbs/bin
PROBE_CFG="${PROBE_CFG:?must set PROBE_CFG}"
TARGET_EPOCHS="${TARGET_EPOCHS:?must set TARGET_EPOCHS}"
CHAIN_Q="${CHAIN_Q:-debug-scaling}"
CHAIN_WALL="${CHAIN_WALL:-01:00:00}"
CHAIN_MAX="${CHAIN_MAX:-8}"
CHAIN_N="${CHAIN_N:-1}"   # which chunk this is (incremented on resubmit)
# nodes for this + future chunks
CHAIN_SELECT="${CHAIN_SELECT:-$(sort -u "${PBS_NODEFILE}" | wc -l)}"

FOLDER=$(grep -E "^folder:" "$PROBE_CFG" | awk '{print $2}')
PTAG=$(grep -E "^tag:" "$PROBE_CFG" | awk '{print $2}')
CSV="$FOLDER/video_classification_frozen/$PTAG/log_r0.csv"

epochs_done() { [ -f "$CSV" ] && grep -c '^[0-9]' "$CSV" 2>/dev/null || echo 0; }

echo "[chain $CHAIN_N/$CHAIN_MAX] $(date) cfg=$PROBE_CFG target=$TARGET_EPOCHS q=$CHAIN_Q select=$CHAIN_SELECT"
echo "[chain] CSV=$CSV  epochs_before=$(epochs_done)"

# --- already done? stop the chain.
if [ "$(epochs_done)" -ge "$TARGET_EPOCHS" ]; then
  echo "[chain] target reached before start ($(epochs_done)>=$TARGET_EPOCHS); no run needed."
  exit 0
fi

# --- resubmit the NEXT chunk FIRST (queued while this one runs) so there is no
#     gap waiting for the scheduler. It will no-op-exit if the target is hit.
NEXT=$((CHAIN_N + 1))
if [ "$NEXT" -le "$CHAIN_MAX" ]; then
  NEXTID=$(qsub -q "$CHAIN_Q" -A ModCon -l select="$CHAIN_SELECT" -l walltime="$CHAIN_WALL" \
    -l filesystems=home:flare -N "probe_chain_$NEXT" \
    -v PROBE_CFG="$PROBE_CFG",TARGET_EPOCHS="$TARGET_EPOCHS",CHAIN_Q="$CHAIN_Q",CHAIN_WALL="$CHAIN_WALL",CHAIN_MAX="$CHAIN_MAX",CHAIN_N="$NEXT",CHAIN_SELECT="$CHAIN_SELECT" \
    "$ROOT/scripts/probe_resume_chain.sh" 2>&1)
  echo "[chain] queued next chunk $NEXT: $NEXTID"
  echo "$NEXTID" > "/flare/ModCon/ngetty/logs/probe_chain_next.$PBS_JOBID"
fi

# --- run this chunk (resumes from latest.pt via resume_checkpoint:true)
echo "[chain] launching probe chunk $CHAIN_N $(date)"
PROBE_CFG="$PROBE_CFG" bash "$ROOT/scripts/run_asformer_probe_aurora.sh"
RC=$?
echo "[chain] chunk $CHAIN_N probe rc=$RC epochs_after=$(epochs_done) $(date)"

# --- if THIS chunk already finished the job, cancel the queued next chunk.
if [ "$(epochs_done)" -ge "$TARGET_EPOCHS" ]; then
  echo "[chain] TARGET REACHED ($(epochs_done)>=$TARGET_EPOCHS)."
  if [ -n "${NEXTID:-}" ]; then
    NID=$(echo "$NEXTID" | grep -oE '^[0-9]+' )
    [ -n "$NID" ] && qdel "$NID" 2>/dev/null && echo "[chain] cancelled unneeded next chunk $NID"
  fi
fi
echo "[chain $CHAIN_N] DONE $(date)"
