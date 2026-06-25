#!/usr/bin/env bash
# Master sequencer for the probe campaign. Runs ONE debug job at a time so we
# never exceed the queue's 1R+1Q limit and never race for the slot. Order:
#   1. (already running) v3_e9_full cross-check chain -> just wait for it
#   2. benchmark (batch x sdpa) -> pick optimal recipe
#   3. v1_e9_full cross-check (the other ranking-critical full-data probe)
#   4. fs10 v1 epoch-match probes (v1p1_e12, v1_e19, v1_e39)
# Steps 3-4 still run at the CURRENT (bs2) recipe for comparability with the
# existing fs10 table; the benchmark result informs a SEPARATE fast re-anchor
# the human approves after seeing the numbers.
set -uo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
LOG=/flare/ModCon/ngetty/logs/probe_master_driver.log
exec >> "$LOG" 2>&1
echo "=== master driver start $(date) ==="

# Wait until I have ZERO debug-queue jobs (R or Q) before launching the next.
wait_debug_clear () {
  while qstat -u "$USER" 2>/dev/null | awk 'NR>5 && $3=="debug"{n++} END{exit (n>0)?0:1}'; do sleep 90; done
}

echo "[1] waiting for v3_e9_full cross-check chain to finish..."
wait_debug_clear
echo "[1] v3_e9_full done $(date)"

echo "[2] launching benchmark $(date)"
qsub "$ROOT/scripts/bench_probe_pbs.sh"
wait_debug_clear
echo "[2] benchmark done $(date) -- see bench log for (batch,sdpa) table"

echo "[3] launching v1_e9_full cross-check $(date)"
qsub -v PROBE_CFG="$ROOT/configs/heads/sarrarp50/full_xcheck/v1_e9_full.yaml" "$ROOT/scripts/probe_chain_debugscaling.sh"
wait_debug_clear
echo "[3] v1_e9_full done $(date)"

for tag in v1p1_e12 v1_e19 v1_e39; do
  echo "[4] launching fs10 orchestrator: $tag $(date)"
  EXPORT_QUEUE=debug PROBE_QUEUE=debug bash "$ROOT/scripts/fs10_probe_orchestrate.sh" "$tag"
  echo "[4] $tag orchestrator returned $(date)"
done
echo "=== master driver done $(date) ==="
