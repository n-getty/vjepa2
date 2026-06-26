#!/usr/bin/env bash
# Overnight autonomous driver: complete the v3 trajectory test.
# Waits for v3 training to produce e14 and e19 checkpoints, then runs the
# full-data cached probe for each (the decisive "does v3 hold above Meta?" test).
# Also probes them on fs10 (cheap within-version trend). One probe at a time on
# debug to respect queue limits; v3 training runs on debug-scaling/capacity.
set -uo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
CKDIR=/flare/ModCon/ngetty/checkpoints/surg_2_1_v3_lambdaoff_cleandata/lr75e6_wu2_256_n16g12_weak
LOG=/flare/ModCon/ngetty/logs/overnight_v3_trajectory.log
exec >> "$LOG" 2>&1
echo "=== overnight v3 trajectory driver start $(date) ==="

wait_debug_clear () {
  # Gate ONLY on MY probe jobs (asformer_probe / probe_chain), NOT all debug-queue
  # jobs -- a co-tenant agent runs other jobs (br_gopred, sft_compi, etc.) under
  # the SAME account, and counting those blocks us forever. Match by jobname.
  while qstat -u "$USER" 2>/dev/null | awk 'NR>5 && $3=="debug" && ($4 ~ /asformer/ || $4 ~ /probe/){n++} END{exit (n>0)?0:1}'; do sleep 120; done
}
wait_ckpt () {  # $1 = e14 / e19 ; wait until that checkpoint file exists
  local ck="$CKDIR/$1.pth.tar"
  echo "[traj] waiting for $ck ..."
  while [[ ! -f "$ck" ]]; do sleep 300; done
  sleep 60   # let the save finish flushing
  echo "[traj] $1 appeared $(date)"
}

for ep in e14 e19; do
  wait_ckpt "$ep"
  tag="v3_${ep}"
  # full-data cached (the decisive vs-Meta test)
  echo "[traj] full-data cached probe: $tag $(date)"
  wait_debug_clear
  PROBE_SET=full_cached CACHE_NS=full_cache TAGPFX=full EXPORT_QUEUE=debug PROBE_QUEUE=debug \
    bash "$ROOT/scripts/fs10_probe_orchestrate.sh" "$tag" || echo "[traj] $tag full-data FAILED"
  # fs10 cached (cheap trend point)
  echo "[traj] fs10 cached probe: $tag $(date)"
  wait_debug_clear
  EXPORT_QUEUE=debug PROBE_QUEUE=debug \
    bash "$ROOT/scripts/fs10_probe_orchestrate.sh" "$tag" || echo "[traj] $tag fs10 FAILED"
done
echo "=== overnight v3 trajectory driver done $(date) ==="
