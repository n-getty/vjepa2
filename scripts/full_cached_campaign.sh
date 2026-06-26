#!/usr/bin/env bash
# Run full-data CACHED probes for the trend table, one checkpoint at a time
# (each needs ~345GB cache; orchestrator deletes after its probe). Cross-checks
# the fs10 ranking against full-data. metaraw already done (71.69).
set -uo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
LOG=/flare/ModCon/ngetty/logs/full_cached_campaign.log
exec >> "$LOG" 2>&1
echo "=== full-cached campaign start $(date) ==="
wait_debug_clear () {
  while qstat -u "$USER" 2>/dev/null | awk 'NR>5 && $3=="debug"{n++} END{exit (n>0)?0:1}'; do sleep 90; done
}
for tag in v3_e9 v1_e9 v1p1_e12 v2_e9; do
  echo "[campaign] waiting for debug slot before $tag $(date)"
  wait_debug_clear
  echo "[campaign] launching full-data cached: $tag $(date)"
  PROBE_SET=full_cached CACHE_NS=full_cache TAGPFX=full EXPORT_QUEUE=debug PROBE_QUEUE=debug \
    bash "$ROOT/scripts/fs10_probe_orchestrate.sh" "$tag"
  echo "[campaign] $tag returned $(date)"
done
echo "=== full-cached campaign done $(date) ==="
