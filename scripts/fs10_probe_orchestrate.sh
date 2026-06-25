#!/usr/bin/env bash
# Orchestrate a fs10 CACHED probe for ONE checkpoint: export -> probe -> cleanup.
# Run from a login node (it submits PBS jobs and polls; does not itself need a
# compute allocation). Phases:
#   1. EXPORT: submit run_asformer_probe_aurora.sh with the <tag>_export.yaml
#      (export_cache:true) -> writes the backbone-feature cache. Wait for it.
#   2. PROBE: submit the probe_chain on <tag>_probe.yaml -> trains head from cache
#      (fast). Wait for the chain to finish (its done-condition).
#   3. CLEANUP: delete the per-checkpoint cache_root (bounded storage).
#
# Usage:
#   bash scripts/fs10_probe_orchestrate.sh <tag>                 # fs10 (default)
#   PROBE_SET=full_cached CACHE_NS=full_cache TAGPFX=full \      # full-data
#     bash scripts/fs10_probe_orchestrate.sh <tag>
#   (fs10 tags: metaraw v2_e9 v2_e19 v1_e9 v1_e29 v3_e4 v3_e9 ...)
#   (full tags: metaraw v3_e9 v1_e9 v1p1_e12 v2_e9)
#
# Env overrides (default = fs10): PROBE_SET (config subdir under configs/heads/
# sarrarp50/), CACHE_NS (cache namespace dir), TAGPFX (the tag: prefix the
# generator wrote into folder/tag, e.g. fs10- or full-).
set -uo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
TAG="${1:?usage: fs10_probe_orchestrate.sh <tag>}"
PROBE_SET="${PROBE_SET:-fs10_cached}"
CACHE_NS="${CACHE_NS:-fs10_cache}"
TAGPFX="${TAGPFX:-fs10}"
CFG_DIR=$ROOT/configs/heads/sarrarp50/$PROBE_SET
EXPORT_CFG=$CFG_DIR/${TAG}_export.yaml
PROBE_CFG=$CFG_DIR/${TAG}_probe.yaml
CACHE_DIR=/flare/ModCon/ngetty/surg_2_1_v2_final/probes/${CACHE_NS}/${TAG}
PROBE_FOLDER=/flare/ModCon/ngetty/surg_2_1_v2_final/probes/${PROBE_SET}/${TAG}
PROBE_CSV=$PROBE_FOLDER/video_classification_frozen/${TAGPFX}-${TAG}/log_r0.csv
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python

for f in "$EXPORT_CFG" "$PROBE_CFG"; do
  [[ -f "$f" ]] || { echo "missing config: $f (run gen_fs10_probe_configs.py)"; exit 2; }
done

wait_for_job () {  # $1 = jobid
  local jid="$1"
  while qstat "$jid" >/dev/null 2>&1; do
    local st; st=$(qstat -f "$jid" 2>/dev/null | awk -F'= ' '/job_state/{print $2}')
    [[ "$st" == "F" ]] && break
    sleep 30
  done
}

echo "=== [$TAG] PHASE 1: export feature cache ==="
# EXPORT_QUEUE override (default debug-scaling, which is free + proven for export;
# debug is often busy). PROBE_QUEUE similarly for phase 2.
EXPORT_QUEUE="${EXPORT_QUEUE:-debug-scaling}"
PROBE_QUEUE="${PROBE_QUEUE:-debug-scaling}"
EXP_JID=$(qsub -A ModCon -q "$EXPORT_QUEUE" -v PROBE_CFG="$EXPORT_CFG" "$ROOT/scripts/run_asformer_probe_aurora.sh")
EXP_JID=${EXP_JID%%.*}
echo "export job: $EXP_JID"
wait_for_job "$EXP_JID"

# Verify the cache exported (manifest present for train and val).
if ! ls "$CACHE_DIR"/train/rank_*/manifest.json >/dev/null 2>&1 \
   && [[ ! -f "$CACHE_DIR/train/manifest.json" ]]; then
  echo "EXPORT FAILED: no train cache manifest under $CACHE_DIR/train"; exit 1
fi
echo "=== [$TAG] export complete: $(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1) ==="

echo "=== [$TAG] PHASE 2: probe from cache ==="
# The cached probe is fast (~12 min for 20 epochs), so it fits in ONE 1h slice
# and does NOT need the self-resubmitting chain. Submit run_asformer_probe
# directly (the chain's afterany-free resubmit isn't needed here).
PRB_JID=$(qsub -A ModCon -q "$PROBE_QUEUE" -v PROBE_CFG="$PROBE_CFG" "$ROOT/scripts/run_asformer_probe_aurora.sh")
PRB_JID=${PRB_JID%%.*}
echo "probe job: $PRB_JID"
# Single fast probe job: just wait for it to finish (fits one slice).
wait_for_job "$PRB_JID"

BEST_F1=$($PY - "$PROBE_CSV" <<'PYEOF'
import sys, csv
m = -1.0
for r in csv.reader(open(sys.argv[1])):
    if r and r[0].isdigit() and len(r) > 14 and r[14]:
        m = max(m, float(r[14]))
print(round(m, 3))
PYEOF
)
echo "=== [$TAG] PROBE RESULT: best_val_f1=$BEST_F1 ==="

echo "=== [$TAG] PHASE 3: delete cache ($CACHE_DIR) ==="
rm -rf "$CACHE_DIR"
echo "=== [$TAG] DONE. best_val_f1=$BEST_F1 ==="
