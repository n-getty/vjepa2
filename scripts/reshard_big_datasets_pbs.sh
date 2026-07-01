#!/usr/bin/env bash
# Reshard the BIG datasets (sitl/surgtoolloc2022/surgvu24/kinetics400/surgenet_robotic)
# into many small shards so per-node-disjoint staging works at world_size=192.
#
# Why this exists:
#   At 16 nodes x 12 ranks (world_size=192) the loader does urls[rank::192].
#   Datasets with shard_count < 192 leave most ranks with empty slices unless
#   we fall back to per-node full replication (658 GB to /tmp). Resharding
#   the three giants to ~150MB shards puts each over 1000 shards, making
#   disjoint per-node staging fit in ~42 GB/node.
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/reshard_big_datasets_pbs.sh
#
#PBS -N vjepa_reshard_big
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
INPUT_ROOT=/flare/ModCon/ngetty/data/surg_vid_webdataset
OUTPUT_ROOT=/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded
PYTHON=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python

mkdir -p "$OUTPUT_ROOT"
mkdir -p /flare/ModCon/ngetty/logs

cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-}"
echo "  input  : $INPUT_ROOT"
echo "  output : $OUTPUT_ROOT"
echo "  python : $PYTHON"
echo

# (dataset, target_shards). Targets sized to give per-node disjoint slices of
# ~150 MB each (target >= 1024 for the giants so 192-rank slicing is meaningful).
declare -a JOBS=(
  "kinetics400        500"
  "surgenet_robotic   500"
  "sitl               600"
  "surgtoolloc2022    1500"
  "surgvu24           2000"
)

# Run all 5 in parallel - they're independent and Python tar I/O is mostly
# bound by single-stream throughput, so parallelism helps even on one node.
pids=()
for spec in "${JOBS[@]}"; do
  ds=$(echo "$spec" | awk '{print $1}')
  tgt=$(echo "$spec" | awk '{print $2}')
  in_dir="$INPUT_ROOT/$ds"
  out_dir="$OUTPUT_ROOT/$ds"
  log="$OUTPUT_ROOT/$ds.reshard.log"
  if [[ ! -d "$in_dir" ]]; then
    echo "  SKIP $ds (input missing: $in_dir)" >&2
    continue
  fi
  echo "  start $ds -> $out_dir target=$tgt (log: $log)"
  "$PYTHON" "$ROOT/scripts/reshard_webdataset.py" \
      --input "$in_dir" --output "$out_dir" \
      --target-shards "$tgt" --force > "$log" 2>&1 &
  pids+=($!)
done

# Drain.
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=$((fail + 1))
  fi
done

echo
echo "JOB END: $(date)  failed=$fail"
if (( fail > 0 )); then
  echo "Inspect *.reshard.log under $OUTPUT_ROOT" >&2
  exit 1
fi

# Summary: confirm each output has a metadata.json with the expected shard count.
echo "=== summary ==="
for spec in "${JOBS[@]}"; do
  ds=$(echo "$spec" | awk '{print $1}')
  mf="$OUTPUT_ROOT/$ds/metadata.json"
  if [[ -f "$mf" ]]; then
    sc=$("$PYTHON" -c "import json; print(json.load(open('$mf')).get('shard_count'))" 2>/dev/null || echo "?")
    echo "  [OK]   $ds: shard_count=$sc"
  else
    echo "  [MISS] $ds (no metadata.json)"
  fi
done
