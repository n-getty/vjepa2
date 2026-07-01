#!/usr/bin/env bash
# Reshard the under-sharded surgical WebDataset corpora for the 16-node
# Aurora V-JEPA 2.1 plan. Run on a single Aurora compute node (CPU work only).
#
# Env vars:
#   INPUT_ROOT      default /flare/ModCon/ngetty/data/surg_vid_webdataset
#   OUTPUT_ROOT     default /flare/ModCon/ngetty/data/surg_vid_webdataset_resharded
#   PYTHON          default /usr/bin/python3.10 (stdlib-only; works on login too)
#   TARGET_SHARDS   default 32
#   PARALLEL        default 4 (datasets reharded concurrently; each is single-threaded)
#   INCLUDE_SITL    if "1", also reshard sitl (18 -> 32; bigger I/O cost)
#
# Submit on Aurora with something like:
#   qsub -A AuroraGPT -q debug -l select=1 -l walltime=02:00:00 \
#        -l filesystems=home:flare -- bash scripts/reshard_all_aurora.sh
#
# Or just run interactively on a login node — it's stdlib + tar I/O only.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${INPUT_ROOT:-/flare/ModCon/ngetty/data/surg_vid_webdataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded}"
PYTHON="${PYTHON:-/usr/bin/python3.10}"
TARGET_SHARDS="${TARGET_SHARDS:-32}"
PARALLEL="${PARALLEL:-4}"
INCLUDE_SITL="${INCLUDE_SITL:-0}"

# Under-sharded datasets per the 16-node plan: singletons + surgenet_robotic.
DATASETS=(
  crcd
  endovis15
  jigsaw
  miccai_2017
  miccai_endoseg
  surgvisdom
  surgenet_robotic
)
if [[ "$INCLUDE_SITL" == "1" ]]; then
  DATASETS+=(sitl)
fi

mkdir -p "$OUTPUT_ROOT"
echo "Resharding ${#DATASETS[@]} datasets to ${TARGET_SHARDS} shards each."
echo "  input  : $INPUT_ROOT"
echo "  output : $OUTPUT_ROOT"
echo "  python : $PYTHON"
echo

pids=()
slots=0
for ds in "${DATASETS[@]}"; do
  in_dir="$INPUT_ROOT/$ds"
  out_dir="$OUTPUT_ROOT/$ds"
  if [[ ! -d "$in_dir" ]]; then
    echo "  skip $ds (input missing: $in_dir)" >&2
    continue
  fi
  log="$OUTPUT_ROOT/$ds.reshard.log"
  echo "  start $ds -> $out_dir  (log: $log)"
  "$PYTHON" "$ROOT/scripts/reshard_webdataset.py" \
      --input "$in_dir" \
      --output "$out_dir" \
      --target-shards "$TARGET_SHARDS" \
      --force \
      > "$log" 2>&1 &
  pids+=($!)
  slots=$((slots + 1))
  if (( slots >= PARALLEL )); then
    wait -n
    slots=$((slots - 1))
  fi
done

# Drain remaining.
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=$((fail + 1))
  fi
done

echo
if (( fail > 0 )); then
  echo "$fail dataset(s) failed; inspect *.reshard.log under $OUTPUT_ROOT" >&2
  exit 1
fi
echo "All ${#DATASETS[@]} datasets resharded. Per-dataset summary:"
for ds in "${DATASETS[@]}"; do
  s="$OUTPUT_ROOT/$ds/reshard_summary.json"
  [[ -f "$s" ]] && echo "  $s"
done
