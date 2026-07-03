#!/usr/bin/env bash
# Reshard the 5 datasets that have <192 shards so 16n x 12-rank (world_size=192)
# per-node-disjoint staging fits /tmp. These were added this session and only
# resharded to a handful of shards; at 192 ranks the loader replicates any
# <192-shard set FULLY onto every node -> /tmp overflow (564 GiB > 503 GiB) that
# crashed job 8641339. Bringing each to >=192 shards makes staging slice to ~1/16
# per node instead.
#
# SAFETY (CRITICAL): the non-resharded SOURCE dirs for these 5 are GONE — the
# current *_resharded/<ds> dir is the ONLY copy. So this does NOT reshard in
# place (reshard_webdataset.py --force deletes output tars first). It reads the
# current dir, writes a NEW temp dir, verifies sample_count is unchanged, then
# atomically swaps: <ds> -> <ds>_bak_under192, temp -> <ds>. The backup is kept
# (delete manually once the training run confirms the new shards load).
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/reshard_under192_pbs.sh
#
#PBS -N vjepa_reshard_u192
#PBS -A ModCon
# debug-scaling accepts 1-node jobs on Aurora and usually drains faster than
# debug; use it to dodge debug backlog. (Either queue works for this 1n/1h job.)
#PBS -q debug-scaling
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
DATA=/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded
PYTHON=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python

mkdir -p /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-}"

# (dataset, target_shards). Target 256 (>192) so 192-rank slicing is meaningful;
# small sets clamp down via --min-samples-per-shard but all stay >=192:
#   grasp 1988/cholec80 2916/heichole 894/gynsurg 3053/lapgyn6_events 2155 samples.
#   heichole clamps to ~223 (894/4) which is still >192. --min-samples-per-shard 3
#   keeps a safety margin (894/3=298) so no set drops under 192.
declare -a JOBS=(
  "grasp           256"
  "cholec80        256"
  "heichole        256"
  "gynsurg         256"
  "lapgyn6_events  256"
)

# --- Reshard current -> temp (read-only on the source dir; nothing destroyed). ---
pids=()
for spec in "${JOBS[@]}"; do
  ds=$(echo "$spec" | awk '{print $1}')
  tgt=$(echo "$spec" | awk '{print $2}')
  in_dir="$DATA/$ds"
  tmp_dir="$DATA/${ds}_reshard192_tmp"
  log="$DATA/${ds}.reshard_u192.log"
  if [[ ! -d "$in_dir" ]]; then echo "  SKIP $ds (missing: $in_dir)" >&2; continue; fi
  rm -rf "$tmp_dir"
  echo "  start $ds ($in_dir) -> $tmp_dir target=$tgt (log: $log)"
  "$PYTHON" "$ROOT/scripts/reshard_webdataset.py" \
      --input "$in_dir" --output "$tmp_dir" \
      --target-shards "$tgt" --min-samples-per-shard 3 --seed 0 > "$log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
if (( fail > 0 )); then
  echo "RESHARD FAILED (failed=$fail). Source dirs UNTOUCHED. Inspect *.reshard_u192.log" >&2
  exit 1
fi

# --- Verify + atomic swap. Only swap a dataset whose sample_count is unchanged
#     AND whose new shard_count >=192. Any mismatch => leave that dataset as-is. ---
echo "=== verify + swap ==="
swap_fail=0
for spec in "${JOBS[@]}"; do
  ds=$(echo "$spec" | awk '{print $1}')
  in_dir="$DATA/$ds"
  tmp_dir="$DATA/${ds}_reshard192_tmp"
  [[ -d "$tmp_dir" ]] || { echo "  [MISS] $ds: no temp dir"; swap_fail=$((swap_fail+1)); continue; }

  read -r old_n new_n new_sh < <("$PYTHON" - "$in_dir/metadata.json" "$tmp_dir/metadata.json" <<'PY'
import json, sys
old = json.load(open(sys.argv[1])); new = json.load(open(sys.argv[2]))
print(old["sample_count"], new["sample_count"], new["shard_count"])
PY
)
  if [[ "$old_n" != "$new_n" ]]; then
    echo "  [FAIL] $ds: sample_count $old_n -> $new_n (MISMATCH) — NOT swapping, source kept" >&2
    swap_fail=$((swap_fail+1)); continue
  fi
  if (( new_sh < 192 )); then
    echo "  [FAIL] $ds: new shard_count $new_sh < 192 — NOT swapping, source kept" >&2
    swap_fail=$((swap_fail+1)); continue
  fi
  # Atomic-ish swap on the same filesystem (mv is rename()).
  bak="$DATA/${ds}_bak_under192"
  rm -rf "$bak"
  mv "$in_dir" "$bak"
  mv "$tmp_dir" "$in_dir"
  echo "  [OK]   $ds: samples=$new_n shards $($PYTHON -c "import json;print(json.load(open('$bak/metadata.json'))['shard_count'])") -> $new_sh (backup: $bak)"
done

echo
echo "JOB END: $(date)  reshard_fail=$fail swap_fail=$swap_fail"
if (( swap_fail > 0 )); then
  echo "Some datasets not swapped — do NOT launch training until all 5 are >=192 shards." >&2
  exit 1
fi
echo "All 5 resharded to >=192 shards and swapped in. Backups at *_bak_under192 (delete after training confirms)."
