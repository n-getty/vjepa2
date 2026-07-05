#!/usr/bin/env bash
# Scan ALL remaining new sources for non-finite clips in ONE debug job, serially.
# Replaces the fragile per-source chain (a monitor died after surgenet_lap). Each
# source is ~2-5 min at NPROC=48, so all 7 fit well inside the 1h debug wall.
#
# Motivation: the 2B epoch-16 NaN crash (see memory nan-crash-rootcause). The
# runtime isfinite gate already makes training safe; this bakes the drop into the
# shards at rest AND gives us the ground-truth count of how many corrupt clips
# actually exist per source (4/4 sources scanned so far = ZERO — so this doubles
# as evidence for/against the baked-in-corruption hypothesis).
#
# Submit:  qsub scripts/filter_black_clips_all_pbs.sh
#
#PBS -N vjepa_blackfilter_all
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1:ncpus=104
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -uo pipefail   # NOT -e: one bad source must not abort the rest

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
RESHARD_ROOT=/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded
PYTHON=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
NPROC=48

# Remaining sources (surgenet_lap already scanned: 0 nonfinite).
DATASETS=(sitl sitl_2026 lemon heichole multibypass140 gynsurg lapgyn6_events)

mkdir -p /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-} NPROC=$NPROC datasets=${DATASETS[*]}"

for DATASET in "${DATASETS[@]}"; do
  INPUT="$RESHARD_ROOT/$DATASET"
  OUTPUT="$RESHARD_ROOT/${DATASET}_clean"
  echo "==================== $DATASET  $(date) ===================="
  NSHARDS=$("$PYTHON" -c "import glob; print(len(glob.glob('$INPUT/*.tar')))")
  echo "  input shards: $NSHARDS"
  if [[ "$NSHARDS" -eq 0 ]]; then echo "  no shards, skip"; continue; fi
  mkdir -p "$OUTPUT"

  CHUNK=$(( (NSHARDS + NPROC - 1) / NPROC ))
  pids=()
  for (( lo=0; lo<NSHARDS; lo+=CHUNK )); do
    hi=$(( lo + CHUNK )); (( hi > NSHARDS )) && hi=$NSHARDS
    log="$OUTPUT/_worker_${lo}_${hi}.log"
    "$PYTHON" "$ROOT/scripts/filter_black_clips_reshard.py" \
        --input "$INPUT" --output "$OUTPUT" \
        --min-clip-std 1.0 --fps 4 --frames-per-clip 16 \
        --shard-start "$lo" --shard-end "$hi" > "$log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
  echo "  workers done, failed=$fail"

  # Finalize + print the per-source aggregate (kept/dropped/NONFINITE).
  "$PYTHON" "$ROOT/scripts/filter_black_clips_reshard.py" \
      --input "$INPUT" --output "$OUTPUT" --finalize 2>&1 | sed 's/^/  /'
  "$PYTHON" - "$OUTPUT" <<'PYEOF'
import json, glob, sys, os
out = sys.argv[1]
nf = kept = drop = 0
for p in glob.glob(os.path.join(out, "_partial_*.json")):
    try:
        d = json.load(open(p))
        for s, v in d.get("per_shard", {}).items():
            nf += v.get("nonfinite", 0); kept += v.get("kept", 0); drop += v.get("dropped", 0)
    except Exception:
        pass
print(f"  AGG {os.path.basename(out)}: kept={kept} dropped={drop} NONFINITE={nf}")
PYEOF
done

echo "ALL DONE: $(date)"
