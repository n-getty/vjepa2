#!/usr/bin/env bash
# Measure how much of surgenet_laparoscopic (uncleaned YouTube clips) is already
# covered by LEMON (newer/larger/cleaned YouTube scrape). Decides whether the
# transfer+clean effort is worth it or LEMON subsumes it.
#
# Phase 1: build a LEMON pHash reference from the segmented lemon_staging tars.
# Phase 2: gate every surgenet_laparoscopic clip against it; report % distinct.
# Both phases fan workers over source-list ranges. CPU-only.
#
# Requires: lemon_staging/ (done) + surgenet_laparoscopic/ transferred to flare.
#
# Submit:
#   qsub scripts/measure_surgenetlap_overlap_pbs.sh
#
#PBS -N surglap_overlap
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=1:ncpus=104
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
umask 077
module load frameworks 2>/dev/null || module load frameworks

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
DATA=/flare/ModCon/ngetty/data
LEMON_STAGING="$DATA/surg_vid_webdataset_resharded/lemon_staging"
SURGLAP="$DATA/surgenet_laparoscopic"
REF="$DATA/LEMON/lemon_overlap_ref.json"
SUMMARY="$DATA/surgenet_laparoscopic/_overlap_vs_lemon.json"
PYTHON="$(command -v python3)"
NPROC="${NPROC:-24}"
THRESH="${THRESH:-0.10}"

mkdir -p /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-}"
[[ -d "$SURGLAP" ]] || { echo "missing $SURGLAP (transfer first)" >&2; exit 1; }

run_parallel () {  # $1=mode $2=source-flag $3=source-path $4=total-count-expr
  local mode="$1" sflag="$2" spath="$3" ntot="$4"
  local ch=$(( (ntot + NPROC - 1) / NPROC ))
  local pids=() lo hi
  for (( lo=0; lo<ntot; lo+=ch )); do
    hi=$(( lo+ch )); (( hi>ntot )) && hi=$ntot
    "$PYTHON" "$ROOT/scripts/measure_overlap.py" --mode "$mode" \
        "$sflag" "$spath" --ref "$REF" --out "$SUMMARY" \
        --frames-per-video 16 --threshold "$THRESH" \
        --start "$lo" --end "$hi" > "/tmp/ov_${mode}_${lo}.log" 2>&1 &
    pids+=($!)
  done
  local fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
  (( fail>0 )) && { echo "$mode workers failed=$fail; see /tmp/ov_${mode}_*.log" >&2; cat /tmp/ov_${mode}_*.log | tail -20; exit 1; }
}

# ---- Phase 1: build LEMON reference from staged tars ----
NLEM=$("$PYTHON" -c "import glob;print(len(glob.glob('$LEMON_STAGING/*.tar')))")
echo "=== Phase 1: LEMON ref from $NLEM staged tars ==="
rm -f "$REF" "$REF.partials"/*.json 2>/dev/null || true
run_parallel build-ref --from-staging "$LEMON_STAGING" "$NLEM"
"$PYTHON" "$ROOT/scripts/measure_overlap.py" --mode finalize-ref --ref "$REF"
chmod 600 "$REF"

# ---- Phase 2: gate surgenet_laparoscopic tree ----
NSL=$("$PYTHON" -c "import os;print(sum(len([f for f in fs if f.lower().endswith(('.mp4','.avi','.mkv','.mov','.m4v'))]) for _,_,fs in os.walk('$SURGLAP')))")
echo "=== Phase 2: gate $NSL surgenet_lap clips vs LEMON ref ==="
rm -f "$SUMMARY.partials"/*.json 2>/dev/null || true
run_parallel gate --from-tree "$SURGLAP" "$NSL"
"$PYTHON" "$ROOT/scripts/measure_overlap.py" --mode finalize-gate \
    --from-tree "$SURGLAP" --ref "$REF" --out "$SUMMARY" --threshold "$THRESH"

echo "JOB END: $(date)"
echo "=== OVERLAP SUMMARY ==="
"$PYTHON" -c "import json;print(json.dumps({k:v for k,v in json.load(open('$SUMMARY')).items() if k!='top20_most_distinct'}, indent=2))"
