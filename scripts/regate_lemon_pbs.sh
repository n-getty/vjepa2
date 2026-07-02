#!/usr/bin/env bash
# Re-gate the already-segmented LEMON staging tars against the DENSE pHash ref.
# Hashes the clips already in each staging tar (no re-decode of the raw 925GB)
# and moves matching source tars to lemon_staging/_dropped/. Fan N workers over
# disjoint tar ranges, then finalize.
#
# Requires the rebuilt dense phash_ref.json (build_phash_ref_pbs.sh) + the
# lemon_staging/ from segment_lemon_pbs.sh.
#
# Submit:
#   qsub scripts/regate_lemon_pbs.sh
#
#PBS -N lemon_regate
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
STAGING="$DATA/surg_vid_webdataset_resharded/lemon_staging"
REF="$DATA/LEMON/phash_ref.json"
PYTHON="$(command -v python3)"
NPROC="${NPROC:-24}"
THRESH="${THRESH:-0.10}"

mkdir -p /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-} NPROC=$NPROC THRESH=$THRESH"
[[ -f "$REF" ]] || { echo "missing $REF" >&2; exit 1; }
rm -f "$STAGING"/_regate_*.json

NTAR=$("$PYTHON" -c "import glob;print(len(glob.glob('$STAGING/lemon__*.tar')))")
echo "staging tars: $NTAR"
CH=$(( (NTAR + NPROC - 1) / NPROC ))
pids=()
for (( lo=0; lo<NTAR; lo+=CH )); do
  hi=$(( lo+CH )); (( hi>NTAR )) && hi=$NTAR
  "$PYTHON" "$ROOT/scripts/regate_lemon_staging.py" \
      --staging "$STAGING" --phash-ref "$REF" --threshold "$THRESH" \
      --tar-start "$lo" --tar-end "$hi" > "$STAGING/_relog_${lo}_${hi}.log" 2>&1 &
  pids+=($!)
done
fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "workers done, failed=$fail"; (( fail>0 )) && { echo "see $STAGING/_relog_*.log" >&2; exit 1; }

"$PYTHON" "$ROOT/scripts/regate_lemon_staging.py" --staging "$STAGING" --finalize
echo "JOB END: $(date)"
"$PYTHON" -c "import json;print(json.dumps(json.load(open('$STAGING/_regate_summary.json'))['eval_leak'] if False else {k:json.load(open('$STAGING/_regate_summary.json'))[k] for k in ['n_tars','eval_leak','train_dup','dropped_total']}, indent=2))"
echo "NOTE: next run scripts/reshard_lemon_pbs.sh"
