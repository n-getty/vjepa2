#!/usr/bin/env bash
# Phase 2 only: gate surgenet_laparoscopic against the (already-finalized) LEMON
# overlap reference. Split out from measure_surgenetlap_overlap_pbs.sh because
# building the LEMON ref alone consumed the 1h window; the ref is now done, so
# this just runs the gate + finalize.
#
# Submit:
#   qsub scripts/gate_surgenetlap_pbs.sh
#
#PBS -N surglap_gate
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
SURGLAP="$DATA/surgenet_laparoscopic"
REF="$DATA/LEMON/lemon_overlap_ref.json"
SUMMARY="$DATA/surgenet_laparoscopic/_overlap_vs_lemon.json"
PYTHON="$(command -v python3)"
NPROC="${NPROC:-32}"
THRESH="${THRESH:-0.10}"

mkdir -p /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-}"
[[ -f "$REF" ]] || { echo "missing ref $REF" >&2; exit 1; }

NSL=$("$PYTHON" -c "import os;print(sum(len([f for f in fs if f.lower().endswith(('.mp4','.avi','.mkv','.mov','.m4v'))]) for _,_,fs in os.walk('$SURGLAP')))")
echo "gating $NSL surgenet_lap clips vs LEMON ref ($NPROC workers)"
rm -f "$SUMMARY.partials"/*.json 2>/dev/null || true
CH=$(( (NSL + NPROC - 1) / NPROC ))
pids=()
for (( lo=0; lo<NSL; lo+=CH )); do
  hi=$(( lo+CH )); (( hi>NSL )) && hi=$NSL
  "$PYTHON" "$ROOT/scripts/measure_overlap.py" --mode gate \
      --from-tree "$SURGLAP" --ref "$REF" --out "$SUMMARY" \
      --frames-per-video 16 --threshold "$THRESH" \
      --start "$lo" --end "$hi" > "/tmp/ovgate_${lo}.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
(( fail>0 )) && { echo "gate workers failed=$fail" >&2; cat /tmp/ovgate_*.log | tail -30; exit 1; }

"$PYTHON" "$ROOT/scripts/measure_overlap.py" --mode finalize-gate \
    --from-tree "$SURGLAP" --ref "$REF" --out "$SUMMARY" --threshold "$THRESH"
echo "JOB END: $(date)"
echo "=== OVERLAP SUMMARY ==="
"$PYTHON" -c "import json;print(json.dumps({k:v for k,v in json.load(open('$SUMMARY')).items() if k!='top20_most_distinct'}, indent=2))"
