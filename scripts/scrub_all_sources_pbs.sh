#!/usr/bin/env bash
# Gate every YouTube-sourced training set against the crop-augmented eval ref and
# report/drop eval-leaked source videos. Runs three gates in parallel:
#   - LEMON staging  (gate-staging): move leaked source tars -> _evalleak/  [--apply]
#   - surgenet_lap   (gate-tree):    drop-list of leaked loose mp4s (report)
#   - surgenet_robotic (gate-shards): leaked source-video ids (report; re-reshard later)
# Query hashes RAW frames; the reference carries crop/mask variants.
#
# Requires eval_aug_ref.json (build_eval_aug_ref_pbs.sh).
# Env: APPLY=1 to actually move LEMON leaked tars (default dry-run report).
#
# Submit:  qsub scripts/scrub_all_sources_pbs.sh
#
#PBS -N scrub_evalleak
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
REF="$DATA/LEMON/eval_aug_ref.json"
PYTHON="$(command -v python3)"
NPROC="${NPROC:-10}"
THRESH="${THRESH:-0.10}"
APPLYFLAG=""; [[ "${APPLY:-0}" == "1" ]] && APPLYFLAG="--apply"

LEM="$DATA/surg_vid_webdataset_resharded/lemon_staging"
SR="$DATA/surg_vid_webdataset_resharded/surgenet_robotic"
SL="$DATA/surgenet_laparoscopic"

LOGD=/flare/ModCon/ngetty/logs/scrub_$$
mkdir -p /flare/ModCon/ngetty/logs "$LOGD"
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-} APPLY=${APPLY:-0} LOGD=$LOGD"
[[ -f "$REF" ]] || { echo "missing $REF" >&2; exit 1; }

gate_range () {  # $1=mode $2=srcflag $3=srcpath $4=ntot $5=extra
  local mode="$1" sf="$2" sp="$3" ntot="$4" extra="$5"
  local ch=$(( (ntot + NPROC - 1) / NPROC )) pids=() lo hi
  for (( lo=0; lo<ntot; lo+=ch )); do
    hi=$(( lo+ch )); (( hi>ntot )) && hi=$ntot
    "$PYTHON" "$ROOT/scripts/scrub_eval_leakage.py" --mode "$mode" \
        "$sf" "$sp" --eval-ref "$REF" --threshold "$THRESH" \
        --start "$lo" --end "$hi" $extra > "$LOGD/scrub_${mode}_${lo}.log" 2>&1 &
    pids+=($!)
  done
  local fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
  if (( fail>0 )); then echo "$mode failed=$fail"; tail -20 "$LOGD"/scrub_${mode}_*.log 2>/dev/null; fi
  return 0
}

# --- surgenet_lap tree (gate-tree) ---
NSL=$("$PYTHON" -c "import os;print(sum(len([f for f in fs if f.lower().endswith(('.mp4','.avi','.mkv','.mov','.m4v'))]) for _,_,fs in os.walk('$SL')))")
echo "=== surgenet_lap: $NSL clips ==="
rm -f "$DATA/surgenet_laparoscopic/_evalleak_report.json.partials"/gt_*.json 2>/dev/null || true
gate_range gate-tree --tree "$SL" "$NSL" "--out $DATA/surgenet_laparoscopic/_evalleak_report.json --partial-dir $DATA/surgenet_laparoscopic/_evalleak_report.json.partials"

# --- surgenet_robotic shards (gate-shards) ---
NSR=$("$PYTHON" -c "import glob;print(len(glob.glob('$SR/*.tar')))")
echo "=== surgenet_robotic: $NSR shards ==="
rm -f "$SR/_evalleak_report.json.partials"/gsh_*.json 2>/dev/null || true
gate_range gate-shards --shards "$SR" "$NSR" "--out $SR/_evalleak_report.json --partial-dir $SR/_evalleak_report.json.partials"

# --- LEMON staging (gate-staging) [APPLY moves tars] ---
NLE=$("$PYTHON" -c "import glob;print(len(glob.glob('$LEM/*.tar')))")
echo "=== LEMON staging: $NLE source tars (APPLY=${APPLY:-0}) ==="
rm -f "$LEM/_evalleak_report.json.partials"/gs_*.json 2>/dev/null || true
gate_range gate-staging --staging "$LEM" "$NLE" "--out $LEM/_evalleak_report.json --partial-dir $LEM/_evalleak_report.json.partials --frames-per-video 16 $APPLYFLAG"

echo "=== TALLIES ==="
"$PYTHON" - <<PYEOF
import glob, json
def tally(pdir, key, label):
    leak=tot=0; ex=[]
    for p in glob.glob(pdir+'/*.json'):
        d=json.load(open(p))
        if key=='shards':
            ls=d.get('leaked_sources',{}); leak+=len(ls); tot+=d.get('sources',0); ex+=list(ls.items())
        else:
            for r in d.get('results',[]):
                tot+=1
                if r.get('leak'): leak+=1; ex.append((r.get('tar') or r.get('file'), r.get('dup')))
    print(f"{label}: {leak}/{tot} leaked; examples: {sorted(ex,key=lambda x:-(x[1] or 0))[:5]}")
tally('$DATA/surgenet_laparoscopic/_evalleak_report.json.partials','tree','surgenet_lap')
tally('$SR/_evalleak_report.json.partials','shards','surgenet_robotic (source videos)')
tally('$LEM/_evalleak_report.json.partials','staging','LEMON')
PYEOF
echo "JOB END: $(date)"
