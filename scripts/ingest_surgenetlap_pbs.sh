#!/usr/bin/env bash
# Ingest surgenet_laparoscopic into the WebDataset training format.
# Two inputs merged into one staging dir, then resharded:
#   1. RAW procedure dirs (1620 variable-length 4fps videos) -> re-segment to 60s
#      clips, EVAL-GATED (re-segmentation changes boundaries so re-run the gate).
#   2. clips_1min/ (1859 already-60s 4fps clips, already eval-scrubbed) -> packed
#      into per-source staging tars as-is (passthrough, no re-segment).
# Then reshard the union to fine shards. Black filter is a SEPARATE later step.
#
# Requires eval_aug_ref.json (crop-augmented eval ref).
#
# Submit:  qsub scripts/ingest_surgenetlap_pbs.sh
#
#PBS -N surglap_ingest
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
SRC="$DATA/surgenet_laparoscopic"
STAGING="$DATA/surg_vid_webdataset_resharded/surgenet_lap_staging"
FINAL="$DATA/surg_vid_webdataset_resharded/surgenet_lap"
REF="$DATA/LEMON/eval_aug_ref.json"
PYTHON="$(command -v python3)"
NPROC="${NPROC:-24}"

mkdir -p "$STAGING" /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-}"
[[ -f "$REF" ]] || { echo "missing $REF" >&2; exit 1; }

# ---- Phase 1: segment raw procedure dirs (eval-gated), fan over video ranges ----
NV=$("$PYTHON" -c "
import os
root='$SRC'; n=0
for dp,_,fs in os.walk(root):
    top=os.path.relpath(dp,root).split(os.sep)[0]
    if top=='clips_1min' or top.startswith('_evalleak'): continue
    n+=sum(1 for f in fs if f.lower().endswith(('.mp4','.avi','.mkv','.mov','.m4v')))
print(n)")
echo "=== Phase 1: segment $NV raw procedure videos ($NPROC workers) ==="
CH=$(( (NV + NPROC - 1) / NPROC )); pids=()
for (( lo=0; lo<NV; lo+=CH )); do
  hi=$(( lo+CH )); (( hi>NV )) && hi=$NV
  "$PYTHON" "$ROOT/scripts/segment_videos_to_wds.py" --dataset surgenet_lap \
      --input-dir "$SRC" --output-staging "$STAGING" --tmpdir "/tmp/sl_${lo}" \
      --phash-ref "$REF" --phash-threshold 0.10 --phash-sample-frames 16 \
      --segment-seconds 60 --min-clip-seconds 8 \
      --video-start "$lo" --video-end "$hi" > "$STAGING/_seg_${lo}.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
if (( fail>0 )); then echo "phase1 failed=$fail"; tail -15 "$STAGING"/_seg_*.log; exit 1; fi

# ---- Phase 2: pack clips_1min (already 60s + eval-scrubbed) into staging tars ----
echo "=== Phase 2: pack clips_1min as-is ==="
"$PYTHON" "$ROOT/scripts/pack_clips_to_staging.py" \
    --input-dir "$SRC/clips_1min" --output-staging "$STAGING" \
    --dataset surgenet_lap --source-prefix clips1min --clips-per-tar 100

# ---- Phase 3: reshard union ----
echo "=== Phase 3: reshard union -> $FINAL ==="
NCLIPS=$("$PYTHON" -c "import glob,tarfile
t=0
for x in glob.glob('$STAGING/*.tar'):
    with tarfile.open(x,'r|') as tf: t+=sum(1 for m in tf if m.isfile() and m.name.endswith('.mp4'))
print(t)")
echo "staging clips: $NCLIPS"
TGT=$(( NCLIPS / 16 )); (( TGT < 64 )) && TGT=64
"$PYTHON" "$ROOT/scripts/reshard_webdataset.py" \
    --input "$STAGING" --output "$FINAL" --prefix surgenet_lap \
    --target-shards "$TGT" --seed 0 --force
chmod 700 "$FINAL"; chmod 600 "$FINAL"/*.tar "$FINAL"/*.json 2>/dev/null || true

echo "JOB END: $(date)"
"$PYTHON" -c "import json;d=json.load(open('$FINAL/metadata.json'));print({k:d[k] for k in ['name','shard_count','sample_count']})"
echo "=== phase-1 eval-gate tally ==="
"$PYTHON" -c "import glob,json
leak=corrupt=clips=0
for p in glob.glob('$STAGING/_partial_*.json'):
    d=json.load(open(p)); leak+=d.get('eval_leak',0); corrupt+=d.get('corrupt',0); clips+=d['clips']
print('proc-dir clips:',clips,'| eval-leak dropped:',leak,'| corrupt:',corrupt)"
echo "NOTE: next black-filter (qsub -v DATASET=surgenet_lap filter_black_clips_pbs.sh) then add to configs."
