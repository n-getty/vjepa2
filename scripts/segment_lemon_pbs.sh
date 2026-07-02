#!/usr/bin/env bash
# Segment LEMON (Surg-3M): a flat directory of 4194 loose YouTube .mp4 files ->
# WebDataset clip staging, with a perceptual-hash dedup gate (drops videos that
# overlap the yt_robotic_chole EVAL or the surgenet_robotic TRAIN set) and
# per-video robotic/procedure labels from labels.json in each sidecar.
#
# The directory is seekable -> fan NPROC workers over disjoint file-index ranges
# (unlike grasp's serial gzip). Each worker runs the pHash gate independently
# (the ref json is read-only, shared). CPU-only: stream-copy segmentation + a
# ~48-frame decode per video for the gate. NOT a full decode of 925GB.
#
# Requires phash_ref.json (scripts/build_phash_ref_pbs.sh) to exist first.
#
# Submit:
#   qsub scripts/segment_lemon_pbs.sh
#
#PBS -N lemon_segment
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=1:ncpus=104
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
umask 077   # owner-only outputs

module load frameworks 2>/dev/null || module load frameworks

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
DATA=/flare/ModCon/ngetty/data
INPUT="$DATA/LEMON"
STAGING="$DATA/surg_vid_webdataset_resharded/lemon_staging"
REF="$DATA/LEMON/phash_ref.json"
LABELS="$DATA/LEMON/labels.json"
PYTHON="$(command -v python3)"
NPROC="${NPROC:-24}"
THRESH="${THRESH:-0.10}"

mkdir -p "$STAGING" /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-} NPROC=$NPROC THRESH=$THRESH"
"$PYTHON" -c "import av,cv2,numpy; print('av',av.__version__)"
[[ -f "$REF" ]] || { echo "missing pHash ref $REF (run build_phash_ref_pbs.sh first)" >&2; exit 1; }

# Total video count (dash-safe python listing).
NVID=$("$PYTHON" -c "import os;print(sum(1 for f in os.listdir('$INPUT') if f.lower().endswith('.mp4')))")
echo "lemon videos: $NVID; fanning $NPROC workers"
CHUNK=$(( (NVID + NPROC - 1) / NPROC ))

pids=()
for (( lo=0; lo<NVID; lo+=CHUNK )); do
  hi=$(( lo + CHUNK )); (( hi > NVID )) && hi=$NVID
  log="$STAGING/_worker_${lo}_${hi}.log"
  "$PYTHON" "$ROOT/scripts/segment_videos_to_wds.py" \
      --dataset lemon --input-dir "$INPUT" \
      --output-staging "$STAGING" --tmpdir "/tmp/seglemon_${lo}" \
      --labels-json "$LABELS" \
      --phash-ref "$REF" --phash-threshold "$THRESH" \
      --video-start "$lo" --video-end "$hi" > "$log" 2>&1 &
  pids+=($!)
done

fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "workers done, failed=$fail"; (( fail > 0 )) && { echo "see $STAGING/_worker_*.log" >&2; exit 1; }

# Aggregate drop/corrupt tallies across all worker partials.
echo "=== segmentation tallies ==="
"$PYTHON" -c "
import glob, json
tot=dict(sources=0,clips=0,corrupt=0,eval_leak=0,train_dup=0,empty=0)
for p in glob.glob('$STAGING/_partial_*.json'):
    d=json.load(open(p))
    tot['sources']+=d['sources']; tot['clips']+=d['clips']
    tot['corrupt']+=d.get('corrupt',0); tot['eval_leak']+=d.get('eval_leak',0)
    tot['train_dup']+=d.get('train_dup',0)
    tot['empty']+=sum(1 for r in d['per_source'] if r.get('status')=='empty')
print(json.dumps(tot, indent=2))
print('EVAL-LEAK dropped (correctness-critical):', tot['eval_leak'])
print('TRAIN-DUP dropped:', tot['train_dup'])
print('CORRUPT skipped:', tot['corrupt'])
"
echo "JOB END: $(date)"
echo "NOTE: next run scripts/reshard_lemon_pbs.sh, then filter_black_clips_pbs.sh -v DATASET=lemon"
