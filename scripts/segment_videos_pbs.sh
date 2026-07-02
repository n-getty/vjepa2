#!/usr/bin/env bash
# Segment raw GraSP + Cholec80 videos into the WebDataset clip format, then
# reshard into the final source dirs. CPU-only (PyAV stream-copy demux/remux,
# no GPU). Pipeline per dataset:
#   segment_videos_to_wds.py -> <name>_staging/ (per-source tars)
#   reshard_webdataset.py    -> <name>/        (sharded + metadata.json)
# The downstream black-clip filter (scripts/filter_black_clips_pbs.sh) is run
# SEPARATELY afterwards to make <name>_clean/ (long procedures have out-of-body
# black spans) — see that script.
#
# cholec80: seekable zip -> fan NPROC workers over disjoint video-index ranges.
# grasp:    127GB serial gzip -> ONE serial worker (cannot fan within the .tar.gz).
#
# Submit (one dataset per job):
#   qsub -v DATASET=cholec80 scripts/segment_videos_pbs.sh
#   qsub -v DATASET=grasp    scripts/segment_videos_pbs.sh
#
#PBS -N vjepa_segment
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1:ncpus=104
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail   # NOT -u: `module load frameworks` references unbound vars

# PyAV (import av) is only importable after `module load frameworks` sets up the
# env — the bare interpreter path does NOT resolve it. Load the module and use
# its python3 on PATH.
module load frameworks 2>/dev/null || module load frameworks

DATASET="${DATASET:?set -v DATASET=cholec80 or grasp}"
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
RESHARD_ROOT=/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded
INCOMING=/flare/ModCon/ngetty/data/incoming_robotic
PYTHON="$(command -v python3)"
NPROC="${NPROC:-24}"
echo "using python: $PYTHON"
"$PYTHON" -c "import av; print('PyAV', av.__version__)"

mkdir -p /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-} DATASET=$DATASET NPROC=$NPROC"

STAGING="$RESHARD_ROOT/${DATASET}_staging"
FINAL="$RESHARD_ROOT/${DATASET}"
mkdir -p "$STAGING"

case "$DATASET" in
  cholec80)
    ARCHIVE="$INCOMING/cholec80/cholec80.zip"
    NVID=$("$PYTHON" -c "import zipfile;z=zipfile.ZipFile('$ARCHIVE');print(sum(1 for n in z.namelist() if n.lower().endswith(('.mp4','.avi','.mov','.mkv')) and not n.endswith('/')))")
    echo "cholec80 videos: $NVID; fanning $NPROC workers"
    CHUNK=$(( (NVID + NPROC - 1) / NPROC ))
    pids=()
    for (( lo=0; lo<NVID; lo+=CHUNK )); do
      hi=$(( lo + CHUNK )); (( hi > NVID )) && hi=$NVID
      log="$STAGING/_worker_${lo}_${hi}.log"
      "$PYTHON" "$ROOT/scripts/segment_videos_to_wds.py" \
          --dataset cholec80 --archive "$ARCHIVE" \
          --output-staging "$STAGING" --tmpdir "/tmp/segwork_${lo}" \
          --video-start "$lo" --video-end "$hi" > "$log" 2>&1 &
      pids+=($!)
    done
    fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
    echo "cholec80 workers done, failed=$fail"; (( fail > 0 )) && { echo "see $STAGING/_worker_*.log" >&2; exit 1; }
    ;;
  grasp)
    ARCHIVE="$INCOMING/grasp/GraSP/GraSP_videos/videos.tar.gz"
    echo "grasp: single serial worker over $ARCHIVE"
    "$PYTHON" "$ROOT/scripts/segment_videos_to_wds.py" \
        --dataset grasp --archive "$ARCHIVE" \
        --output-staging "$STAGING" --tmpdir "/tmp/segwork_grasp" \
        > "$STAGING/_worker_serial.log" 2>&1
    ;;
  *) echo "unknown DATASET=$DATASET" >&2; exit 2 ;;
esac

# Reshard staging -> final. --prefix pins the metadata 'name' + shard filenames
# to the clean dataset name (reshard otherwise uses the staging dirname).
echo "resharding $STAGING -> $FINAL"
NCLIPS=$("$PYTHON" -c "import glob,tarfile,sys
tot=0
for t in glob.glob('$STAGING/*.tar'):
    with tarfile.open(t) as tf:
        tot+=sum(1 for n in tf.getnames() if n.endswith('.mp4'))
print(tot)")
echo "total clips in staging: $NCLIPS"
# ~16 clips/shard density (matches surgtoolloc); floor 32 shards.
TARGET=$(( NCLIPS / 16 )); (( TARGET < 32 )) && TARGET=32
echo "target shards: $TARGET"
"$PYTHON" "$ROOT/scripts/reshard_webdataset.py" \
    --input "$STAGING" --output "$FINAL" \
    --prefix "$DATASET" --target-shards "$TARGET" --seed 0 --force

echo "JOB END: $(date)"
echo "=== final metadata ==="
"$PYTHON" -c "import json;d=json.load(open('$FINAL/metadata.json'));print({k:d[k] for k in ['name','shard_count','sample_count']})"
echo "NOTE: run scripts/filter_black_clips_pbs.sh (pointed at $FINAL) to make ${DATASET}_clean."
