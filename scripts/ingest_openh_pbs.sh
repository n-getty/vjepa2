#!/usr/bin/env bash
# Ingest nvidia/PhysicalAI-Robotics-Open-H-Embodiment into the WebDataset training
# format. STREAMING per-clip download -> 512p/8fps re-encode -> pack (raw 1080p/60fps
# bytes never persist; see scripts/ingest_openh_to_staging.py). Then reshard the union.
#
# NETWORK-BOUND: every clip is pulled through the ALCF proxy, so this is one node with
# many parallel download+encode worker ranges. The whole raw set is ~3.2 TB (cmr is
# ~3.0 TB of it at 1080p/60fps); re-encode lands ~250-500 GB.
#
# IDEMPOTENT + RESUBMITTABLE: each worker range writes ONE tar
# (openh__range_<lo>_<hi>.tar) and is SKIPPED if that tar already exists. If the job
# hits walltime before finishing, just resubmit — completed ranges are skipped and only
# the remainder runs. The final reshard step runs only once all ranges exist.
#
# Submit (capacity = longest walltime; adjust -q to a free queue, see `qstat -Q`):
#   qsub scripts/ingest_openh_pbs.sh
#   NWORKERS=48 qsub -v NWORKERS=48 scripts/ingest_openh_pbs.sh
#   FINALIZE=1  qsub -v FINALIZE=1 scripts/ingest_openh_pbs.sh   # reshard only, after ranges done
#
#PBS -N openh_ingest
#PBS -A AuroraGPT
#PBS -q capacity
#PBS -l select=1:ncpus=208
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
umask 077
module load frameworks 2>/dev/null || module load frameworks

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
DATA=/flare/ModCon/ngetty/data
STAGING="$DATA/surg_vid_webdataset_resharded/openh_staging"
FINAL="$DATA/surg_vid_webdataset_resharded/openh"
PYTHON="$(command -v python3)"
NWORKERS="${NWORKERS:-32}"      # parallel download+encode ranges
FFT="${FFT:-3}"                 # x264 threads per encode; NWORKERS*FFT <= ncpus
export IMAGEIO_FFMPEG_EXE="${IMAGEIO_FFMPEG_EXE:-/home/ngetty/.local/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"

mkdir -p "$STAGING" /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-} NWORKERS=$NWORKERS FFT=$FFT"

INGEST="$ROOT/scripts/ingest_openh_to_staging.py"

reshard_union () {
  echo "=== reshard union -> $FINAL ==="
  NCLIPS=$("$PYTHON" -c "import glob,tarfile
t=0
for x in glob.glob('$STAGING/*.tar'):
    try:
        with tarfile.open(x,'r|') as tf: t+=sum(1 for m in tf if m.isfile() and m.name.endswith('.mp4'))
    except Exception: pass
print(t)")
  echo "staging clips: $NCLIPS"
  TGT=$(( NCLIPS / 16 )); (( TGT < 64 )) && TGT=64
  "$PYTHON" "$ROOT/scripts/reshard_webdataset.py" \
      --input "$STAGING" --output "$FINAL" --prefix openh \
      --target-shards "$TGT" --seed 0 --force
  chmod 700 "$FINAL" 2>/dev/null || true
  chmod 600 "$FINAL"/*.tar "$FINAL"/*.json 2>/dev/null || true
  echo "=== final metadata ==="
  "$PYTHON" -c "import json;d=json.load(open('$FINAL/metadata.json'));print({k:d[k] for k in ['name','shard_count','sample_count']})"
}

if [[ "${FINALIZE:-0}" == "1" ]]; then
  reshard_union
  echo "JOB END (finalize-only): $(date)"
  exit 0
fi

# ---- total kept-file count -> FIXED-SIZE ranges (deterministic across resubmits) ----
# Range boundaries are a function of CHUNK ONLY, never NWORKERS, so a resubmit with a
# different NWORKERS still targets the SAME ranges (openh__range_<lo>_<hi>.tar) and
# skips completed ones — no overlap, no duplicate clips. NWORKERS only bounds how many
# of those fixed ranges we run concurrently in THIS job.
NFILES=$("$PYTHON" "$INGEST" --output-staging "$STAGING" --list-only 2>/dev/null | tail -1)
CHUNK="${CHUNK:-1000}"          # files per range; FIXED (do not derive from NWORKERS)
echo "kept MP4 files: $NFILES ; fixed CHUNK=$CHUNK ; max concurrent=$NWORKERS"

# Build the list of ranges still missing their tar.
mapfile -t TODO < <("$PYTHON" -c "
import os
n=$NFILES; ch=$CHUNK; stg='$STAGING'
for lo in range(0,n,ch):
    hi=min(lo+ch,n)
    tar=os.path.join(stg,f'openh__range_{lo:06d}_{hi:06d}.tar')
    if not os.path.exists(tar): print(f'{lo} {hi}')
")
echo "ranges remaining: ${#TODO[@]}"

running=0; fail=0; pids=()
for spec in "${TODO[@]}"; do
  lo=${spec% *}; hi=${spec#* }
  "$PYTHON" "$INGEST" \
      --output-staging "$STAGING" --tmpdir "/tmp/openh_${lo}" \
      --file-start "$lo" --file-end "$hi" \
      --short-side 512 --fps 8 --crf 23 --gop 16 --ffmpeg-threads "$FFT" \
      > "$STAGING/_ingest_${lo}.log" 2>&1 &
  pids+=($!)
  running=$((running+1))
  # throttle to NWORKERS concurrent
  if (( running >= NWORKERS )); then
    if wait -n; then :; else fail=$((fail+1)); fi
    running=$((running-1))
  fi
done
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "worker ranges done this job; failed=$fail"

# ---- completeness check: are ALL fixed ranges present? ----
EXPECT=$("$PYTHON" -c "print((($NFILES)+$CHUNK-1)//$CHUNK)")
GOT=$(ls "$STAGING"/openh__range_*.tar 2>/dev/null | wc -l)
echo "range tars: $GOT / $EXPECT"
if (( GOT < EXPECT )); then
  echo "INCOMPLETE ($GOT/$EXPECT ranges) — resubmit to fill remaining ranges (idempotent), then FINALIZE=1."
  echo "=== per-range tallies so far ==="
  "$PYTHON" -c "import glob,json
ok=dl=enc=0; bi=bo=0
for p in glob.glob('$STAGING/_partial_*.json'):
    d=json.load(open(p)); ok+=d['clips_ok']; dl+=d['dl_fail']; enc+=d['enc_fail']; bi+=d['bytes_in']; bo+=d['bytes_out']
print(f'clips_ok={ok} dl_fail={dl} enc_fail={enc} size={bi/1e12:.2f}->{bo/1e12:.2f}TB')" 2>/dev/null || true
  echo "JOB END (incomplete): $(date)"
  exit 0
fi

# ---- all ranges present -> reshard ----
reshard_union
echo "=== ingest tally ==="
"$PYTHON" -c "import glob,json
ok=dl=enc=0; bi=bo=0
for p in glob.glob('$STAGING/_partial_*.json'):
    d=json.load(open(p)); ok+=d['clips_ok']; dl+=d['dl_fail']; enc+=d['enc_fail']; bi+=d['bytes_in']; bo+=d['bytes_out']
print(f'clips_ok={ok} dl_fail={dl} enc_fail={enc} raw={bi/1e12:.2f}TB enc={bo/1e12:.2f}TB ({100*bo/bi:.0f}%)')" 2>/dev/null || true
echo "JOB END: $(date)"
echo "NEXT: optional black-filter (qsub -v DATASET=openh filter_black_clips_pbs.sh), then add openh to configs."
