#!/usr/bin/env bash
# Reshard LEMON staging (per-source tars from segment_lemon_pbs.sh) into the
# loader-ready WebDataset format: fine shards + metadata.json. Reuses the
# generic scripts/reshard_webdataset.py (source-aware shuffle across the many
# lemon__<youtubeId> source tars). CPU/IO only.
#
# Submit (after segment_lemon_pbs.sh finishes):
#   qsub scripts/reshard_lemon_pbs.sh
#
#PBS -N lemon_reshard
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
OUTPUT="$DATA/surg_vid_webdataset_resharded/lemon"
PYTHON="$(command -v python3)"

mkdir -p /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-}"

# Count clips in staging to size shards (~8 clips/shard, floor 512 so world_size=192
# rank-slicing works; ceil for the true count).
NCLIPS=$("$PYTHON" -c "import glob,tarfile
tot=0
for t in glob.glob('$STAGING/*.tar'):
    with tarfile.open(t,'r|') as tf:
        tot+=sum(1 for m in tf if m.isfile() and m.name.endswith('.mp4'))
print(tot)")
echo "staging clips: $NCLIPS"
TARGET=$(( NCLIPS / 8 )); (( TARGET < 512 )) && TARGET=512
echo "target shards: $TARGET"

"$PYTHON" "$ROOT/scripts/reshard_webdataset.py" \
    --input "$STAGING" --output "$OUTPUT" \
    --prefix lemon --target-shards "$TARGET" --seed 0 --force

chmod 700 "$OUTPUT"; chmod 600 "$OUTPUT"/*.tar "$OUTPUT"/*.json 2>/dev/null || true
echo "JOB END: $(date)"
"$PYTHON" -c "import json;d=json.load(open('$OUTPUT/metadata.json'));print({k:d[k] for k in ['name','shard_count','sample_count']})"
stat -c '%A %U %n' "$OUTPUT"
echo "NOTE: next run scripts/filter_black_clips_pbs.sh -v DATASET=lemon"
