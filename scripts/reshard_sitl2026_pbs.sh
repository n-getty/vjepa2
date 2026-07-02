#!/usr/bin/env bash
# Reshard the raw Globus-delivered sitl_2026 into the loader-ready WebDataset
# format. The source is 46 huge per-rank shards (~5.4GB each,
# sitl_2026-rNN-NNNNNN.tar) with NO metadata.json and keys of the form
# `v2_videoNNNN__<clip>__segNNN`. Two problems that this pass fixes:
#   1. 46 shards < world_size(192) trips the loader's degenerate "every rank
#      reads all shards" path (src/datasets/webdataset.py:504) -> 192 ranks
#      hammer 46 5GB files. Reshard to ~512 fine shards (~8 clips each) so the
#      rank-slice (webdataset.py:519) gives disjoint coverage.
#   2. No metadata.json. reshard_webdataset.py writes one.
# The `__segNNN` keys don't match SOURCE_VIDEO_RE (_clip_N), so they fall into
# `__unknown__` and are distributed round-robin -- harmless: clips still shuffle
# evenly across shards and the loader keys off the member stem, not the regex.
#
# CPU/IO only (tar demux/remux, no decode, no GPU). Single streaming process.
#
# Submit:
#   qsub scripts/reshard_sitl2026_pbs.sh
#
#PBS -N sitl2026_reshard
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1:ncpus=104
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
umask 077   # owner-only outputs (protection requirement)

module load frameworks 2>/dev/null || module load frameworks

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
DATA=/flare/ModCon/ngetty/data
INPUT="$DATA/sitl_2026"
OUTPUT="$DATA/surg_vid_webdataset_resharded/sitl_2026"
PYTHON="$(command -v python3)"
TARGET=512

mkdir -p /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-}"
echo "python: $PYTHON"
echo "input : $INPUT  ($(ls "$INPUT"/*.tar | wc -l) shards)"
echo "output: $OUTPUT  (target $TARGET shards)"

"$PYTHON" "$ROOT/scripts/reshard_webdataset.py" \
    --input "$INPUT" --output "$OUTPUT" \
    --prefix sitl_2026 --target-shards "$TARGET" --seed 0 --force

# Enforce owner-only on the output (umask handles new files, be explicit anyway).
chmod 700 "$OUTPUT"
chmod 600 "$OUTPUT"/*.tar "$OUTPUT"/*.json 2>/dev/null || true

echo "JOB END: $(date)"
echo "=== final metadata ==="
"$PYTHON" -c "import json;d=json.load(open('$OUTPUT/metadata.json'));print({k:d[k] for k in ['name','shard_count','sample_count']})"
echo "=== output perms (must be owner-only) ==="
stat -c '%A %U %n' "$OUTPUT"; stat -c '%A %n' "$OUTPUT"/sitl_2026-000000.tar
