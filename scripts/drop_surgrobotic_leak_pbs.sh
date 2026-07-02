#!/usr/bin/env bash
# Re-reshard surgenet_robotic dropping the 40 eval-leaked source videos ->
# surgenet_robotic_clean. Serial (small: ~500 shards / 3624 clips) but on a
# compute node (login-node inline timed out). See drop_sources_reshard.py.
#
# Submit:  qsub scripts/drop_surgrobotic_leak_pbs.sh
#
#PBS -N surgrob_declean
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
SR=/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/surgenet_robotic
PYTHON="$(command -v python3)"

cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-}"
"$PYTHON" "$ROOT/scripts/drop_sources_reshard.py" \
    --input "$SR" --output "${SR}_clean" \
    --drop-json "$SR/_evalleak_sources.json" \
    --prefix surgenet_robotic --seed 0 --force

chmod 700 "${SR}_clean"; chmod 600 "${SR}_clean"/*.tar "${SR}_clean"/*.json 2>/dev/null || true
echo "JOB END: $(date)"
"$PYTHON" -c "import json;d=json.load(open('${SR}_clean/metadata.json'));print({k:d[k] for k in ['name','shard_count','sample_count']})"
cat "${SR}_clean/reshard_summary.json"
