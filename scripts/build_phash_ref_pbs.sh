#!/usr/bin/env bash
# Build the pHash reference (eval yt_robotic_chole + surgenet_robotic) that the
# LEMON segmenter gates against. CPU-only frame decode; single process (numpy
# hashing is fast, the cost is ~21k seeks). Small job.
#
# Submit:
#   qsub scripts/build_phash_ref_pbs.sh
#
#PBS -N lemon_phash_ref
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
PYTHON="$(command -v python3)"

mkdir -p /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-}"
"$PYTHON" -c "import av,cv2,numpy; print('av',av.__version__,'cv2',cv2.__version__)"

"$PYTHON" "$ROOT/scripts/build_phash_ref.py" \
    --eval-dir "$DATA/yt_chole_tool_windows" \
    --surgenet-dir "$DATA/surg_vid_webdataset_resharded/surgenet_robotic" \
    --out "$DATA/LEMON/phash_ref.json"

chmod 600 "$DATA/LEMON/phash_ref.json"
echo "JOB END: $(date)"
ls -la "$DATA/LEMON/phash_ref.json"
