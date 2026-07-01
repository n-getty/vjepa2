#!/usr/bin/env bash
# 1-node 1h debug-scaling PBS job to run feature-space analysis.
#
# Submit with:
#   qsub -A ModCon -q debug-scaling -l select=1 -l walltime=01:00:00 \
#        -l filesystems=home:flare scripts/feature_analysis_pbs.sh
#
#PBS -N vjepa_feat
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
mkdir -p /flare/ModCon/ngetty/logs

cd /lus/flare/projects/ModCon/ngetty/vjepa2

module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=16
export TMPDIR=/tmp
export PYTHONFAULTHANDLER=1

# Single-process — eval is small. Pin to one tile.
export ZE_AFFINITY_MASK=0

python scripts/analyze_features.py --samples-per-dataset 30 2>&1 | grep -vE "h264|mmco|reference picture|unref short"

echo "feature analysis complete"
