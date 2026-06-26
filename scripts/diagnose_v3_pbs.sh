#!/usr/bin/env bash
#PBS -N v3_diag
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -eo pipefail
ROOT=${PBS_O_WORKDIR:-/lus/flare/projects/ModCon/ngetty/vjepa2}
cd "$ROOT"
module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TMPDIR=/tmp OMP_NUM_THREADS=16
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
# single XPU tile is enough; pin to tile 0
export ZE_AFFINITY_MASK=0.0
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
echo "=== v3 regression diagnosis $(date) ==="
$PY "$ROOT/scripts/diagnose_v3_regression.py" --n 32 --out /flare/ModCon/ngetty/probe_bench/v3_diag.json
echo "=== done $(date) ==="
