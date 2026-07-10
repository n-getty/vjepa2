#!/bin/bash -l
# Run the per-size batch calibration cells SEQUENTIALLY on ONE node (12 tiles each).
# Node-frugal: ~60 iters/cell (~2-3 min incl startup) x 12 cells ~= 30 min on 1 node.
# Each cell is a normal 12-tile run (the validated pilot path) — both calib questions
# (does bigger per_rank_bs raise clips/s? what bs OOMs?) are answered by 12-tile runs:
# bs-scaling trend is DDP-comm-invariant, OOM is per-tile memory (same at 1 or 12 tiles).
# OOM-tolerant: a cell that OOMs is logged and skipped, next cell still runs.
#
# Submit:
#   qsub -A AuroraGPT -q debug-scaling -l select=1 -l walltime=00:60:00 \
#        -l filesystems=home:flare scripts/run_calib_pbs.sh
#
#PBS -N k400_calib
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/k400_calib.log

set -o pipefail   # NOT set -u (lmod trap); NOT set -e (an OOM cell must not kill the job)

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
CFG_GLOB="$ROOT/configs/scaling/calib/calib_*.yaml"
TILES=12

cd "$ROOT"
mkdir -p /flare/ModCon/ngetty/logs
module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_OP_SYNC=1
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring
export CCL_CHUNK_SIZE=16777216
export FI_PROVIDER=cxi
export PYTHONFAULTHANDLER=1
export TMPDIR=/tmp
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"

HOST=$(head -1 "$PBS_NODEFILE")
echo "CALIB START $(date) on $HOST"

i=0
for cfg in $CFG_GLOB; do
  slug=$(basename "$cfg" .yaml)
  port=$((29700 + i)); i=$((i+1))
  echo "===== [$slug] $(date) ====="
  (
    export MASTER_ADDR="$HOST"; export MASTER_PORT=$port; export WORLD_SIZE=$TILES
    timeout 360 mpiexec --pmi=pmix -n "$TILES" -ppn "$TILES" \
        --hostfile "$PBS_NODEFILE" --cpu-bind depth --depth 16 \
        python -m app.main_dist_aurora --train_mode \
            --fname "$cfg" --params_path "$cfg"
  ) > "/flare/ModCon/ngetty/logs/calib_${slug}.log" 2>&1
  rc=$?
  # verdict from the trainer's csv, if any
  csv=$(python3 -c "import yaml;print(yaml.safe_load(open('$cfg'))['folder'])" 2>/dev/null)/log_r0.csv
  if [ -f "$csv" ]; then
    n=$(($(wc -l < "$csv") - 1))
    echo "[$slug] rc=$rc iters_logged=$n"
  else
    echo "[$slug] rc=$rc NO CSV (likely OOM/fail) — see calib_${slug}.log tail:"
    tail -3 "/flare/ModCon/ngetty/logs/calib_${slug}.log"
  fi
done

echo "CALIB DONE $(date)"
echo "=== summary ==="
python3 -m scaling.read_calib --exp-root /flare/ModCon/ngetty/experiments/scaling_calib || true
