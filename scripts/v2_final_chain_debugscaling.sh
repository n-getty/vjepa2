#!/usr/bin/env bash
# Self-resubmitting clean-test pretraining loop, 1h walltime on debug-scaling.
#
# Runs the surg_2_1_v2 "clean bug-fix" experiment: the EXACT regression recipe
# (lr=5.25e-4 constant, lambda_value_vid=0.5 progressive, bs=2, global 384) but
# with the SDPA layout fix + temperature-0.5 dataset mixing, from the Meta
# distilled init. Tests whether the downstream regression still happens once the
# attention/data bugs are fixed.
#
# Capacity queue was backed up (~750 nodes queued); debug-scaling backfills 1h
# jobs faster and is a SEPARATE per-user slot (max_run=1). This chain
# self-resubmits via afterany so it makes continuous progress 1h at a time;
# the run is resumable (load_checkpoint: true, save_every_freq=1000 iters).
#
# Submit first instance with:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/v2_final_chain_debugscaling.sh
#
#PBS -N v2_final_chain
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=16
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
BASE_CFG=$ROOT/configs/vitl16_surg_vid_webdataset_single4/v2_final_lr75e6.yaml
RUNTIME_CFG=$ROOT/.runtime_configs/n16g12_weak/configs/vitl16_surg_vid_webdataset_single4/v2_final_lr75e6.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/surg_2_1_v2_final/lr75e6_wu2_decay_n16g12_weak
PARAMS=$CKPT_DIR/params-pretrain.yaml
SELF=$ROOT/scripts/v2_final_chain_debugscaling.sh
LOCK=$CKPT_DIR/.training.lock
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID"

# Stage the runtime config into the ckpt dir as the authoritative params file
# (so the topology-rewritten config is what every chained slice reads).
if [[ ! -f "$PARAMS" ]]; then
  echo "Staging runtime cfg -> $PARAMS"
  $PY $ROOT/scripts/prepare_runtime_config.py \
    $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 16 --weak-scale > /dev/null
  cp $RUNTIME_CFG $PARAMS
fi

NUM_EPOCHS=$($PY -c "import yaml; print(yaml.safe_load(open('$PARAMS'))['optimization']['epochs'])")
CURRENT_EPOCH=0
if [[ -f $CKPT_DIR/latest.pth.tar ]]; then
  CURRENT_EPOCH=$($PY -c "import torch; print(torch.load('$CKPT_DIR/latest.pth.tar', map_location='cpu', weights_only=False).get('epoch', 0))" 2>/dev/null || echo 0)
fi
echo "progress: epoch $CURRENT_EPOCH / $NUM_EPOCHS"
if (( CURRENT_EPOCH >= NUM_EPOCHS )); then
  echo "clean-test complete (epoch >= num_epochs). chain stops."
  rm -f "$LOCK"
  exit 0
fi

# Keep the chain alive: resubmit a successor unless one is already queued.
# NOTE: do NOT use `-W depend=afterany:$PBS_JOBID`. On Aurora PBS, when a slice
# is killed at walltime (Exit_status -29) the dependent does NOT auto-release —
# it stays stuck in system-hold (Hold_Types=s), which the user cannot qrls, and
# the chain stalls (observed 2026-06-24, job 8558504). Serialization is already
# guaranteed by the LOCK guard below, so submit the successor dependency-free:
# if it starts while this slice is still training, it sees the lock, skips, and
# rechains. The successor sits queued and backfills after this slice ends.
DS_QUEUED=$(qstat -u $USER 2>/dev/null | awk '$3=="debug-sca" && $10=="Q"' | wc -l)
if (( DS_QUEUED >= 1 )); then
  echo "skip resubmit: $DS_QUEUED debug-scaling job already queued"
else
  NEXT_JOB=$(qsub $SELF)
  echo "Chained next job (no dependency): $NEXT_JOB"
fi

# Lock guard: never train two slices against the same latest.pth.tar.
if [[ -f "$LOCK" ]]; then
  HOLDER=$(cat "$LOCK" 2>/dev/null || echo "")
  if [[ -n "$HOLDER" ]] && qstat "$HOLDER" 2>/dev/null | awk 'NR>2{print $5}' | grep -q '^R$'; then
    echo "LOCK held by running job $HOLDER — another chain slice is training."
    echo "JOB END (skipped): $(date)"
    exit 0
  else
    echo "stale lock from $HOLDER (not running) — taking over."
  fi
fi
echo "$PBS_JOBID" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd $ROOT
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
export OMP_NUM_THREADS=16
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export WDS_LOCAL_SLICING=1
if [[ -f "${PBS_NODEFILE:-}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=192
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging shards to $LOCAL_DATA_ROOT (per-node disjoint) ---"
mpiexec -n 16 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS \
        --local-root $LOCAL_DATA_ROOT \
        --num-nodes 16 --local-world-size 12 --workers 8
echo "--- staging complete ---"

mpiexec --pmi=pmix -n 192 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
echo "JOB END: $(date)"
