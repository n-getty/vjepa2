#!/usr/bin/env bash
# ViT-g @ 384 surgical CPT — 64-frame COOLDOWN, single long CAPACITY job (12h).
#
# Companion to vitg384_cooldown_64f_chain.sh (the 1h debug-scaling backfill).
# This is the EFFICIENT main path: one 12h capacity job runs the whole cooldown
# with the staging tax paid ONCE, instead of ~5 min staging + walltime-kill churn
# every hour. 64f is ~14.3s/iter (4x 16f) so an ipe=150 epoch is ~36 min compute;
# a 12h capacity job clears ~18 epochs in one go vs the backfill's ~1/hour.
#
# Strategy: capacity (≤128 nodes cluster-wide, can wait) is queued separately
# from debug-scaling (fast 1h backfill). BOTH point at the same CKPT_DIR and
# resume from latest.pth.tar; the shared LOCK guarantees only one trains at a
# time. The orchestrator's swap-over drains the debug-scaling chain once this
# capacity job is RUNNING.
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/vitg384_cooldown_64f_capacity.sh
#
#PBS -N vitg_cdcap
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=16
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/vitg384_cooldown_64f.yaml
RUNTIME_CFG=$ROOT/.runtime_configs/n16g12_weak/configs/vitg16_surg_vid_webdataset_single4/vitg384_cooldown_64f.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
# CKPT_DIR must match the topology-suffixed folder prepare_runtime_config.py writes
# (latest.pth.tar lands THERE). Shared with the debug-scaling cooldown chain.
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/surg_2_1_vitg384_cooldown/vitg384_cd64f_n16g12_weak
PARAMS=$CKPT_DIR/params-pretrain.yaml
LOCK=$CKPT_DIR/.training.lock
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID"

# Stage the runtime config into the ckpt dir as the authoritative params file.
# Shared single file: both this capacity job and the debug-scaling chain read it,
# so they apply the IDENTICAL ipe/epochs schedule to the shared checkpoint.
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
  echo "ViT-g cooldown complete (epoch >= num_epochs). nothing to do."
  rm -f "$LOCK"
  exit 0
fi

# Single long capacity job (no self-resubmit). Shares CKPT_DIR + LOCK with the
# debug-scaling cooldown chain; whichever holds the lock trains, the other skips.
# NOTE: do NOT set VJEPA_EXIT_AFTER_CKPT here — this is a long job, it should run
# many epochs per slice, not exit after one.

# Lock guard: never train two slices against the same latest.pth.tar.
if [[ -f "$LOCK" ]]; then
  HOLDER=$(cat "$LOCK" 2>/dev/null || echo "")
  if [[ -n "$HOLDER" ]] && qstat "$HOLDER" 2>/dev/null | awk 'NR>2{print $5}' | grep -q '^R$'; then
    echo "LOCK held by running job $HOLDER — another slice is training. Skipping."
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
# --- ViT-g optimizations (validated 2026-06-26: -32% iter-time, no loss cost) ---
export VJEPA_BF16_COMM=1
export VJEPA_DDP_BUCKET_MB=50
if [[ -f "${PBS_NODEFILE:-}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=192
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"
echo "OPT ENVS: VJEPA_BF16_COMM=$VJEPA_BF16_COMM VJEPA_DDP_BUCKET_MB=$VJEPA_DDP_BUCKET_MB"

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
