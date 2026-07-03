#!/usr/bin/env bash
# 1-NODE HSDP GATE for the ViT-G 2B run. Two checks in one job:
#   (1) tests/test_hsdp_ema.py — the EMA-over-sharded-params unit test (the single
#       riskiest piece; asserts FSDP EMA == unsharded reference). FSDP needs a real
#       accelerator so this can only run on a compute node.
#   (2) A 1-node HSDP training smoke on the real 2B (SMOKE_vitG384.yaml): confirms
#       the model builds + wraps (FSDP1 _HYBRID_SHARD_ZERO2), the optimizer is built
#       AFTER wrap (use_orig_params), Meta 2B checkpoint loads under FULL_STATE_DICT,
#       iters log clean, and — crucially — max_memory_allocated DROPS vs the ~57.6GB
#       DDP baseline (sharding across 12 tiles should land ~20-30GB).
#
# VJEPA_DIST_STRATEGY=hsdp is the ONLY behavioral change vs vitG384_smoke_pt213.sh.
# torch 2.13 XPU venv (native xccl + FSDP1). No checkpoint saved.
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/vitG384_hsdp_smoke.sh
#
#PBS -N vitG_hsdp_sm
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:40:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/SMOKE_vitG384.yaml
RUNTIME_CFG=$ROOT/.runtime_configs/g12_weak/configs/vitg16_surg_vid_webdataset_single4/SMOKE_vitG384.yaml
VENV=/flare/ModCon/ngetty/venvs/torchtune-pt213-xpu
PY_STAGE=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/SMOKE_vitG384/hsdp_n1g12
PARAMS=$CKPT_DIR/params-pretrain.yaml
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID  [1-node HSDP gate]"

$PY_STAGE $ROOT/scripts/prepare_runtime_config.py \
    $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 1 --weak-scale > /dev/null
cp $RUNTIME_CFG $PARAMS
echo "staged runtime cfg -> $PARAMS"

cd $ROOT
module load frameworks
export PYTHONNOUSERSITE=1
source $VENV/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH
python -c "import torch; print('torch', torch.__version__)" 2>&1 | grep -viE "UserWarning|warn" | head -1

# -------- common Aurora / CCL env (matches the pt213 spiketests) --------
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
# allocator/MR stacking insurance (cheap; torchtune-validated)
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95
export FI_MR_CACHE_MONITOR=disabled
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
unset XPU_USM_ALLOC_SO

# -------- HSDP knobs --------
export VJEPA_DIST_STRATEGY=hsdp
export LOCAL_WORLD_SIZE=12          # ranks per node (Aurora tiles); mesh shard dim
export FSDP_SHARDING=shard_grad_op  # _HYBRID_SHARD_ZERO2 (the ZERO2 default)
# HSDP handles reduce dtype via MixedPrecision; the DDP bf16 comm hook is skipped.
unset VJEPA_BF16_COMM

MASTER_ADDR=$(head -n1 "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || hostname); export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=12
echo "HSDP smoke: WORLD_SIZE=$WORLD_SIZE LOCAL_WORLD_SIZE=$LOCAL_WORLD_SIZE FSDP_SHARDING=$FSDP_SHARDING"

# ===== CHECK 1: EMA-over-sharded-params unit test (MPI-native, 2 ranks, xccl) =====
# Must launch under mpiexec so CCL's mpi transport has its launcher; the test
# pins ZE_AFFINITY_MASK per rank from PALS_LOCAL_RANKID before torch import.
echo "=== CHECK 1: tests/test_hsdp_ema.py (mpiexec -n 2) ==="
mpiexec --pmi=pmix -n 2 -ppn 2 --cpu-bind depth --depth 16 \
    python tests/test_hsdp_ema.py 2>&1 | grep -viE "UserWarning|warnings.warn|FutureWarning" | tail -8
echo "=== CHECK 1 done ==="

# ===== CHECK 2: 1-node HSDP training smoke on the real 2B =====
export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging shards to $LOCAL_DATA_ROOT ---"
mpiexec -n 1 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS --local-root $LOCAL_DATA_ROOT \
        --num-nodes 1 --local-world-size 12 --workers 8
echo "--- staging complete ---"

echo "=== CHECK 2: 1-node HSDP train smoke ==="
mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
echo "JOB END: $(date)"

# Post-run memory check: HSDP should be well under the ~57.6GB DDP baseline.
echo "=== [mem: ...] trajectory (want << 5.76e+04 MB) ==="
grep -oE "\[mem: [0-9.e+]+\]" /flare/ModCon/ngetty/logs/${PBS_JOBID%%.*}.*.OU 2>/dev/null | tail -5 || true
