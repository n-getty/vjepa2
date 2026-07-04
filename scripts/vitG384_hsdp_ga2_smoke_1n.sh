#!/usr/bin/env bash
# 1-NODE ga=2 SQUEEZE-FIX SMOKE GATE (pre-registered §4d step 2, before 16n grad_accum).
#
# The 16n env-diff (§4e, job 8643398) proved the residual is FABRIC CONTENTION (cohort
# spikes + FLAT l0-free/l0-ext), so the legitimate lever is gradient accumulation
# (halve inter-node AllReduce frequency). But ga=2 @ bs=2 -> micro-bs=1, which historically
# tripped the weight_distance_loss d_ij.unsqueeze(2) bs>=2 landmine. The squeeze(1) fix
# (app/vjepa_2_1/models/utils/masks_dist.py:81) is in place; THIS smoke verifies it runs
# end-to-end before ga=2 touches a 16n job.
#
# SMOKE_vitG384.yaml has bs=2 + weight_distance_loss:true + ipe40 -> ga=2 exercises exactly
# the micro-bs=1 path. NOTE: our trainer's accum path calls encoder/predictor .no_sync()
# on the non-final microbatch (train.py:1017) — that IS PRISM's FSDP_NO_SYNC_ACCUM behavior;
# there is no separate env to set. VJEPA_GRAD_ACCUM=2 is the only knob.
#
# PASS = iters log clean through ipe40 with grad_accum=2 (no d_ij IndexError, no NaN),
# loss tracks the ga=1 smoke, mem still sharded (<< 57.6GB DDP baseline).
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/vitG384_hsdp_ga2_smoke_1n.sh
#
#PBS -N vitG_ga2_sm
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
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/SMOKE_vitG384/ga2_n1g12
PARAMS=$CKPT_DIR/params-pretrain.yaml
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs
# trainer APPENDS to per-rank CSVs; clear stale rows from a prior run in this CKPT_DIR
rm -f $CKPT_DIR/log_r*.csv

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID  [1-node ga=2 squeeze-fix smoke]"

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

# -------- common Aurora / CCL env (1-node: pmix/mpi transport is fine at 1 node) --------
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

# -------- HSDP knobs --------
export VJEPA_DIST_STRATEGY=hsdp
export LOCAL_WORLD_SIZE=12          # ranks per node (Aurora tiles); mesh shard dim
export FSDP_SHARDING=shard_grad_op  # _HYBRID_SHARD_ZERO2 (the ZERO2 default)
unset VJEPA_BF16_COMM

# -------- THE LEVER UNDER TEST --------
# ga=2 @ bs=2 -> micro-bs=1; exercises the weight_distance_loss squeeze(1) fix.
export VJEPA_GRAD_ACCUM=2

MASTER_ADDR=$(head -n1 "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || hostname); export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=12
echo "ga2 smoke: WORLD_SIZE=$WORLD_SIZE GRAD_ACCUM=$VJEPA_GRAD_ACCUM FSDP_SHARDING=$FSDP_SHARDING"

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging shards to $LOCAL_DATA_ROOT ---"
mpiexec -n 1 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS --local-root $LOCAL_DATA_ROOT \
        --num-nodes 1 --local-world-size 12 --workers 8
echo "--- staging complete ---"

echo "=== 1-node HSDP + ga=2 train smoke ==="
mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
echo "JOB END: $(date)"

# ---- VERDICT: iters logged + grad-accum engaged + mem sharded ----
CSV="$CKPT_DIR/log_r0.csv"
OU=$(ls -t /flare/ModCon/ngetty/logs/${PBS_JOBID%%.*}.*.OU 2>/dev/null | head -1)
echo "=== grad-accum engaged? (want 'Gradient accumulation ON: grad_accum=2') ==="
[ -n "$OU" ] && grep -m1 "Gradient accumulation ON" "$OU" || echo "(log line not found — check OU)"
echo "=== iters logged (want ~40, no crash) ==="
[ -f "$CSV" ] && echo "rank0 CSV rows: $(($(wc -l < "$CSV")-1))" || echo "NO CSV — crashed before first iter"
echo "=== any d_ij / IndexError / NaN? (want none) ==="
[ -n "$OU" ] && grep -cE "IndexError|d_ij|nan|NaN|Traceback" "$OU" || true
echo "=== [mem: ...] trajectory (want << 5.76e+04 MB) ==="
[ -n "$OU" ] && grep -oE "\[mem: [0-9.e+]+\]" "$OU" | tail -5 || true
