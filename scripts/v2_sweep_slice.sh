#!/usr/bin/env bash
# Single-shot LR-validation slice (NO self-resubmit, NO lock). Runs ~2 epochs
# of one sweep candidate from Meta init to see if loss_pred collapses (<0.10)
# or holds the healthy plateau (~0.23) under the fixed engine.
#
# Submit with the candidate config via -v:
#   qsub -v VAL_CFG=configs/vitl16_surg_vid_webdataset_single4/sweep/val_lr75e6.yaml \
#        scripts/v2_sweep_slice.sh
#
#PBS -N v2_sweep
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=16
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
: "${VAL_CFG:?must pass -v VAL_CFG=<path to sweep config>}"
[[ "$VAL_CFG" != /* ]] && VAL_CFG="$ROOT/$VAL_CFG"
[[ -f "$VAL_CFG" ]] || { echo "config not found: $VAL_CFG" >&2; exit 2; }

module load frameworks
set -o pipefail
cd $ROOT
echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID  VAL_CFG=$VAL_CFG"

# Topology-rewrite the candidate (weak scale, 16n x 12g => global 384, bs2).
$PY $ROOT/scripts/prepare_runtime_config.py "$VAL_CFG" \
    --root $ROOT --num-gpus 12 --num-nodes 16 --weak-scale > /dev/null
REL=${VAL_CFG#$ROOT/}
PARAMS=$ROOT/.runtime_configs/n16g12_weak/$REL
echo "runtime params: $PARAMS"
$PY -c "import yaml;c=yaml.safe_load(open('$PARAMS'));o=c['optimization'];print('  lr',o['lr'],'start',o['start_lr'],'warmup',o['warmup'],'epochs',o['epochs'],'bs',c['data']['batch_size'])"

# The runtime config's folder carries the _n16g12_weak topology suffix. The
# trainer does NOT create it; every rank's CSVLogger opens <folder>/log_r<N>.csv
# and dies with FileNotFoundError if the dir is missing. Create it here (the
# self-chaining production script did this via `mkdir -p $CKPT_DIR`).
RUN_FOLDER=$($PY -c "import yaml;print(yaml.safe_load(open('$PARAMS'))['folder'])")
echo "run folder: $RUN_FOLDER"
mkdir -p "$RUN_FOLDER"

export ZE_FLAT_DEVICE_HIERARCHY=FLAT MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_KVS_CONNECTION_TIMEOUT=600 CCL_OP_SYNC=1 CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring CCL_CHUNK_SIZE=16777216
export FI_PROVIDER=cxi PYTHONFAULTHANDLER=1 TMPDIR=/tmp OMP_NUM_THREADS=16
export http_proxy="http://proxy.alcf.anl.gov:3128" https_proxy="http://proxy.alcf.anl.gov:3128" ftp_proxy="http://proxy.alcf.anl.gov:3128"
export WDS_LOCAL_SLICING=1
if [[ -f "${PBS_NODEFILE:-}" ]]; then MASTER_ADDR=$(head -n1 "$PBS_NODEFILE"); else MASTER_ADDR=$(hostname); fi
export MASTER_ADDR MASTER_PORT=29500 WORLD_SIZE=192
echo "MASTER_ADDR=$MASTER_ADDR WORLD_SIZE=$WORLD_SIZE"

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging shards to $LOCAL_DATA_ROOT ---"
mpiexec -n 16 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS --local-root $LOCAL_DATA_ROOT \
        --num-nodes 16 --local-world-size 12 --workers 8
echo "--- staging complete ---"

mpiexec --pmi=pmix -n 192 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
echo "JOB END: $(date)"
