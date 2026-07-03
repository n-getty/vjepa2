#!/usr/bin/env bash
# 1-NODE torch-2.13 GATE for the 2B ViT-G run. Purpose: confirm vjepa2 RUNS AT ALL
# on the torch 2.13 XPU venv (/flare/ModCon/ngetty/venvs/torchtune-pt213-xpu) —
# imports, model build, NATIVE xccl init (torch 2.13 has native XPU + xccl, no
# ipex / no oneccl_bindings), on-device 2B checkpoint load, iters logging clean.
#
# WHY: the perf-investigation REPORT (BaseMM_PRISM .../scaling-study/investigation
# /REPORT.md:548-551) found the intermittent backward-collapse (bwd 9-22s, our
# exact 2B spike symptom) occurred on torch 2.10 and DISAPPEARED on torch 2.13.
# Our production run is on torch 2.10.0a0. This gate is step 1 (does it even run);
# the 16n test (vitG384_spiketest_pt213.sh) is what actually validates the fix,
# since the spikes are an inter-node collective effect a 1-node run cannot show.
#
# ISOLATION: byte-identical to vitG384_smoke_debug.sh EXCEPT PY -> pt213 venv +
# PYTHONNOUSERSITE=1 (venv has include-system-site-packages, block ~/.local
# shadowing). CCL env kept the SAME as our 2.10 launcher on purpose, so the ONLY
# variable is the torch version. If native xccl init fails with this CCL block,
# the fallback is torchtune's ofi/launcher=none block (see REPORT) — try that next.
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/vitG384_smoke_pt213.sh
#
#PBS -N vitG_smoke213
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
# Stage the runtime config with the FRAMEWORKS python (venv torch not needed for yaml).
PY_STAGE=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/SMOKE_vitG384/smoke213_n1g12
PARAMS=$CKPT_DIR/params-pretrain.yaml
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID"

$PY_STAGE $ROOT/scripts/prepare_runtime_config.py \
    $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 1 --weak-scale > /dev/null
cp $RUNTIME_CFG $PARAMS
echo "staged runtime cfg -> $PARAMS"

cd $ROOT
module load frameworks
# ---- torch 2.13 venv layered on top of frameworks (module FIRST, then activate) ----
export PYTHONNOUSERSITE=1
source $VENV/bin/activate
echo "python: $(which python)"
python -c "import torch; print('torch', torch.__version__); import torch.distributed as d; print('xccl_available', hasattr(d,'is_xccl_available') and d.is_xccl_available())" 2>&1 | grep -viE "UserWarning|warn" | head

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
export VJEPA_BF16_COMM=1
export VJEPA_DDP_BUCKET_MB=50

if [[ -f "${PBS_NODEFILE:-}" ]]; then MASTER_ADDR=$(head -n1 "$PBS_NODEFILE"); else MASTER_ADDR=$(hostname); fi
export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=12
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging shards to $LOCAL_DATA_ROOT ---"
mpiexec -n 1 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS --local-root $LOCAL_DATA_ROOT \
        --num-nodes 1 --local-world-size 12 --workers 8
echo "--- staging complete ---"

mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
echo "JOB END: $(date)"
