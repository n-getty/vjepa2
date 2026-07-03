#!/usr/bin/env bash
# 16-NODE SPIKE TEST CONTROL on torch 2.10 + CCL_OP_SYNC=0. One-shot, no chain.
# Purpose: isolate whether the fix is the TORCH VERSION (2.13) or just the
# CCL_OP_SYNC env knob. The perf REPORT flags CCL_OP_SYNC=1 as -5.5% ("kills
# overlap") and warns `module load frameworks` RE-ASSERTS =1, so it must be
# re-exported =0 AFTER the module load (REPORT Bug 2). Our production launcher
# set =1 AND loads frameworks -> doubly on. This cell tests 2.10 with =0.
#
# ISOLATION vs the pt213 test (vitG384_spiketest_pt213.sh): identical EXCEPT
#   - torch 2.10 (frameworks python, not the pt213 venv)
#   - CCL_OP_SYNC=0 (the pt213 test keeps our original =1 to change ONLY torch)
# So: pt213 test = "does newer torch fix it"; this = "does OP_SYNC=0 fix it on 2.10".
# If BOTH are clean, prefer 2.10+OP_SYNC=0 (smaller change). If only pt213 is
# clean, the torch upgrade is required.
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/vitG384_spiketest_op0.sh
#
#PBS -N vitG_spkop0
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=16
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/vitG384_cleandata.yaml
PY_STAGE=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/SPIKETEST_vitG384/op0_n16g12
RUNTIME_CFG=$ROOT/.runtime_configs/n16g12_weak/configs/vitg16_surg_vid_webdataset_single4/vitG384_cleandata.yaml
PARAMS=$CKPT_DIR/params-pretrain.yaml
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID  [torch 2.10 + OP_SYNC=0 control]"

$PY_STAGE $ROOT/scripts/prepare_runtime_config.py \
    $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 16 --weak-scale > /dev/null
$PY_STAGE - "$RUNTIME_CFG" "$PARAMS" "$CKPT_DIR" <<'PY'
import sys, yaml
src, dst, folder = sys.argv[1], sys.argv[2], sys.argv[3]
d = yaml.safe_load(open(src))
d["folder"] = folder
d["optimization"]["ipe"] = 120
d["optimization"]["epochs"] = 1
d["meta"]["save_every_freq"] = 1000000000
yaml.safe_dump(d, open(dst, "w"), sort_keys=False)
print(f"patched test params -> {dst} (ipe120, epochs1, no-save, folder={folder})")
PY

cd $ROOT
module load frameworks
echo "python: $(which python) ; torch $($PY_STAGE -c 'import torch;print(torch.__version__)' 2>/dev/null)"

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_OP_SYNC=0          # <-- the change under test (re-exported AFTER module load)
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
echo "CCL_OP_SYNC=$CCL_OP_SYNC (control cell)"

MASTER_ADDR=$(head -n1 "$PBS_NODEFILE"); export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=192
echo "MASTER_ADDR=$MASTER_ADDR WORLD_SIZE=$WORLD_SIZE"

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging ---"
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
