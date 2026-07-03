#!/usr/bin/env bash
#PBS -N vitG_hsdp_val
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:35:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2; cd $ROOT
PY_STAGE=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/SMOKE_vitG384.yaml
RUNTIME_CFG=$ROOT/.runtime_configs/g12_weak/configs/vitg16_surg_vid_webdataset_single4/SMOKE_vitG384.yaml
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/SMOKE_vitG384/hsdp_val_n1g12
PARAMS=$CKPT_DIR/params-pretrain.yaml
mkdir -p $CKPT_DIR
$PY_STAGE $ROOT/scripts/prepare_runtime_config.py $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 1 --weak-scale > /dev/null
$PY_STAGE - "$RUNTIME_CFG" "$PARAMS" "$CKPT_DIR" <<'PY'
import sys, yaml
src,dst,folder=sys.argv[1],sys.argv[2],sys.argv[3]
d=yaml.safe_load(open(src)); d["folder"]=folder+"/fresh"; d["meta"]["save_every_freq"]=10**9
yaml.safe_dump(d,open(dst,"w"),sort_keys=False); print("folder",d["folder"])
PY
mkdir -p $CKPT_DIR/fresh
module load frameworks; export PYTHONNOUSERSITE=1
source /flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH
export ZE_FLAT_DEVICE_HIERARCHY=FLAT MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_KVS_CONNECTION_TIMEOUT=600 CCL_OP_SYNC=1 CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring CCL_CHUNK_SIZE=16777216 FI_PROVIDER=cxi TMPDIR=/tmp OMP_NUM_THREADS=16
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95 FI_MR_CACHE_MONITOR=disabled CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
export VJEPA_DIST_STRATEGY=hsdp LOCAL_WORLD_SIZE=12 FSDP_SHARDING=shard_grad_op
export MASTER_ADDR=$(head -n1 $PBS_NODEFILE) MASTER_PORT=29500 WORLD_SIZE=12
export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "=== staging ==="; mpiexec -n 1 -ppn 1 --cpu-bind none python $ROOT/scripts/stage_node_shards.py --params $PARAMS --local-root $LOCAL_DATA_ROOT --num-nodes 1 --local-world-size 12 --workers 8 2>&1 | tail -3
echo "=== 1n HSDP train (validate pre-wrap load: loss must be ~0.33, matched keys high) ==="
timeout 700 mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode --fname $PARAMS --params_path $PARAMS --local_data_root $LOCAL_DATA_ROOT 2>&1 \
  | grep -viE "UserWarning|warnings.warn|FutureWarning|comm_dev_uuids|CCL_WARN" \
  | grep -iE "matched|missing.*keys|wrapped|mem:|loss:|Epoch [0-9]|Error|Traceback|banned|UR_RESULT" | tail -40
echo "val rc=${PIPESTATUS[0]}"
