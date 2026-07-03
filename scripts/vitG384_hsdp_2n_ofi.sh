#!/usr/bin/env bash
#PBS -N vitG_hsdp2ofi
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2; cd $ROOT
PY_STAGE=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/SMOKE_vitG384.yaml
RUNTIME_CFG=$ROOT/.runtime_configs/n2g12_weak/configs/vitg16_surg_vid_webdataset_single4/SMOKE_vitG384.yaml
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/SPIKETEST_vitG384/hsdp_2n_ofi
PARAMS=$CKPT_DIR/params-pretrain.yaml
mkdir -p $CKPT_DIR
$PY_STAGE $ROOT/scripts/prepare_runtime_config.py $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 2 --weak-scale > /dev/null
$PY_STAGE - "$RUNTIME_CFG" "$PARAMS" "$CKPT_DIR" <<'PY'
import sys, yaml
src,dst,folder=sys.argv[1],sys.argv[2],sys.argv[3]
d=yaml.safe_load(open(src)); d["folder"]=folder
d["optimization"]["ipe"]=60; d["optimization"]["epochs"]=1; d["meta"]["save_every_freq"]=10**9
yaml.safe_dump(d,open(dst,"w"),sort_keys=False); print("folder",folder,"ipe60")
PY
module load frameworks; export PYTHONNOUSERSITE=1
source /flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH
export ZE_FLAT_DEVICE_HIERARCHY=FLAT MPICH_GPU_SUPPORT_ENABLED=1
# PRISM's VALIDATED multinode transport: launcher=none + ofi (not pmix/mpi)
export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi CCL_KVS_IFACE=hsn0
export CCL_OP_SYNC=1 CCL_WORKER_COUNT=1 CCL_ALLREDUCE=ring CCL_CHUNK_SIZE=16777216
export FI_PROVIDER=cxi TMPDIR=/tmp OMP_NUM_THREADS=16
export FI_CXI_RX_MATCH_MODE=hybrid FI_CXI_OFLOW_BUF_SIZE=8388608 FI_CXI_DEFAULT_CQ_SIZE=131072
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95 FI_MR_CACHE_MONITOR=disabled CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
export VJEPA_DIST_STRATEGY=hsdp LOCAL_WORLD_SIZE=12 FSDP_SHARDING=shard_grad_op
export MASTER_ADDR=$(head -n1 $PBS_NODEFILE) MASTER_PORT=29500 WORLD_SIZE=24
export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "=== staging (3 small datasets) ==="
mpiexec -n 2 -ppn 1 --cpu-bind none python $ROOT/scripts/stage_node_shards.py --params $PARAMS --local-root $LOCAL_DATA_ROOT --num-nodes 2 --local-world-size 12 --workers 8 2>&1 | tail -4
# watchdog
CSV="$CKPT_DIR/log_r0.csv"
( start=$(date +%s); lr=-1; lc=$start
  while sleep 20; do pgrep -f app.main_dist_aurora >/dev/null || exit 0; now=$(date +%s)
    r=0; [ -f "$CSV" ] && r=$(awk -F, 'NR>1 && $2 ~ /^[0-9]+$/' "$CSV" 2>/dev/null|wc -l)
    if [ "$r" -gt 0 ]; then [ "$r" != "$lr" ] && { lr=$r; lc=$now; }; [ $((now-lc)) -gt 240 ] && { echo "WATCHDOG STALL r=$r">&2; pkill -9 -f app.main_dist_aurora; exit 1; }
    else [ $((now-start)) -gt 420 ] && { echo "WATCHDOG NO-ITER">&2; pkill -9 -f app.main_dist_aurora; exit 1; }; fi
  done ) & WD=$!
echo "=== 2n HSDP train on small data (isolate inter-node collective) ==="
timeout 900 mpiexec -n 24 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode --fname $PARAMS --params_path $PARAMS --local_data_root $LOCAL_DATA_ROOT 2>&1 \
  | grep -viE "UserWarning|warnings.warn|comm_dev_uuids|CCL_WARN" | grep -iE "wrapped|mem:|loss:|Epoch [0-9]|itr|clip-diag #|Error|Traceback|banned|WATCHDOG" | tail -30
kill $WD 2>/dev/null
echo "2n-smoke rc=${PIPESTATUS[0]}"
