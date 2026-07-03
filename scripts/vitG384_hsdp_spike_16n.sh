#!/usr/bin/env bash
# 16-NODE HSDP VERIFICATION — the no-wedge gate before any long launch.
#
# Under DDP the 2B backward-ms climbs monotonically (13s -> 328s by iter ~42) as
# CCL external memory starves the 90%-full tile. HSDP shards params/grads/optimizer
# across the 12 intra-node tiles (~90% -> ~30% occupancy), restoring L0 headroom.
# PASS = per-rank backward-ms stays FLAT across all 192 ranks through ipe120
# (no monotonic climb), and [mem:] is well under the ~57.6GB DDP baseline.
#
# ISOLATION vs the DDP wedge run (vitG384_spiketest_C_pt213_nohook.sh): same torch
# 2.13 venv, same CCL env, same config — the ONLY change is VJEPA_DIST_STRATEGY=hsdp
# (+ LOCAL_WORLD_SIZE, FSDP_SHARDING, allocator stacking env). ipe120, no save.
#
# Run vitG384_hsdp_smoke.sh (1-node gate incl. the EMA unit test) FIRST.
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/vitG384_hsdp_spike_16n.sh
#
#PBS -N vitG_hsdp16
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
VENV=/flare/ModCon/ngetty/venvs/torchtune-pt213-xpu
PY_STAGE=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/SPIKETEST_vitG384/hsdp_n16g12
RUNTIME_CFG=$ROOT/.runtime_configs/n16g12_weak/configs/vitg16_surg_vid_webdataset_single4/vitG384_cleandata.yaml
PARAMS=$CKPT_DIR/params-pretrain.yaml
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID  [16-node HSDP no-wedge verify]"

$PY_STAGE $ROOT/scripts/prepare_runtime_config.py \
    $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 16 --weak-scale > /dev/null
$PY_STAGE - "$RUNTIME_CFG" "$PARAMS" "$CKPT_DIR" <<'PY'
import sys, yaml
src, dst, folder = sys.argv[1], sys.argv[2], sys.argv[3]
d = yaml.safe_load(open(src))
d["folder"] = folder
d["optimization"]["ipe"] = 120           # long enough to see the DDP wedge onset (~42)
d["optimization"]["epochs"] = 1
d["meta"]["save_every_freq"] = 1000000000 # no checkpoint saving
yaml.safe_dump(d, open(dst, "w"), sort_keys=False)
print(f"patched test params -> {dst} (ipe120, epochs1, no-save, folder={folder})")
PY

cd $ROOT
module load frameworks
export PYTHONNOUSERSITE=1
source $VENV/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH
python -c "import torch; print('torch', torch.__version__)" 2>&1 | grep -viE "UserWarning|warn" | head -1

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MPICH_GPU_SUPPORT_ENABLED=1
# HSDP TRANSPORT FIX (validated 2n, job 8643134: 60 iters clean vs pmix/mpi hang):
# FSDP's rapid intra/inter-node subgroup collectives DEADLOCK under the pmix/mpi
# ATL transport (a single bare all_reduce passes, but sustained FSDP load hangs at
# iter 0). PRISM's launcher=none + ofi transport handles it. NOTE: with
# launcher=none the training mpiexec must NOT pass --pmi=pmix.
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_KVS_IFACE=hsn0
export CCL_OP_SYNC=1
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring
export CCL_CHUNK_SIZE=16777216
export FI_PROVIDER=cxi
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
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

# -------- HSDP knobs (the only behavioral change vs the DDP wedge run) --------
export VJEPA_DIST_STRATEGY=hsdp
export LOCAL_WORLD_SIZE=12
export FSDP_SHARDING=shard_grad_op   # _HYBRID_SHARD_ZERO2
unset VJEPA_BF16_COMM
unset VJEPA_DDP_BUCKET_MB

MASTER_ADDR=$(head -n1 "$PBS_NODEFILE"); export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=192
echo "HSDP 16n: WORLD_SIZE=$WORLD_SIZE LOCAL_WORLD_SIZE=$LOCAL_WORLD_SIZE FSDP_SHARDING=$FSDP_SHARDING MASTER_ADDR=$MASTER_ADDR"

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging ---"
mpiexec -n 16 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS --local-root $LOCAL_DATA_ROOT \
        --num-nodes 16 --local-world-size 12 --workers 8
echo "--- staging complete ---"

# ---- STALL WATCHDOG: fast-fail if no iters get logged ----
# A silent load/collective hang (e.g. job 8642347: 40min, 0 iters) must NOT idle
# 16 nodes to walltime. This background watchdog kills the training mpiexec if the
# per-rank CSV (log_r0.csv) does not gain rows within FIRST_ITER_DEADLINE of launch,
# or stalls (no new rows) for STALL_DEADLINE thereafter.
CSV_WATCH="$CKPT_DIR/log_r0.csv"   # folder key == $CKPT_DIR (set in the patch above)
FIRST_ITER_DEADLINE=600   # 10 min: staging+wrap+load+first iter must land by here
STALL_DEADLINE=300        # 5 min with no new row after training starts = wedged
(
    start=$(date +%s); last_rows=-1; last_change=$start
    while true; do
        sleep 30
        # stop watching once the trainer mpiexec is gone
        pgrep -f "app.main_dist_aurora" >/dev/null 2>&1 || exit 0
        now=$(date +%s)
        rows=0; [ -f "$CSV_WATCH" ] && rows=$(($(wc -l < "$CSV_WATCH" 2>/dev/null || echo 1)-1))
        if [ "$rows" -gt 0 ]; then
            if [ "$rows" -ne "$last_rows" ]; then last_rows=$rows; last_change=$now; fi
            if [ $((now-last_change)) -gt $STALL_DEADLINE ]; then
                echo "WATCHDOG: STALL — no new iters for ${STALL_DEADLINE}s at row $rows. Killing." >&2
                pkill -9 -f "app.main_dist_aurora"; exit 1
            fi
        else
            if [ $((now-start)) -gt $FIRST_ITER_DEADLINE ]; then
                echo "WATCHDOG: NO FIRST ITER within ${FIRST_ITER_DEADLINE}s (load/collective hang). Killing." >&2
                pkill -9 -f "app.main_dist_aurora"; exit 1
            fi
        fi
    done
) &
WATCHDOG_PID=$!

mpiexec -n 192 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
kill $WATCHDOG_PID 2>/dev/null
echo "JOB END: $(date)"
