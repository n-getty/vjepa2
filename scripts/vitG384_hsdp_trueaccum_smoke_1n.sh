#!/usr/bin/env bash
# 1-NODE TRUE-ACCUMULATION SMOKE GATE (§4f/§4g fix; before any 16n true-accum run).
#
# VJEPA_TRUE_ACCUM=2 fetches 2 SEPARATE loader batches per optimizer step and defers the
# inter-node collective (no_sync on encoder+predictor) to the last -> ONE ReduceScatter per
# step instead of 2. This is the REAL fabric lever (unlike VJEPA_GRAD_ACCUM which only slices
# one batch). This smoke gates the correctness + MEMORY of that path before 16n.
#
# CRITICAL (reviewer point 7): no_sync() retains the UNSHARDED gradient across the deferred
# backward, which could erode the HSDP headroom that fixed the DDP wedge. Our 2B bf16 grad is
# ~4GB vs ~15GB free L0 (env-diff min l0-free=14.9GiB), so it SHOULD fit — but this smoke
# MUST verify it. This is where we diverge from PRISM (their 7B chose no_sync-OFF; 14GB grad
# didn't fit). PASS REQUIRES the memory checks below, not just no-crash.
#
# SMOKE_vitG384.yaml: bs=2 + weight_distance_loss:true + ipe40. true_accum=2 also exercises
# the d_ij distance-loss path at full bs=2 per sub-batch (no micro-bs=1 landmine here — each
# accumulated batch is a full bs=2 loader batch).
#
# PASS = (1) ipe40 clean, no NaN/IndexError/OOM; (2) "TRUE gradient accumulation ON" logged;
#        (3) l0-free stays > ~5GiB headroom (no_sync full-grad fits); (4) loss sane/decreasing.
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/vitG384_hsdp_trueaccum_smoke_1n.sh
#
#PBS -N vitG_tacc_sm
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
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/SMOKE_vitG384/trueaccum_n1g12
PARAMS=$CKPT_DIR/params-pretrain.yaml
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs
# trainer APPENDS to per-rank CSVs; clear stale rows from a prior run in this CKPT_DIR
rm -f $CKPT_DIR/log_r*.csv

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID  [1-node TRUE-ACCUM memory smoke]"

$PY_STAGE $ROOT/scripts/prepare_runtime_config.py \
    $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 1 --weak-scale > /dev/null
# CRITICAL: patch the run folder to THIS run's unique CKPT_DIR. Otherwise the trainer
# writes to the config's shared `smoke_weak` dir, auto-resumes its STALE latest.pth.tar
# (job 8643448 read epoch=1 from a Jul-1 ckpt -> range(1,epochs=1) empty -> 0 iters,
# exit 0, NEVER ran the true-accum path). Patch folder + start from Meta init (epoch 0).
$PY_STAGE - "$RUNTIME_CFG" "$PARAMS" "$CKPT_DIR" <<'PY'
import sys, yaml
src, dst, folder = sys.argv[1], sys.argv[2], sys.argv[3]
d = yaml.safe_load(open(src))
d["folder"] = folder
d["meta"]["save_every_freq"] = 1000000000  # no checkpoint saving in a smoke
yaml.safe_dump(d, open(dst, "w"), sort_keys=False)
print(f"patched smoke params -> {dst} (folder={folder}, fresh Meta init, no-save)")
PY
# ensure a clean start: no stale ckpt/CSV in this run's folder
rm -f $CKPT_DIR/latest.pth.tar $CKPT_DIR/log_r*.csv
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
# TRUE accumulation over 2 loader batches; no_sync defers the collective to the last.
export VJEPA_TRUE_ACCUM=2

MASTER_ADDR=$(head -n1 "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || hostname); export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=12
echo "true-accum smoke: WORLD_SIZE=$WORLD_SIZE TRUE_ACCUM=$VJEPA_TRUE_ACCUM FSDP_SHARDING=$FSDP_SHARDING"

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging shards to $LOCAL_DATA_ROOT ---"
mpiexec -n 1 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS --local-root $LOCAL_DATA_ROOT \
        --num-nodes 1 --local-world-size 12 --workers 8
echo "--- staging complete ---"

echo "=== 1-node HSDP + TRUE_ACCUM=2 train smoke ==="
mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
echo "JOB END: $(date)"

# ---- VERDICT: engaged + iters + MEMORY (the critical no_sync gate) + loss ----
OU=$(ls -t /flare/ModCon/ngetty/logs/${PBS_JOBID%%.*}.*.OU 2>/dev/null | head -1)
# folder was patched to $CKPT_DIR above, so the CSV is HERE (no more shared-dir pollution).
CSV=$CKPT_DIR/log_r0.csv
echo "=== (1) TRUE accum engaged? (want 'TRUE gradient accumulation ON: true_accum=2') ==="
[ -n "$OU" ] && grep -m1 "TRUE gradient accumulation ON" "$OU" || echo "FAIL: not engaged — check OU"
echo "=== (2) iters logged (want ~40, no crash) [CSV=$CSV] ==="
[ -f "$CSV" ] && echo "rank0 CSV rows: $(($(wc -l < "$CSV")-1))" || echo "NO CSV — crashed before first iter"
echo "=== (3) NaN/IndexError/OOM/unmap? (want 0) ==="
[ -n "$OU" ] && grep -cE "IndexError|nan|NaN|Traceback|out of memory|OutOfMemory|could not unmap" "$OU" || true
echo "=== (4) MEMORY GATE — l0-free MiB min/last (want min > ~5000 = no_sync full-grad fits) ==="
if [ -f "$CSV" ]; then
  awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="l0-free-mib")c=i}
           NR>1 && c && $c ~ /^[0-9.]+$/ {v=$c; if(min==""||v<min)min=v; last=v}
           END{if(min!="")printf "l0-free min=%.0f MiB last=%.0f MiB -> %s\n",min,last,(min>5000?"PASS":"FAIL <5GiB headroom"); else print "(no l0-free col)"}' "$CSV"
else echo "(no CSV)"; fi
echo "=== (5) loss trajectory (want sane, ~decreasing, no NaN) ==="
[ -f "$CSV" ] && awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="loss")c=i} NR>1 && c{print $2": "$c}' "$CSV" | sed -n '1p;2p;20p;$p' || true
