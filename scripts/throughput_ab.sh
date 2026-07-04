#!/usr/bin/env bash
# MODE-A DIAGNOSTIC — isolate the DataLoader-startup shm-unmap crash. SHORT jobs only.
#
# Mode A: "MapAllocator::close(): could not unmap the shared memory file: Unknown error 12517375"
# kills a rank at DataLoader startup (after model-load + "Initializing loader", before iter 0).
# 7/12 capacity jobs, different node each time, ONLY at 16n (1n smokes ran clean). => scales with
# worker-process count (192 ranks x N workers) / /dev/shm pressure, possibly stale shm from
# prior killed jobs on the same node.
#
# This script is PARAMETERIZED by env + `-l select=N` at qsub time. Diagnostic ladder:
#   Step 1 (1n, workers=0):  qsub -l select=1  -v VJEPA_NUM_WORKERS=0,VJEPA_PIN_MEM=0 scripts/modeA_diag.sh
#   Step 2 (16n, workers=0): qsub -l select=16 -v VJEPA_NUM_WORKERS=0,VJEPA_PIN_MEM=0,DIAG_SHM_CLEAN=1 scripts/modeA_diag.sh
#   Then reintroduce ONE knob: VJEPA_NUM_WORKERS=1 -> +VJEPA_PIN_MEM=1 -> +VJEPA_PREFETCH_FACTOR=2
# PASS = reaches iter 0 and logs a few iters (ipe=5), no shm-unmap/segv. Logs df -h /dev/shm +
# ipcs per node BEFORE training so exhaustion is visible either way.
#
#PBS -N tput_ab
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l walltime=00:20:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
# NOTE: pass -l select=N on the qsub command line (1 or 16). Do NOT hardcode select here.

set -eo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
# DIAG_CFG env overrides the base config. DEFAULT is the REAL fixedshape config (batches_per_rank
# ~795), NOT SMOKE (bp_r=2). The SMOKE config's tiny data caused constant re-iteration churn that
# stalled the loader and produced invalid throughput numbers — must measure on real data.
BASE_CFG=${DIAG_CFG:-$ROOT/configs/vitg16_surg_vid_webdataset_single4/vitG384_fixedshape.yaml}
VENV=/flare/ModCon/ngetty/venvs/torchtune-pt213-xpu
PY_STAGE=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python

# node count from PBS_NODEFILE (set by -l select=N at qsub)
NNODES=$(sort -u "$PBS_NODEFILE" | wc -l)
LWS=12                       # tiles/ranks per node
WS=$((NNODES * LWS))
NW=${VJEPA_NUM_WORKERS:-0}   # default: workerless isolator (Mode A fix)
PIN=${VJEPA_PIN_MEM:-0}
WC=${CCL_WORKER_COUNT:-1}
AR=${CCL_ALLREDUCE:-ring}
CFGTAG=$(basename "$BASE_CFG" .yaml)
TAG="${CFGTAG}_n${NNODES}_nw${NW}_wc${WC}_ar${AR}"
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/MODEA_DIAG/$TAG
PARAMS=$CKPT_DIR/params-pretrain.yaml
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs
rm -f $CKPT_DIR/log_r*.csv $CKPT_DIR/latest.pth.tar   # always a FRESH start (SMOKE, no resume)

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID  [MODE-A DIAG $TAG: nodes=$NNODES ws=$WS num_workers=$NW pin_mem=$PIN shm_clean=${DIAG_SHM_CLEAN:-0}]"

# prepare_runtime_config.py PRINTS the output path; capture it rather than reconstructing the
# topology-suffix dir (1n -> g12_weak, Nn -> n{N}g12_weak — reconstructing it was a bug).
RUNTIME_CFG=$($PY_STAGE $ROOT/scripts/prepare_runtime_config.py \
    $BASE_CFG --root $ROOT --num-gpus $LWS --num-nodes $NNODES --weak-scale | tail -1)
echo "runtime cfg -> $RUNTIME_CFG"
# patch folder->CKPT_DIR + ipe=5 + no-save (fresh startup-only test)
$PY_STAGE - "$RUNTIME_CFG" "$PARAMS" "$CKPT_DIR" <<'PY'
import sys, yaml
src, dst, folder = sys.argv[1], sys.argv[2], sys.argv[3]
d = yaml.safe_load(open(src))
d["folder"] = folder
d["optimization"]["ipe"] = 30
d["optimization"]["epochs"] = 1
d["meta"]["save_every_freq"] = 1000000000
yaml.safe_dump(d, open(dst, "w"), sort_keys=False)
print(f"patched diag params -> {dst} (ipe5, folder={folder})")
PY

cd $ROOT
module load frameworks
export PYTHONNOUSERSITE=1
source $VENV/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MPICH_GPU_SUPPORT_ENABLED=1
export CCL_OP_SYNC=1
export CCL_WORKER_COUNT=$WC
export CCL_ALLREDUCE=$AR
export CCL_CHUNK_SIZE=16777216
export FI_PROVIDER=cxi
export PYTHONFAULTHANDLER=1
export TMPDIR=/tmp
export OMP_NUM_THREADS=16
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export WDS_LOCAL_SLICING=1

# transport: 1n uses pmix/mpi (fine); >1n needs launcher=none + OFI (pmix/mpi deadlocks FSDP)
if [[ "$NNODES" -eq 1 ]]; then
  export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi CCL_KVS_USE_MPI_RANKS=1
  export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_KVS_CONNECTION_TIMEOUT=600
  PMI_FLAG="--pmi=pmix"
else
  export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi CCL_KVS_IFACE=hsn0
  export FI_CXI_RX_MATCH_MODE=hybrid FI_CXI_OFLOW_BUF_SIZE=8388608 FI_CXI_DEFAULT_CQ_SIZE=131072
  PMI_FLAG=""
fi

# HSDP + the Mode-A knobs under test
export VJEPA_DIST_STRATEGY=hsdp
export LOCAL_WORLD_SIZE=$LWS
export FSDP_SHARDING=shard_grad_op
export VJEPA_NUM_WORKERS=$NW
export VJEPA_PIN_MEM=$PIN
# VJEPA_PREFETCH_FACTOR passes through if set
unset VJEPA_BF16_COMM VJEPA_DDP_BUCKET_MB VJEPA_GRAD_ACCUM VJEPA_TRUE_ACCUM
# insurance flags stay unset (env-diff baseline)
unset PYTORCH_ALLOC_CONF FI_MR_CACHE_MONITOR CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD

MASTER_ADDR=$(head -n1 "$PBS_NODEFILE"); export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=$WS
echo "DIAG: nodes=$NNODES WORLD_SIZE=$WS num_workers=$NW pin_mem=$PIN prefetch=${VJEPA_PREFETCH_FACTOR:-n/a} transport=$CCL_ATL_TRANSPORT"

# -------- SHM EVIDENCE + optional hygiene, per node, BEFORE training --------
echo "=== per-node /dev/shm + ipcs BEFORE training ==="
mpiexec -n $NNODES -ppn 1 --cpu-bind none bash -c '
  echo "[$(hostname)] shm: $(df -h /dev/shm 2>/dev/null | awk "NR==2{print \$3\"/\"\$2\" used\"}") | ipcs-seg: $(ipcs -m 2>/dev/null | grep -c 0x) | shm-files: $(ls /dev/shm 2>/dev/null | wc -l)"
' 2>&1 | grep -viE "warn" | sort | head -20 || true
if [[ "${DIAG_SHM_CLEAN:-0}" == "1" ]]; then
  echo "=== SHM HYGIENE: removing stale torch/psm shm (best-effort, never fatal) ==="
  # NOTE: entire block guarded with `|| true` — under `set -e` a benign nonzero from
  # rm/ipcrm (nothing to remove) must NOT abort the diagnostic (that killed job 8644220).
  mpiexec -n $NNODES -ppn 1 --cpu-bind none bash -c \
    'rm -f /dev/shm/torch_* /dev/shm/*psm* /dev/shm/sem.* 2>/dev/null; true' 2>&1 \
    | grep -viE "warn" | head -3 || true
  echo "  hygiene done"
fi

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging ---"
mpiexec -n $NNODES -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS --local-root $LOCAL_DATA_ROOT \
        --num-nodes $NNODES --local-world-size $LWS --workers 8
echo "--- staging complete ---"

echo "=== TRAIN (ipe5 startup test) ==="
mpiexec $PMI_FLAG -n $WS -ppn $LWS --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
RC=$?
echo "JOB END: $(date) (train rc=$RC)"

# ---- VERDICT ----
CSV=$CKPT_DIR/log_r0.csv
OU=$(ls -t /flare/ModCon/ngetty/logs/${PBS_JOBID%%.*}.*.OU 2>/dev/null | head -1)
echo "=== VERDICT [$TAG] ==="
rows=$([ -f "$CSV" ] && echo $(($(wc -l < "$CSV")-1)) || echo 0)
echo "iters logged: $rows (want >=1 = reached training)"
[ -n "$OU" ] && echo "shm-crash signatures: $(grep -cE "could not unmap|died from signal (11|6)" "$OU")"
# THROUGHPUT METRIC: cross-rank backward-ms p50/p90 (the straggler-sensitive collective cost).
# Compare TAGs — lower p90 = better jitter tolerance for that worker/algo combo.
echo "=== cross-rank backward spread (drop warmup itr0 + timer-overflow) ==="
vals=$(for f in "$CKPT_DIR"/log_r*.csv; do awk -F, '$2~/^[0-9]+$/ && $2>0 && $9>0{print $9/1000}' "$f" 2>/dev/null; done | sort -n)
echo "$vals" | awk -v T="$TAG" '{a[NR]=$1} END{if(NR>10)printf "THROUGHPUT[%s]: n=%d p50=%.1fs p90=%.1fs p99=%.1fs max=%.1fs\n",T,NR,a[int(NR*0.5)],a[int(NR*0.9)],a[int(NR*0.99)],a[NR]; else print "THROUGHPUT["T"]: insufficient data (n="NR")"}'
# also rank0 wall-clock iter time (end-user throughput)
[ -f "$CKPT_DIR/log_r0.csv" ] && awk -F, -v T="$TAG" '$2~/^[0-9]+$/ && $2>0{print $4/1000}' "$CKPT_DIR/log_r0.csv" | sort -n | awk -v T="$TAG" '{a[NR]=$1} END{if(NR>0)printf "WALL[%s r0]: p50=%.1fs p90=%.1fs max=%.1fs\n",T,a[int(NR*0.5)],a[int(NR*0.9)],a[NR]}'
if [ "$rows" -ge 1 ]; then echo "RESULT: PASS ($TAG reached training)"; else echo "RESULT: FAIL — crashed before iter 0 ($TAG)"; fi
