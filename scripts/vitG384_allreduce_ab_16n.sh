#!/usr/bin/env bash
# 16-NODE CCL_ALLREDUCE A/B — ring vs double_tree, the reviewer's fabric-topology test.
#
# WHY: per-rank CSV analysis of the live campaign proved the Mode-B throughput spikes are
# COHORT-WIDE (mean 87% of 192 ranks stuck high together on every spike, low min) with a FLAT
# backward floor (~2s, no rising trend) — i.e. fabric-collective contention, NOT compute
# stragglers and NOT L0/mem accumulation. That is the ring-AllReduce signature at N=16 (ring
# depth = 15 inter-node hops; one congested CXI link stalls the whole chain). This A/B tests
# whether a log-depth topology (double_tree, ~4 hops) flattens the spikes.
#
# ISOLATION: identical to the production recipe (fixedshape cfg, HSDP shard_grad_op, OFI, wc=1,
# workers=0) — the ONLY variable is CCL_ALLREDUCE, passed via qsub -v ALGO=ring|double_tree.
# ipe200, NO checkpoint save, separate SPIKETEST dir — does NOT touch the capacity campaign.
#
# Submit BOTH arms (debug-scaling is serial, so they run back-to-back):
#   qsub -v ALGO=ring        scripts/vitG384_allreduce_ab_16n.sh
#   qsub -v ALGO=double_tree scripts/vitG384_allreduce_ab_16n.sh
#
#PBS -N vitG_ar_ab
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=16
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
ALGO="${ALGO:-ring}"   # ring | double_tree | recursive_doubling | topo
# fixedshape (NOT cleandata) — pinned masks remove the mask-variance confound; matches production.
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/vitG384_fixedshape.yaml
VENV=/flare/ModCon/ngetty/venvs/torchtune-pt213-xpu
PY_STAGE=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/SPIKETEST_vitG384/ar_${ALGO}_n16g12
RUNTIME_CFG=$ROOT/.runtime_configs/n16g12_weak/configs/vitg16_surg_vid_webdataset_single4/vitG384_fixedshape.yaml
PARAMS=$CKPT_DIR/params-pretrain.yaml
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID  [CCL_ALLREDUCE A/B: ALGO=$ALGO]"

$PY_STAGE $ROOT/scripts/prepare_runtime_config.py \
    $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 16 --weak-scale > /dev/null
$PY_STAGE - "$RUNTIME_CFG" "$PARAMS" "$CKPT_DIR" <<'PY'
import sys, yaml
src, dst, folder = sys.argv[1], sys.argv[2], sys.argv[3]
d = yaml.safe_load(open(src))
d["folder"] = folder
d["optimization"]["ipe"] = 200            # long enough to measure steady-state spike RATE
d["optimization"]["epochs"] = 1
d["meta"]["save_every_freq"] = 1000000000  # no checkpoint saving
# start FRESH from Meta init (no resume) so both arms are identical cold-start -> steady-state.
yaml.safe_dump(d, open(dst, "w"), sort_keys=False)
print(f"patched test params -> {dst} (ipe200, no-save, folder={folder})")
PY

cd $ROOT
module load frameworks
export PYTHONNOUSERSITE=1
source $VENV/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_KVS_IFACE=hsn0
export CCL_OP_SYNC=1
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=$ALGO          # <<< THE ONLY VARIABLE >>>
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
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95
export FI_MR_CACHE_MONITOR=disabled
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
unset XPU_USM_ALLOC_SO

export VJEPA_DIST_STRATEGY=hsdp
export LOCAL_WORLD_SIZE=12
export FSDP_SHARDING=shard_grad_op
export VJEPA_NUM_WORKERS=0
unset VJEPA_BF16_COMM
unset VJEPA_DDP_BUCKET_MB

MASTER_ADDR=$(head -n1 "$PBS_NODEFILE"); export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=192
echo "AR-AB 16n: ALGO=$ALGO WORLD_SIZE=$WORLD_SIZE FSDP_SHARDING=$FSDP_SHARDING"

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging ---"
mpiexec -n 16 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS --local-root $LOCAL_DATA_ROOT \
        --num-nodes 16 --local-world-size 12 --workers 8
echo "--- staging complete ---"

CSV_WATCH="$CKPT_DIR/log_r0.csv"
FIRST_ITER_DEADLINE=600
STALL_DEADLINE=600   # tolerate a spike; only a true wedge should kill this probe
(
    start=$(date +%s); last_rows=-1; last_change=$start
    while true; do
        sleep 30
        pgrep -f "app.main_dist_aurora" >/dev/null 2>&1 || exit 0
        now=$(date +%s)
        rows=0; [ -f "$CSV_WATCH" ] && rows=$(($(wc -l < "$CSV_WATCH" 2>/dev/null || echo 1)-1))
        if [ "$rows" -gt 0 ]; then
            if [ "$rows" -ne "$last_rows" ]; then last_rows=$rows; last_change=$now; fi
            if [ $((now-last_change)) -gt $STALL_DEADLINE ]; then
                echo "WATCHDOG: STALL — no new iters ${STALL_DEADLINE}s at row $rows. Killing." >&2
                pkill -9 -f "app.main_dist_aurora"; exit 1
            fi
        elif [ $((now-start)) -gt $FIRST_ITER_DEADLINE ]; then
            echo "WATCHDOG: NO FIRST ITER within ${FIRST_ITER_DEADLINE}s. Killing." >&2
            pkill -9 -f "app.main_dist_aurora"; exit 1
        fi
    done
) &
WATCHDOG_PID=$!

mpiexec -n 192 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
kill $WATCHDOG_PID 2>/dev/null

# ---- CROSS-RANK SPIKE VERDICT (drop first 50 iters = warmup) ----
echo "=== ALGO=$ALGO cross-rank backward-ms verdict (steady-state, iters 50+) ==="
$PY_STAGE - "$CKPT_DIR" "$ALGO" <<'PY'
import sys, glob, numpy as np
from collections import defaultdict
CK, algo = sys.argv[1], sys.argv[2]
d = defaultdict(list)
for f in glob.glob(f"{CK}/log_r*.csv"):
    rows = open(f).read().splitlines()
    for l in rows[1:]:
        p = l.split(",")
        if len(p) < 9: continue
        try: e, it, b = int(p[0]), int(p[1]), float(p[8])
        except: continue
        if it >= 50 and b >= 0: d[(e, it)].append(b)   # drop warmup
    # per-rank iter-time floor too (col3 iter-time ms)
# cohort-wide spike metrics
spikes = 0; total = 0; fracs = []
allbwd = []
for k, v in d.items():
    if len(v) < 150: continue
    total += 1
    v = np.array(v); allbwd.extend(v.tolist())
    if v.max() > 15000:
        spikes += 1
        fracs.append((v > 0.5*v.max()).mean())
if allbwd:
    a = np.sort(allbwd)
    n = len(a)
    print(f"ALGO={algo}: backward-ms across ranks (iters50+, n={n}):")
    print(f"  p10={a[int(n*.1)]:.0f} p50={a[int(n*.5)]:.0f} p90={a[int(n*.9)]:.0f} "
          f"p99={a[int(n*.99)]:.0f} max={a[-1]:.0f}")
    print(f"  spike-iters(max>15s)={spikes}/{total} "
          f"({100*spikes/total if total else 0:.0f}%)  "
          f"mean-cohort-fraction-stuck={np.mean(fracs) if fracs else 0:.2f}")
    print(f"  VERDICT KEY: fewer spike-iters + lower p90/p99 than ring => double_tree helps.")
PY
echo "JOB END: $(date)"
