#!/usr/bin/env bash
# 16-NODE CHEAP FABRIC A/B: CCL_WORKER_COUNT=1 -> 4 (§4f follow-up).
#
# The env-diff run (8643398) proved the residual is fabric contention (cohort spikes +
# FLAT l0-free/l0-ext). Comparing OUR launcher against PRISM's production web launcher
# (tools/launch_aurora_web.py) line-by-line: we are IDENTICAL on CCL_ALLREDUCE=ring and
# CCL_CHUNK_SIZE=16MiB — those are NOT the lever. The ONE difference is worker count:
# PRISM sets CCL_WORKER_COUNT=4, we set 1. The CCL progress-engine worker count governs
# how many threads drain in-flight collectives; under contention, 1 worker is a plausible
# bottleneck producing exactly the cohort-wide backward inflation we see. PRISM notes
# WORKER_COUNT=8 -> pthread_create EINVAL, so 4 is the safe max.
#
# This is the cheapest possible probe: ZERO code change, ga=1 (so backward-ms stays clean
# and directly comparable to env-diff 8643398), all 3 insurance flags stay unset. The ONLY
# delta vs 8643398 is CCL_WORKER_COUNT=4.
# PASS = p50/p90 backward (and iter-time) cross-rank trend flattens vs the escalating
#        10->30s of 8643398. Analyze with scripts/analyze_straggler.py.
#
# Submit:  qsub scripts/vitG384_hsdp_cclworkers_16n.sh
#
#PBS -N vitG_cclw4_16
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=16
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/vitG384_fixedshape.yaml
VENV=/flare/ModCon/ngetty/venvs/torchtune-pt213-xpu
PY_STAGE=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/SPIKETEST_vitG384/cclw4_n16g12
RUNTIME_CFG=$ROOT/.runtime_configs/n16g12_weak/configs/vitg16_surg_vid_webdataset_single4/vitG384_fixedshape.yaml
PARAMS=$CKPT_DIR/params-pretrain.yaml
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs
# Clear stale per-rank CSVs from a prior run in this CKPT_DIR — the trainer APPENDS,
# so leftover rows corrupt the windowed drift analysis and the watchdog row-count.
rm -f $CKPT_DIR/log_r*.csv

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID  [16-node HSDP no-wedge verify]"

$PY_STAGE $ROOT/scripts/prepare_runtime_config.py \
    $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 16 --weak-scale > /dev/null
$PY_STAGE - "$RUNTIME_CFG" "$PARAMS" "$CKPT_DIR" <<'PY'
import sys, yaml
src, dst, folder = sys.argv[1], sys.argv[2], sys.argv[3]
d = yaml.safe_load(open(src))
d["folder"] = folder
d["optimization"]["ipe"] = 200           # drift onset was iter ~41-60; 200 confirms truly flat
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
export CCL_WORKER_COUNT=4
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
# ENV-DIFF TEST (§4d): strict diff against PRISM's production launcher
# (tools/launch_aurora_web.py), which is grep-clean of ALL THREE of these "insurance"
# flags yet runs HSDP+ZERO2 at 20N stably. Two force CCL into ACCUMULATION mode
# (never-evict IPC handles + no MR-cache invalidation); the third is a GC-threshold we
# also added preemptively. The cohort-wide spike + creeping-floor signature is exactly
# what a never-evict cache does when it periodically compacts under pressure. Decisive
# test: run WITHOUT all three (keep fixed-shape masks + top-level wrap, change nothing
# else) so a flat result is unambiguous — no re-run needed to isolate which flag mattered.
# If banned:1 returns, hunt the real shape/allocator interaction with default telemetry
# instead of masking it.
unset PYTORCH_ALLOC_CONF
unset FI_MR_CACHE_MONITOR
unset CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD
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
# 12 min: the DDP wedge hit single iters of 328s and RECOVERED; a recovering
# mega-spike must not be mistaken for a true deadlock. 720s still bounds a real
# hang (kills within ~12min of the last iter) while surviving spike-and-recover,
# which is exactly the signal we're trying to observe past iter 120.
STALL_DEADLINE=720
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

# ---- DRIFT VERDICT: windowed mean iter-time from rank-0 CSV ----
# PASS = later windows do NOT climb vs early windows (top-level wrap fixed the
# ~72x IPC-handle churn). The per-layer run drifted 8.9->9.9->16.9->32.3s over
# iters 1..80; a flat trajectory here confirms the fix.
CSV="$CKPT_DIR/log_r0.csv"
if [[ -f "$CSV" ]]; then
  echo "=== iter-time trajectory (rank0, 20-iter windows) — PASS if flat, not climbing ==="
  $PY_STAGE - "$CSV" <<'PY'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1])))
# find the iter-time column (ms). PhaseTimer writes 'iter-ms' or similar.
cand = [c for c in (rows[0].keys() if rows else []) if 'iter' in c.lower() and 'ms' in c.lower()]
col = cand[0] if cand else None
if not col:
    print(f"(no iter-ms column; columns={list(rows[0].keys()) if rows else []})"); sys.exit(0)
vals = []
for r in rows:
    try: vals.append(float(r[col]))
    except (ValueError, KeyError, TypeError): pass
if not vals:
    print("(no numeric iter-times)"); sys.exit(0)
W = 20
for s in range(0, len(vals), W):
    w = vals[s:s+W]
    if w: print(f"  iters {s+1:>3}-{s+len(w):<3}: mean={sum(w)/len(w)/1000:.2f}s  n={len(w)}")
early = vals[:W]; late = vals[-W:]
if early and late:
    e, l = sum(early)/len(early)/1000, sum(late)/len(late)/1000
    ratio = l/e if e else 0
    verdict = "PASS (flat)" if ratio < 1.5 else "FAIL (drift)"
    print(f"  first-window={e:.2f}s last-window={l:.2f}s ratio={ratio:.2f}x -> {verdict}")
PY
fi
echo "JOB END: $(date)"
