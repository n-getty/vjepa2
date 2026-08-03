#!/bin/bash
# PAIRED A/B: does VJEPA_TRUE_ACCUM buy throughput by amortizing collectives?
#
# WHY THIS EXPERIMENT
# -------------------
# Measured on the 64n shakeout (job 8730919, bs=1): backward is 57% of iteration
# time and carries 5.7x variance while forward is narrow -- the signature of
# communication, not compute. At bs=1 each rank does ONE clip of work per
# allreduce, the worst compute-to-communication ratio available.
#
# TRUE_ACCUM fetches N loader batches per optimizer step with the collective
# deferred to the last (no_sync on the rest), so it is ONE allreduce per N clips.
# Raising per-rank bs would amortize collectives identically, but holds N clips
# of activations at once; TRUE_ACCUM keeps the activation peak FLAT, which
# matters because the 2B backward-wedge is an L0-headroom problem.
#
# WHY PAIRED, IN ONE JOB
# ----------------------
# The baseline's iteration time has ~81% coefficient of variation (5.6-88.6 s).
# Two separate jobs would confound the arms with whatever fabric contention
# existed in each hour -- and at that CV, ~25 iters/arm can only resolve a 40-50%
# effect. Running both arms back-to-back on the SAME nodes in ONE allocation
# cancels the between-hour term, and ipe=60 gives each arm enough samples.
#
# WHAT TO MEASURE, AND WHAT NOT TO
# --------------------------------
# Use WALL-CLOCK iter-time. Do NOT use the backward-ms column: PhaseTimer takes
# XPU event deltas, and on the slowest iterations those come back NEGATIVE
# (observed at itrs 0, 12, 24 of job 8730919: -139 s, -326 s, -277 s). The phase
# timer breaks precisely on the stalled collectives this experiment is about, so
# it would silently drop the worst cases and flatter whichever arm stalls more.
#
# Also note accum RAISES global batch (3072 -> 6144 at accum=2), which shifts
# EMA/warmup/lambda. This job measures THROUGHPUT ONLY -- it is not a quality
# comparison, and its checkpoints are not for probing.
#
#   qsub scripts/accum_ab_64n.sh
#
# NOTE: no `set -u` (memory set-u-module-load-trap).
#
#PBS -N accumab
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=64
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
POOL=${DAOS_POOL:-AuroraGPT}
CONT=${DAOS_CONT:-vjepa_surg_wds}
MODELS_CONT=${DAOS_MODELS_CONT:-vjepa_models}
DAOS_MNT=/tmp/${POOL}/${CONT}
MODELS_MNT=/tmp/${POOL}/${MODELS_CONT}
PPN=12
NNODES=$(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | wc -l); NNODES=${NNODES:-64}
WORLD=$(( NNODES * PPN ))
IPE=${AB_IPE:-60}
CFG_NAME=${VJEPA_CFG_NAME:-vitG384_lbA8}
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/${CFG_NAME}.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
TAG="${PBS_JOBID%%.*}"; TAG="${TAG:-manual}"
OUTROOT=/flare/ModCon/ngetty/checkpoints/accum_ab/${TAG}
VERDICT=/flare/ModCon/ngetty/logs/accum_ab_VERDICT.txt
mkdir -p "$OUTROOT" /flare/ModCon/ngetty/logs

echo "JOB START $(date) PBS_JOBID=$PBS_JOBID  ${NNODES}n x ${PPN} = ${WORLD} ranks, ipe=$IPE"

cd $ROOT
module use /soft/modulefiles
module load frameworks
module load daos
export PYTHONNOUSERSITE=1
source /flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_KVS_IFACE=hsn0
# Neutralize the DDP-oriented globals in case an outer AURORA_ENV sets them:
# CCL_KVS_MODE=mpi / CCL_KVS_USE_MPI_RANKS=1 conflict with oneCCL bringing up its
# OWN KVS over CXI, which is what launcher=none + ofi requires (findings 5a).
# Free insurance -- this script does not inherit that global today, but a future
# wrapper might.
# UNSET, not empty. oneCCL validates this enum and rejects '': it raised
# "CCL_KVS_MODE: unexpected value: , expected values: pmi, mpi, pmix_ofi,
# pmix_ofi_shm" and killed the accum=2 arm of job 8731004 at iter 0. findings 5a
# literally says 'CCL_KVS_MODE=   # NEUTRALIZE (empty string)' -- that guidance is
# wrong for this oneCCL build. Removing the vars is what neutralizes them.
unset CCL_KVS_MODE CCL_KVS_USE_MPI_RANKS
export CCL_OP_SYNC=1
export CCL_LOG_LEVEL=${CCL_LOG_LEVEL:-error}
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
export VJEPA_DIST_STRATEGY=hsdp
export LOCAL_WORLD_SIZE=$PPN
export FSDP_SHARDING=shard_grad_op
export VJEPA_NUM_WORKERS=${VJEPA_NUM_WORKERS:-0}
export TORCH_DIST_TIMEOUT_SECONDS=${TORCH_DIST_TIMEOUT_SECONDS:-3600}
export WDS_LOCAL_SLICING=0
unset LD_PRELOAD

if [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR MASTER_PORT=29500 WORLD_SIZE=$WORLD

launch-dfuse.sh ${POOL}:${MODELS_CONT} || { echo "FATAL: launch-dfuse models"; exit 1; }
launch-dfuse.sh ${POOL}:${CONT}        || { echo "FATAL: launch-dfuse corpus"; exit 1; }
timeout 60 ls "$DAOS_MNT" >/dev/null 2>&1 || { echo "FATAL: $DAOS_MNT unresponsive"; exit 1; }
timeout 60 ls "$MODELS_MNT" >/dev/null 2>&1 || { echo "FATAL: $MODELS_MNT unresponsive"; exit 1; }
echo "DAOS mounted (corpus + models)"

$PY $ROOT/scripts/prepare_runtime_config.py \
  $BASE_CFG --root $ROOT --num-gpus $PPN --num-nodes $NNODES --weak-scale > /dev/null || exit 1
RUNTIME_CFG=$ROOT/.runtime_configs/n${NNODES}g${PPN}_weak/configs/vitg16_surg_vid_webdataset_single4/${CFG_NAME}.yaml

run_arm () {
  local accum=$1
  local dir=$OUTROOT/accum${accum}
  local params=$dir/params.yaml
  mkdir -p "$dir"
  cp "$RUNTIME_CFG" "$params" || return 1
  $PY - "$params" "$dir" "$IPE" "$MODELS_MNT" <<'PY'
import sys, yaml, os
p, d, ipe, models = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
c = yaml.safe_load(open(p))
c["folder"] = d
c["optimization"]["ipe"] = ipe
c["optimization"]["epochs"] = 1
meta = c.setdefault("meta", {})
ck = meta.get("pretrain_checkpoint")
if ck:
    cand = os.path.join(models, os.path.basename(ck))
    if not os.path.exists(cand):
        sys.exit(f"FATAL: {cand} missing")
    meta["pretrain_checkpoint"] = cand
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
PY
  echo "===== ARM accum=$accum (gbatch $(( WORLD * accum ))) ====="
  local t0=$(date +%s)
  VJEPA_TRUE_ACCUM=$accum \
  mpiexec -n $WORLD -ppn $PPN --cpu-bind depth --depth 16 --no-vni \
      --env VJEPA_TRUE_ACCUM=$accum \
      -o "$dir/rank.%r.out" -e "$dir/rank.%r.err" \
      python -m app.main_dist_aurora --train_mode \
          --fname $params --params_path $params \
          --local_data_root $DAOS_MNT
  echo "  arm accum=$accum finished rc=$? in $(( $(date +%s) - t0 ))s"
}

# Order matters for fairness: arm 1 pays the cold-cache cost (first DAOS touch,
# first checkpoint read). Run accum=2 FIRST so the cheaper-per-step arm does not
# get handed a warm machine -- if accum=2 still wins from cold, the result is
# conservative rather than flattered.
run_arm 2
run_arm 1

$PY - "$OUTROOT" "$WORLD" <<'PY' | tee "$VERDICT"
import os, sys, statistics
root, world = sys.argv[1], int(sys.argv[2])
print("================ ACCUM A/B (paired, same nodes) ================")
print(f"root {root}  world {world} ranks\n")
res = {}
for accum in (1, 2):
    csv = os.path.join(root, f"accum{accum}", "log_r0.csv")
    if not os.path.isfile(csv):
        print(f"  accum={accum}: NO CSV"); continue
    it = []
    for line in open(csv):
        p = line.split(",")
        if p and p[0] == "epoch":
            it = []; continue
        if len(p) > 3 and p[1].strip().isdigit() and int(p[1]) >= 5:
            it.append(float(p[3]) / 1000.0)
    if not it:
        print(f"  accum={accum}: no iters past warmup"); continue
    s = sorted(it)
    res[accum] = dict(n=len(s), med=statistics.median(s), lo=s[0], hi=s[-1],
                      p25=s[len(s)//4], p75=s[3*len(s)//4])
    r = res[accum]
    # clips/s uses world*accum per step: accum=2 does 2x work per iteration
    thru = world * accum / r["med"]
    print(f"  accum={accum}: n={r['n']:3d}  median {r['med']:6.2f}s  "
          f"IQR {r['p25']:.1f}..{r['p75']:.1f}  range {r['lo']:.1f}..{r['hi']:.1f}  "
          f"-> {thru:7.1f} clips/s")
if 1 in res and 2 in res:
    t1 = world * 1 / res[1]["med"]
    t2 = world * 2 / res[2]["med"]
    print(f"\n  throughput accum2/accum1 = {t2/t1:.2f}x")
    # Overlapping IQRs mean the medians are not separated by this sample.
    a, b = res[1], res[2]
    overlap = not (a["p75"] < b["p25"] or b["p75"] < a["p25"])
    if overlap:
        print("  IQRs OVERLAP -- this run does NOT resolve the difference.")
        print("  Do not report a speedup from it; rerun with more iters.")
    else:
        print("  IQRs disjoint -- the difference is resolved by this sample.")
print("\n  Measured on WALL-CLOCK iter time. The backward-ms column is unusable:")
print("  XPU event deltas go NEGATIVE on stalled collectives (job 8730919 itrs")
print("  0/12/24: -139s/-326s/-277s), i.e. exactly the cases under test.")
PY
echo "JOB END $(date)"
