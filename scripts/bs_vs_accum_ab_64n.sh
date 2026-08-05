#!/bin/bash
# PAIRED A/B at MATCHED GLOBAL BATCH: per-rank bs=2 vs bs=1 + TRUE_ACCUM=2.
#
# THE QUESTION
# ------------
# Both arms move 2 clips/rank/step and issue ONE inter-node collective per
# optimizer step, so global batch, samples/step and comms volume are IDENTICAL.
# The only difference is the activation peak: bs=2 holds two clips live at once;
# TRUE_ACCUM processes them serially with no_sync on the first.
#
# We deploy TRUE_ACCUM on the theory that L0 headroom is the binding constraint
# on Aurora. That theory has never been tested against the alternative -- the
# +36% accum result (job 8731439) was measured against accum=1, i.e. against HALF
# the global batch, so it conflates "amortize the collective" with "do more work
# per step". This is the missing comparison, and it decides which lever the 256n
# recipe should use to grow global batch.
#
# WHY THE ANSWER IS NOT OBVIOUS EITHER WAY
#   bs=2 may win: one fused forward over 2 clips has better kernel efficiency and
#     half the loader/collator round-trips.
#   accum may win: bs2+ckpt-off measured only 1.8 GiB l0-free at 16n (vs ~11 for
#     bs1). If that headroom is what fabric transients need, bs=2 pays for its
#     kernel efficiency in stalls -- or OOMs outright.
# An OOM on the bs=2 arm is a RESULT, not a failed run: it means bs=2 cannot be
# the 256n lever and the question is closed.
#
# WHAT THIS DOES NOT MEASURE
# Throughput only. Both arms run gb = world*2, which shifts EMA/warmup/lambda
# (raw STEP counts). Whether that batch TRAINS well is the separate lbA8/lbB8
# question. These checkpoints are not for probing.
#
#   qsub scripts/bs_vs_accum_ab_64n.sh
#
# NOTE: no `set -u` (memory set-u-module-load-trap).
#
#PBS -N bsaccum
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
# ipe=22, NOT the 30 the accum A/B used. That job had one cheap arm (21.7 s) and
# one heavy (31.9 s); here BOTH arms do 2 clips/step, so both cost ~32 s. Budget
# from job 8731439's own numbers: cold-arm startup ~960 s, warm-arm ~590 s.
#   22 iters: (960 + 22*32) + (590 + 22*32) = 2958 s = 49 min. Fits with margin.
#   30 iters: 3470 s = 58 min -- arm 2 gets truncated by the wall, which is the
#   exact failure mode of job 8731332.
IPE=${AB_IPE:-22}
WARMUP_ITERS=${AB_WARMUP:-3}     # iterations dropped from the head of each arm
CFG_NAME=${VJEPA_CFG_NAME:-vitG384_lbA8}
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/${CFG_NAME}.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
TAG="${PBS_JOBID%%.*}"; TAG="${TAG:-manual}"
OUTROOT=/flare/ModCon/ngetty/checkpoints/bs_vs_accum/${TAG}
VERDICT=/flare/ModCon/ngetty/logs/bs_vs_accum_VERDICT_${TAG}.txt
mkdir -p "$OUTROOT" /flare/ModCon/ngetty/logs
JOB_T0=$(date +%s)
WALL_S=3600

echo "JOB START $(date) PBS_JOBID=$PBS_JOBID  ${NNODES}n x ${PPN} = ${WORLD} ranks, ipe=$IPE"
echo "matched global batch = $(( WORLD * 2 )) on BOTH arms"

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
# UNSET, not empty -- oneCCL validates this enum and rejects '' (that killed job
# 8731004 at iter 0). findings 5a recommends the empty string and is wrong.
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

timeout 600 launch-dfuse.sh ${POOL}:${MODELS_CONT} || { echo "FATAL: launch-dfuse models"; exit 1; }
timeout 600 launch-dfuse.sh ${POOL}:${CONT}        || { echo "FATAL: launch-dfuse corpus"; exit 1; }
timeout 60 ls "$DAOS_MNT"   >/dev/null 2>&1 || { echo "FATAL: $DAOS_MNT unresponsive"; exit 1; }
timeout 60 ls "$MODELS_MNT" >/dev/null 2>&1 || { echo "FATAL: $MODELS_MNT unresponsive"; exit 1; }
echo "DAOS mounted (corpus + models)"

# --weak-scale keeps per-rank batch at the base value (lbA8 ships bs=1); each arm
# then overrides data.batch_size in its OWN params.yaml below. It cannot be done
# through the generator: scale_local_batch() returns base_batch untouched under
# --weak-scale, so a --per-rank-bs there would be silently ignored.
$PY $ROOT/scripts/prepare_runtime_config.py \
  $BASE_CFG --root $ROOT --num-gpus $PPN --num-nodes $NNODES --weak-scale > /dev/null || exit 1
RUNTIME_CFG=$ROOT/.runtime_configs/n${NNODES}g${PPN}_weak/configs/vitg16_surg_vid_webdataset_single4/${CFG_NAME}.yaml

# run_arm <name> <per_rank_bs> <true_accum>
run_arm () {
  local name=$1 bs=$2 accum=$3
  local dir=$OUTROOT/$name
  local params=$dir/params.yaml
  mkdir -p "$dir"
  cp "$RUNTIME_CFG" "$params" || return 1
  $PY - "$params" "$dir" "$IPE" "$MODELS_MNT" "$bs" <<'PY'
import sys, yaml, os
p, d, ipe, models, bs = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
c = yaml.safe_load(open(p))
c["folder"] = d
c["data"]["batch_size"] = bs          # the whole point of the arm
c["optimization"]["ipe"] = ipe
c["optimization"]["epochs"] = 1
meta = c.setdefault("meta", {})
# Fresh folder AND load_checkpoint off, or arm 2 resumes arm 1 and runs 0 iters.
# --folder is ignored under --train_mode, so `folder` must live in the YAML
# (memory train-mode-ignores-folder-flag).
meta["load_checkpoint"] = False
meta["read_checkpoint"] = None
meta["save_every_freq"] = -1          # the end-of-epoch latest.pth.tar still writes
ck = meta.get("pretrain_checkpoint")
if ck:
    cand = os.path.join(models, os.path.basename(ck))
    if not os.path.exists(cand):
        sys.exit(f"FATAL: {cand} missing")
    meta["pretrain_checkpoint"] = cand
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
PY
  [[ $? -eq 0 ]] || { echo "  arm $name: config patch FAILED"; return 1; }
  echo "===== ARM $name : bs=$bs accum=$accum -> gbatch $(( WORLD * bs * accum )) ====="
  local t0=$(date +%s)
  mpiexec -n $WORLD -ppn $PPN --cpu-bind depth --depth 16 --no-vni \
      --env VJEPA_TRUE_ACCUM=$accum \
      -o "$dir/rank.%r.out" -e "$dir/rank.%r.err" \
      python -m app.main_dist_aurora --train_mode \
          --fname $params --params_path $params \
          --local_data_root $DAOS_MNT
  local rc=$?
  echo "  arm $name finished rc=$rc in $(( $(date +%s) - t0 ))s"
  # An OOM here is a legitimate answer, not a broken run -- surface it so the
  # verdict does not read as "no data".
  if grep -qls "OUT_OF_RESOURCES\|out of memory\|OutOfMemory" "$dir"/rank.*.err 2>/dev/null; then
    echo "  *** arm $name hit an OOM -- see $dir/rank.*.err ***"
  fi
}

# Order: run the CHALLENGER (bs=2) FIRST, on the cold machine. Arm 1 pays the
# first DAOS touch and the 28 GB checkpoint read. Our deployed choice is accum,
# so making the challenger run cold means a bs=2 win is conservative.
run_arm bs2_accum1 2 1

ELAPSED=$(( $(date +%s) - JOB_T0 ))
echo "  elapsed ${ELAPSED}s of ${WALL_S}s after arm 1"
if (( ELAPSED > WALL_S - 900 )); then
  # Under 15 min left is not enough for startup + a usable number of iters. Say
  # so rather than launching an arm the wall will cut mid-flight.
  echo "  SKIPPING arm 2: insufficient walltime remaining. Resubmit with lower AB_IPE."
else
  run_arm bs1_accum2 1 2
fi

# Verdict lives in scripts/ab_verdict.py, not inline: reading 1536 per-rank CSVs
# off Lustre takes minutes, and if the wall kills the job before this line the
# CSVs still persist -- rerun the same command by hand to recover the number.
echo "=== verdict: python scripts/ab_verdict.py $OUTROOT --world $WORLD ==="
$PY $ROOT/scripts/ab_verdict.py "$OUTROOT" \
    --world "$WORLD" --warmup "$WARMUP_ITERS" \
    --arms bs2_accum1 bs1_accum2 --clips-per-step 2 2>&1 | tee "$VERDICT"
echo "JOB END $(date)"
