#!/bin/bash
# ############################################################################
# DEPRECATED 2026-08-05 -- use scripts/vitG384_256n_daos.sh instead.
#
# This is the pre-DAOS launcher: it stages the corpus to per-node /tmp, which
# was measured at 0.43 GB/s/node (66 TB at 256n, up to 2.7 h before iter 1)
# against DAOS's 25.07 GB/s. It is kept only as the record of what the staging
# path looked like; every question in the list below has since been answered by
# the DAOS launcher.
#
# It is NOT half-migrated on purpose. It carries the recipe's CCL block but
# WDS_LOCAL_SLICING=1, no --no-vni, and no per-rank log files -- correct for
# per-node staging, silently wrong on DAOS (every node computing the SAME 12
# slices). Deliberately left un-updated rather than partially updated: a script
# that looks current but isn't is the more dangerous artifact. Do not source
# scripts/lib/aurora_hsdp_env.sh into it without also fixing the slicing flag.
# ############################################################################
# 256-NODE SHAKEOUT (debug-scaling, 1h, free). Proves the mechanics of a 3072-rank
# run BEFORE any prod time is requested. This is deliberately NOT a training run:
# it answers four yes/no questions in order and stops.
#
#   1. Does per-node staging fit?  --partition-mode nodes replaces the old
#      global-world_size partition, which at 256n replicated every source onto
#      every node (3.1 TB vs ~503 GB /tmp) and aborted in the preflight. Offline
#      math on real shard bytes says ~264 GB/node at floor 48; this is the first
#      time it runs for real.
#   2. Does a 3072-rank rendezvous complete?  192 ranks already needed
#      TORCH_DIST_TIMEOUT_SECONDS=900 (memory aurora-16n-rendezvous-timeout);
#      3072 is 16x that, so the timeout is raised well above it here.
#   3. Do iterations actually run?  ~50 iters is enough to see a per-step time.
#   4. Is the PhaseTimer CSV sane?  Per-phase breakdown, not just wall time.
#
# EXPECTED: fabric spikes. docs/vitG_2B_HSDP_findings.md:205 calls 16n contention
# "the intrinsic tax -- accept it, make training survive it"; 256n will be worse.
# A slow-but-progressing run is a PASS here. Only a hang, an OOM, a staging abort
# or a rendezvous failure is a FAIL.
#
# debug-scaling allows ONE job queued/running per user -- check `qstat -u $USER`.
#
#   qsub scripts/vitG384_256n_shakeout.sh
#
# NOTE: no `set -u` (memory set-u-module-load-trap).
#
#PBS -N shake256
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=256
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
NNODES=256
PPN=12
WORLD=$(( NNODES * PPN ))
SHARD_FLOOR=${VJEPA_SHARD_FLOOR:-48}

# Config: the large-batch arm, whose EMA/warmup/lambda are already derived for a
# 16x batch. At 256n x 12 x bs2 the batch is REAL (6144), so TRUE_ACCUM stays 1 --
# accum was only the 16n emulation of this.
# Default to the bs=1 arm: job 8730000 validated per-rank batch_size=1, so the
# real 256n target is gb 3072 (3072 ranks x bs1), not 6144. lbA8's EMA/warmup/
# lambda are derived for exactly that. Override with VJEPA_CFG_NAME=vitG384_lbA
# to shake out the bs=2 fallback instead.
CFG_NAME=${VJEPA_CFG_NAME:-vitG384_lbA8}
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/${CFG_NAME}.yaml
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/shakeout_256n/${CFG_NAME}
# Iterations to execute. ~50 is plenty to read a per-iter time; the point is the
# mechanics, not the training.
SHAKE_IPE=${VJEPA_SHAKE_IPE:-50}
# STAGING BUDGET. This is the one thing that can eat the whole 1h slice. Measured
# on a real 16n run (job 8729208): 205 GB/node in 474 s = 0.43 GB/s/node, i.e.
# ~6.9 GB/s aggregate off Lustre. At 256 nodes the full corpus at floor 48 is
# 265 GB/node = 66 TB aggregate -- ~10 min if per-node bandwidth holds, but ~2.7 h
# if Lustre saturates near that 16n aggregate. We cannot know which until we try,
# and losing the slot to a half-finished copy proves nothing.
#
# So the shakeout stages a SUBSET of sources by default. Rank/rendezvous/iteration
# behaviour at 3072 ranks does not depend on how many distinct datasets are mixed,
# and the partition math is already unit-tested and dry-run verified against the
# full corpus. Set VJEPA_SHAKE_ALL_SOURCES=1 to stage everything once the timing
# is known.
SHAKE_ALL=${VJEPA_SHAKE_ALL_SOURCES:-0}
SHAKE_KEEP=${VJEPA_SHAKE_KEEP:-4}
PARAMS=$CKPT_DIR/params-pretrain.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
mkdir -p "$CKPT_DIR" /flare/ModCon/ngetty/logs

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID"
echo "topology: ${NNODES}n x ${PPN} = ${WORLD} ranks; config=$CFG_NAME; shard floor=$SHARD_FLOOR"
[[ -f "$BASE_CFG" ]] || { echo "FATAL: missing $BASE_CFG (run gen_large_batch_configs.py)"; exit 1; }

cd $ROOT
module load frameworks
export PYTHONNOUSERSITE=1
source /flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MPICH_GPU_SUPPORT_ENABLED=1
# Proven multi-node HSDP transport (launcher=none + ofi; pmix/mpi deadlocks FSDP
# subgroup collectives at iter0).
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
export VJEPA_DIST_STRATEGY=hsdp
export LOCAL_WORLD_SIZE=$PPN
export FSDP_SHARDING=shard_grad_op
export VJEPA_NUM_WORKERS=${VJEPA_NUM_WORKERS:-0}
export VJEPA_TRUE_ACCUM=1   # the 256n batch is real, not emulated
# QUESTION 2. 192 ranks needed 900s; 3072 ranks is 16x the rendezvous. Generous
# here on purpose -- a timeout would be indistinguishable from a real hang.
export TORCH_DIST_TIMEOUT_SECONDS=${TORCH_DIST_TIMEOUT_SECONDS:-3600}

# RENDEZVOUS IDENTITY. WORLD_SIZE must be exported explicitly: PMI/PMIx SIZE is
# unreliable across hosts of one multi-host task (different hosts report
# different values, silently breaking every collective), so
# src/utils/distributed.py:157 deliberately PREFERS an orchestrator-supplied
# WORLD_SIZE over PMI SIZE. Omitting it is the documented
# aurora-multi-mpi-per-pbs-worldsize failure, and it would be far worse at 3072
# ranks than at the 192 where it was found.
if [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=$WORLD
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

# Runtime config for THIS topology. --weak-scale keeps per-rank bs at 2, so the
# global batch is 6144 -- exactly what the lbA schedule was derived for.
$PY $ROOT/scripts/prepare_runtime_config.py \
  $BASE_CFG --root $ROOT --num-gpus $PPN --num-nodes $NNODES --weak-scale > /dev/null || {
    echo "FATAL: prepare_runtime_config failed"; exit 1; }
RUNTIME_CFG=$ROOT/.runtime_configs/n${NNODES}g${PPN}_weak/configs/vitg16_surg_vid_webdataset_single4/${CFG_NAME}.yaml
cp "$RUNTIME_CFG" "$PARAMS" || { echo "FATAL: no runtime config at $RUNTIME_CFG"; exit 1; }
$PY - "$PARAMS" "$CKPT_DIR" "$SHAKE_IPE" <<'PY'
import sys, yaml
p, d, ipe = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = yaml.safe_load(open(p))
c["folder"] = d
# Shakeout, not training: cap iterations so the run ends on its own inside the
# 1h slice instead of being cut mid-epoch. Schedules stay as-derived; only the
# number of steps we bother to execute changes.
c["optimization"]["ipe"] = ipe
c["optimization"]["epochs"] = 1
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
print(f"folder pinned to {d}; ipe={ipe} epochs=1 (shakeout)")
PY

if [[ "$SHAKE_ALL" != "1" ]]; then
  $PY - "$PARAMS" "$SHAKE_KEEP" <<'PY'
import sys, yaml, os
p, keep = sys.argv[1], int(sys.argv[2])
c = yaml.safe_load(open(p))
d = c["data"]
ds = d["datasets"]
# Keep the SMALLEST-on-disk sources: they exercise the identical code path at a
# fraction of the bytes. Size is approximated by shard count x mean shard size.
def nbytes(path):
    try:
        tars = [f for f in os.listdir(path) if f.endswith(".tar")]
        if not tars:
            return 0
        sample = tars[: min(5, len(tars))]
        mean = sum(os.stat(os.path.join(path, f)).st_size for f in sample) / len(sample)
        return mean * len(tars)
    except OSError:
        return 0
order = sorted(range(len(ds)), key=lambda i: nbytes(ds[i]))[:keep]
order.sort()
d["datasets"] = [ds[i] for i in order]
for k in ("datasets_weights", "dataset_fpcs"):
    if isinstance(d.get(k), list) and len(d[k]) == len(ds):
        d[k] = [d[k][i] for i in order]
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
print(f"SHAKEOUT SUBSET: {len(ds)} -> {len(d['datasets'])} sources "
      f"({', '.join(os.path.basename(x) for x in d['datasets'])})")
print("  set VJEPA_SHAKE_ALL_SOURCES=1 to stage the full corpus instead")
PY
fi

# ---- QUESTION 1: staging.
export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "=== staging to $LOCAL_DATA_ROOT (--partition-mode nodes, floor $SHARD_FLOOR) ==="
STAGE_T0=$(date +%s)
mpiexec -n $NNODES -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS \
        --local-root $LOCAL_DATA_ROOT \
        --num-nodes $NNODES --local-world-size $PPN --workers 8 \
        --partition-mode nodes --min-shards-per-node $SHARD_FLOOR
STAGE_RC=$?
echo "=== staging rc=$STAGE_RC in $(( $(date +%s) - STAGE_T0 ))s ==="
if (( STAGE_RC != 0 )); then
  echo "SHAKEOUT VERDICT: FAIL at Q1 (staging). Nothing downstream is testable."
  exit 1
fi
echo "--- per-node staged footprint (sample of 8) ---"
mpiexec -n $NNODES -ppn 1 --cpu-bind none bash -c \
  'echo "[$(hostname)] $(du -sh '"$LOCAL_DATA_ROOT"' 2>/dev/null | cut -f1) used, $(df -h /tmp | awk "NR==2{print \$4}") free"' \
  2>&1 | grep -viE "warn" | sort | head -8

# ---- QUESTIONS 2-4: rendezvous, iterations, CSV.
echo "=== launching $WORLD ranks (rendezvous timeout ${TORCH_DIST_TIMEOUT_SECONDS}s) ==="
TRAIN_T0=$(date +%s)
mpiexec -n $WORLD -ppn $PPN --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
TRAIN_RC=$?
TRAIN_DT=$(( $(date +%s) - TRAIN_T0 ))

# ---- Verdict from the CSV, not from the exit code: the walltime cut that ends a
# healthy shakeout is itself a nonzero rc, so rc alone cannot distinguish "ran
# fine until the hour expired" from "died at rank 0".
CSV=$CKPT_DIR/log_r0.csv
ROWS=$(awk -F, '$2 ~ /^[0-9]+$/ {n++} END{print n+0}' "$CSV" 2>/dev/null || echo 0)
# Verdict also to a DETERMINISTIC path -- PBS names its own log
# <jobid>.<server>.OU, which no watcher can predict.
VERDICT_FILE=/flare/ModCon/ngetty/logs/shake256_VERDICT.txt
{
  echo
  echo "================ SHAKEOUT SUMMARY ================"
  echo "job: ${PBS_JOBID:-interactive}  $(date)"
  echo "train rc=$TRAIN_RC  elapsed=${TRAIN_DT}s  csv rows=$ROWS  ranks=$WORLD"
  if [[ -f "$CSV" ]]; then
    echo "--- CSV header + first/last rows (PhaseTimer breakdown) ---"
    head -1 "$CSV"; sed -n '2p' "$CSV"; tail -2 "$CSV"
  fi
  if (( ROWS >= 20 )); then
    echo "SHAKEOUT VERDICT: PASS -- $ROWS iters at $WORLD ranks."
    echo "  Staging, rendezvous and the training loop all work at 256 nodes."
    echo "  Next: read the per-iter time above against the 16n baseline before prod."
  else
    echo "SHAKEOUT VERDICT: FAIL -- only $ROWS iters logged."
    echo "  Staging passed (Q1), so look at rendezvous/first-iter: grep the log for"
    echo "  DistStoreError, timeout, or a rank stack dump."
  fi
  echo "JOB END: $(date)"
} | tee "$VERDICT_FILE"
