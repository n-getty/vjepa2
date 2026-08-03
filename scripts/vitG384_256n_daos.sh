#!/bin/bash
# 256-node training reading from DAOS instead of node-local /tmp staging.
#
# WHAT CHANGED vs scripts/vitG384_256n_shakeout.sh
# ------------------------------------------------
# No staging step at all. The corpus lives once in AuroraGPT/vjepa_surg_wds and
# every node reads it over the fabric. Staging was copying 265 GB/node = 66 TB
# aggregate to deliver ~58 GB/node of actual reads (4.6x write amplification),
# at a measured 0.43 GB/s/node -- up to ~2.7 h before a single iteration.
# Streaming demand during the run is only ~1.5-3 GB/s aggregate, because 15 TB
# of reads spread over hours instead of minutes.
#
# THREE THINGS THAT MUST BE RIGHT (each fails silently or hangs):
#
#   WDS_LOCAL_SLICING=0. This is the subtle one. The flag exists only because
#   each node's /tmp held a DIFFERENT shard subset, so slicing by local
#   rank/world (12) was the correct way to cover it. On DAOS every node sees the
#   WHOLE corpus, so local slicing makes all 256 nodes compute the SAME 12
#   slices -- rank 0 on every node reads identical shards. That is 256x data
#   duplication with no error message. Global slicing (=0) gives each of the
#   3072 ranks a disjoint slice.
#
#   mpiexec --no-vni. Without it DAOS RPCs over libfabric fail with
#   NA_HOSTUNREACH and the job hangs.
#
#   libpil4dfs stays OFF. It hangs FSDP AllGather (DAOS-17499) and we run HSDP.
#   PRISM also measured it as no benefit -- data load is <0.01 s/step anyway.
#
# -l filesystems must include daos_user_fs (the agent only starts in the PBS
# prologue) -- but never add it to non-DAOS jobs: they queue forever when DAOS
# is down.
#
#   qsub scripts/vitG384_256n_daos.sh
#
# NOTE: no `set -u` (memory set-u-module-load-trap).
#
#PBS -N vg256d
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=256
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
POOL=${DAOS_POOL:-AuroraGPT}
CONT=${DAOS_CONT:-vjepa_surg_wds}
DAOS_MNT=/tmp/${POOL}/${CONT}
# Model weights live in their OWN container. Job 8730487 died learning why:
# the corpus was on DAOS but `pretrain_checkpoint` was still a 28.2 GB file on
# Lustre, and all 3072 ranks read it independently -- up to 85 TB against a
# filesystem measured at 2.71 GB/s. Ranks cleared the checkpoint load at only
# ~69/min, so the job would have spent its whole hour loading and never reached
# an iteration. Invisible at 16n (192 ranks x 28 GB is tolerable); fatal at 3072.
#
# Note the oclass differs from the corpus container ON PURPOSE: vjepa_models uses
# EC_16P3GX (stripe each file across ALL servers -- right for ONE big file every
# rank reads), while vjepa_surg_wds uses EC_16P3G32 (right for thousands of
# independent shards). Same reasoning, opposite answer.
MODELS_CONT=${DAOS_MODELS_CONT:-vjepa_models}
MODELS_MNT=/tmp/${POOL}/${MODELS_CONT}
PPN=12
NNODES=$(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | wc -l); NNODES=${NNODES:-256}
WORLD=$(( NNODES * PPN ))
SHAKE_IPE=${VJEPA_SHAKE_IPE:-50}

CFG_NAME=${VJEPA_CFG_NAME:-vitG384_lbA8}
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/${CFG_NAME}.yaml
CKPT_DIR=${VJEPA_CKPT_DIR:-/flare/ModCon/ngetty/checkpoints/daos_256n/${CFG_NAME}}
PARAMS=$CKPT_DIR/params-pretrain.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
VERDICT=/flare/ModCon/ngetty/logs/daos256_VERDICT.txt
mkdir -p "$CKPT_DIR" /flare/ModCon/ngetty/logs

echo "JOB START $(date) PBS_JOBID=$PBS_JOBID"
echo "topology ${NNODES}n x ${PPN} = ${WORLD} ranks; config=$CFG_NAME; data=DAOS $POOL:$CONT"
[[ -f "$BASE_CFG" ]] || { echo "FATAL: missing $BASE_CFG"; exit 1; }

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
export VJEPA_DIST_STRATEGY=hsdp
export LOCAL_WORLD_SIZE=$PPN
export FSDP_SHARDING=shard_grad_op
export VJEPA_NUM_WORKERS=${VJEPA_NUM_WORKERS:-0}
export VJEPA_TRUE_ACCUM=1
export TORCH_DIST_TIMEOUT_SECONDS=${TORCH_DIST_TIMEOUT_SECONDS:-3600}

# THE FLAG THAT MUST FLIP. See the header. =0 -> global-rank slicing.
export WDS_LOCAL_SLICING=0
# Explicitly ensure the FSDP-hanging interception library is not inherited.
unset LD_PRELOAD

if [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=$WORLD
echo "MASTER_ADDR=$MASTER_ADDR WORLD_SIZE=$WORLD_SIZE WDS_LOCAL_SLICING=$WDS_LOCAL_SLICING"

# Mount the container on every node.
launch-dfuse.sh ${POOL}:${MODELS_CONT} || { echo "FATAL: launch-dfuse (models) failed"; exit 1; }
timeout 60 ls "$MODELS_MNT" >/dev/null 2>&1 || { echo "FATAL: $MODELS_MNT unresponsive"; exit 1; }
echo "models container mounted: $(ls "$MODELS_MNT" 2>/dev/null | tr '\n' ' ')"
launch-dfuse.sh ${POOL}:${CONT} || { echo "FATAL: launch-dfuse failed"; exit 1; }
mount | grep -q "$CONT" || { echo "FATAL: not mounted at $DAOS_MNT"; exit 1; }
# A hung dfuse presents as a hang much later, in the loader, on one rank. Catch
# it here where the message is unambiguous.
timeout 60 ls "$DAOS_MNT" >/dev/null 2>&1 || { echo "FATAL: $DAOS_MNT unresponsive"; exit 1; }
echo "DAOS mounted; sources: $(ls "$DAOS_MNT" 2>/dev/null | tr '\n' ' ')"

$PY $ROOT/scripts/prepare_runtime_config.py \
  $BASE_CFG --root $ROOT --num-gpus $PPN --num-nodes $NNODES --weak-scale > /dev/null || {
    echo "FATAL: prepare_runtime_config failed"; exit 1; }
RUNTIME_CFG=$ROOT/.runtime_configs/n${NNODES}g${PPN}_weak/configs/vitg16_surg_vid_webdataset_single4/${CFG_NAME}.yaml
cp "$RUNTIME_CFG" "$PARAMS" || { echo "FATAL: no runtime config"; exit 1; }
$PY - "$PARAMS" "$CKPT_DIR" "$SHAKE_IPE" "$MODELS_MNT" <<'PY'
import sys, yaml, os
p, d, ipe, models = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
c = yaml.safe_load(open(p))
c["folder"] = d
c["optimization"]["ipe"] = ipe
c["optimization"]["epochs"] = 1
# Repoint the init checkpoint at DAOS. Leaving it on Lustre is what killed job
# 8730487: 3072 ranks each pulling 28.2 GB off a 2.71 GB/s filesystem.
meta = c.setdefault("meta", {})
ck = meta.get("pretrain_checkpoint")
if ck:
    cand = os.path.join(models, os.path.basename(ck))
    if os.path.exists(cand):
        meta["pretrain_checkpoint"] = cand
        print(f"pretrain_checkpoint -> {cand} (DAOS)")
    else:
        # Do not silently fall back to Lustre: that is precisely the failure we
        # are fixing, and at 3072 ranks it burns the whole allocation.
        sys.exit(f"FATAL: {cand} missing -- stage the checkpoint into DAOS first")
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
print(f"folder={d} ipe={ipe} epochs=1")
PY

# --local_data_root remaps every dataset dir by BASENAME
# (app/main_dist_aurora.py:366-370), so pointing it at the DAOS mount is a
# drop-in -- no config edit, no trainer change.
T0=$(date +%s)
mpiexec -n $WORLD -ppn $PPN --cpu-bind depth --depth 16 --no-vni \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $DAOS_MNT
RC=$?
DT=$(( $(date +%s) - T0 ))

CSV=$CKPT_DIR/log_r0.csv
ROWS=$(awk -F, '$2 ~ /^[0-9]+$/ {n++} END{print n+0}' "$CSV" 2>/dev/null || echo 0)
{
  echo "================ DAOS 256n ================"
  echo "job ${PBS_JOBID:-interactive}  $(date)"
  echo "rc=$RC elapsed=${DT}s rows=$ROWS ranks=$WORLD  (NO staging step)"
  [[ -f "$CSV" ]] && { echo "--- CSV ---"; head -1 "$CSV"; sed -n '2p' "$CSV"; tail -2 "$CSV"; }
  if (( ROWS >= 20 )); then
    echo "VERDICT: PASS -- $ROWS iters at $WORLD ranks reading DAOS."
    echo "  Compare per-iter time against the 16n baseline (~8 s) before prod."
  else
    echo "VERDICT: FAIL -- only $ROWS iters. Check dfuse mount, rendezvous, NA_HOSTUNREACH."
  fi
} | tee "$VERDICT"
echo "JOB END $(date)"
