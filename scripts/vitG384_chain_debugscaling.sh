#!/usr/bin/env bash
# Self-resubmitting ViT-G (2B) @ 384 surgical CPT chain, 1h walltime on
# debug-scaling. 2B sibling of vitg384_chain_debugscaling.sh (1B).
#
# Runs the 2B optimal CPT: undistilled Meta ViT-G @ native 384, context loss on
# (lambda 0.5), predictor warm-started, bf16 DDP comm hook + bucket50. See
# configs/vitg16_surg_vid_webdataset_single4/vitG384_cleandata.yaml and memory
# vitG-2b-swap for the full rationale (arch swap only + ipe/epochs re-slice).
#
# Strategy: identical to the 1B chain — debug-scaling backfills 1h slices while a
# capacity job (vitG_cap) waits for a large allocation and swaps in via the shared
# LOCK. BOTH point at the same CKPT_DIR and resume from latest.pth.tar.
# The 2B config is ipe=333/epochs=30 (vs 1B ipe500/epochs20), so one epoch is
# ~44min (~8s/iter est) + ~5min staging = fits 1h. VJEPA_EXIT_AFTER_CKPT=1 makes
# each slice exit cleanly after its epoch checkpoint instead of starting a partial
# second epoch that would be walltime-killed and discarded.
#
# Submit first instance:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/vitG384_chain_debugscaling.sh
#
#PBS -N vitG_chain
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=16
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/vitG384_cleandata.yaml
RUNTIME_CFG=$ROOT/.runtime_configs/n16g12_weak/configs/vitg16_surg_vid_webdataset_single4/vitG384_cleandata.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/surg_2_1_vitG384_cleandata/vitG384_n16g12_weak
PARAMS=$CKPT_DIR/params-pretrain.yaml
SELF=$ROOT/scripts/vitG384_chain_debugscaling.sh
LOCK=$CKPT_DIR/.training.lock
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID"

# Stage the runtime config into the ckpt dir as the authoritative params file.
if [[ ! -f "$PARAMS" ]]; then
  echo "Staging runtime cfg -> $PARAMS"
  $PY $ROOT/scripts/prepare_runtime_config.py \
    $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 16 --weak-scale > /dev/null
  cp $RUNTIME_CFG $PARAMS
fi

NUM_EPOCHS=$($PY -c "import yaml; print(yaml.safe_load(open('$PARAMS'))['optimization']['epochs'])")
CURRENT_EPOCH=0
if [[ -f $CKPT_DIR/latest.pth.tar ]]; then
  CURRENT_EPOCH=$($PY -c "import torch; print(torch.load('$CKPT_DIR/latest.pth.tar', map_location='cpu', weights_only=False).get('epoch', 0))" 2>/dev/null || echo 0)
fi
echo "progress: epoch $CURRENT_EPOCH / $NUM_EPOCHS"
if (( CURRENT_EPOCH >= NUM_EPOCHS )); then
  echo "ViT-G CPT complete (epoch >= num_epochs). chain stops."
  rm -f "$LOCK"
  exit 0
fi

# Keep the chain alive: resubmit a successor unless one is already queued, OR a
# capacity job has STARTED running (the swap-over). qstat -u columns: 4=jobname,
# 10=state. Match jobname (col 4).
CAP_RUNNING=$(qstat -u $USER 2>/dev/null | awk '$4=="vitG_cap" && $10=="R"' | wc -l)
DS_QUEUED=$(qstat -u $USER 2>/dev/null | awk '$4=="vitG_chain" && $10=="Q"' | wc -l)
if (( CAP_RUNNING >= 1 )); then
  echo "swap-over: capacity job running -> debug-scaling chain draining (no resubmit)"
elif (( DS_QUEUED >= 1 )); then
  echo "skip resubmit: $DS_QUEUED debug-scaling job already queued"
else
  NEXT_JOB=$(qsub $SELF 2>&1) || NEXT_JOB="(resubmit failed, watchdog re-arms: $NEXT_JOB)"
  echo "Chained next job (no dependency): $NEXT_JOB"
fi

# Lock guard: never train two slices against the same latest.pth.tar.
if [[ -f "$LOCK" ]]; then
  HOLDER=$(cat "$LOCK" 2>/dev/null || echo "")
  if [[ -n "$HOLDER" ]] && qstat "$HOLDER" 2>/dev/null | awk 'NR>2{print $5}' | grep -q '^R$'; then
    echo "LOCK held by running job $HOLDER — another slice is training. Skipping."
    echo "JOB END (skipped): $(date)"
    exit 0
  else
    echo "stale lock from $HOLDER (not running) — taking over."
  fi
fi
echo "$PBS_JOBID" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd $ROOT
module load frameworks
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
# --- ViT-g optimizations (validated 2026-06-26: -32% iter-time, no loss cost) ---
export VJEPA_BF16_COMM=1
export VJEPA_DDP_BUCKET_MB=50
# Debug-scaling chain: exit cleanly after the epoch checkpoint (one epoch/slice
# at ipe333 ~44min fits 1h). Avoids a walltime-killed partial 2nd epoch. The
# capacity job (vitG384_capacity.sh) does NOT set this — it runs continuously.
export VJEPA_EXIT_AFTER_CKPT=1
if [[ -f "${PBS_NODEFILE:-}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=192
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"
echo "OPT ENVS: VJEPA_BF16_COMM=$VJEPA_BF16_COMM VJEPA_DDP_BUCKET_MB=$VJEPA_DDP_BUCKET_MB VJEPA_EXIT_AFTER_CKPT=$VJEPA_EXIT_AFTER_CKPT"

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging shards to $LOCAL_DATA_ROOT (per-node disjoint) ---"
mpiexec -n 16 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS \
        --local-root $LOCAL_DATA_ROOT \
        --num-nodes 16 --local-world-size 12 --workers 8
echo "--- staging complete ---"

mpiexec --pmi=pmix -n 192 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
echo "JOB END: $(date)"
