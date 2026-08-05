#!/usr/bin/env bash
# Aurora (PBS + mpiexec, Intel XPU) launcher for ONE SAR-RARP50 ASFormer probe.
#
# Runs evals.video_classification_frozen (classifier.name=asformer) on a frozen
# V-JEPA 2.1 surgical encoder checkpoint, end-to-end, via app.main_dist_aurora
# --train_mode (which routes to evals.scaffold.main when the YAML has eval_name).
#
# Topology: 2 nodes x 8 tiles = 16 ranks => global batch 64 (batch 4/rank),
# matching Leo's Polaris 4node x 4gpu baseline. Uses the `debug` queue
# (<=2 nodes, 1h), which is a SEPARATE max_run=1 slot from `debug-scaling`,
# so this does NOT compete with the v1 training chain.
#
# The probe is resumable (resume_checkpoint: true, save_every_iters: 200) and
# early-stops (patience 6). If 20 epochs don't fit in 1h, just resubmit the
# same config — it resumes from <folder>/.../latest.pt.
#
# Submit with:
#   PROBE_CFG=configs/heads/sarrarp50/aurora_v1lambda/v1_e29.yaml \
#     qsub -v PROBE_CFG scripts/run_asformer_probe_aurora.sh
#
#PBS -N asformer_probe
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
PROBE_CFG="${PROBE_CFG:?must set PROBE_CFG to an eval yaml (relative to repo root or absolute)}"

# Resolve config to absolute path.
if [[ "$PROBE_CFG" != /* ]]; then
  PROBE_CFG="$ROOT/$PROBE_CFG"
fi
[[ -f "$PROBE_CFG" ]] || { echo "config not found: $PROBE_CFG" >&2; exit 2; }

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID"
echo "PROBE_CFG=$PROBE_CFG"

cd $ROOT
module load frameworks

# --- Aurora XPU + oneCCL env (same block as the pretrain chain) ---
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

if [[ -f "${PBS_NODEFILE:-}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=29610
# Topology DERIVED from the actual PBS allocation, not hardcoded — so a 1-node
# (`-l select=1`) submission Just Works and can't silently mismatch the mpiexec
# geometry (the old `WORLD_SIZE=24` + `-n 24` were pinned to select=2 and would
# desync if select changed). PPN (tiles/rank-per-node) defaults to 12 (full
# Aurora node); override with VJEPA_PPN. Full node = 12 tiles.
NUM_NODES=$(sort -u "${PBS_NODEFILE:?PBS_NODEFILE unset}" | wc -l)
PPN="${VJEPA_PPN:-12}"
export WORLD_SIZE=$(( NUM_NODES * PPN ))

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NUM_NODES=$NUM_NODES PPN=$PPN WORLD_SIZE=$WORLD_SIZE"

# VideoDataset (CSV-of-paths) reads clips directly from flare. The LIVE/uncached
# probe is bottlenecked on random small-file Lustre reads; opt into node-local
# tmpfs staging with VJEPA_STAGE_CLIPS=1 (corpus is ~13 GB, fits /tmp). Cached
# probes read .pt feature shards and do NOT need this (they use mmap instead).
LOCAL_DATA_ROOT=""
if [[ "${VJEPA_STAGE_CLIPS:-0}" == "1" ]]; then
  export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
  echo "=== staging probe clips to $LOCAL_DATA_ROOT on $NUM_NODES node(s) ==="
  mpiexec --pmi=pmix -n "$NUM_NODES" -ppn 1 --cpu-bind none \
      python "$ROOT/scripts/stage_probe_clips.py" \
          --config "$PROBE_CFG" --local-root "$LOCAL_DATA_ROOT" --workers 16 \
      2>&1 | tail -6
fi

# --local_data_root triggers the CSV path-rewrite in app.main_dist_aurora (only
# repoints dataset_{train,val} when the local prefix-swapped CSVs exist).
LOCAL_FLAG=()
[[ -n "$LOCAL_DATA_ROOT" ]] && LOCAL_FLAG=(--local_data_root "$LOCAL_DATA_ROOT")

# Probe seed for 3-seed runs (default 0 = prior single-seed behavior). Exported
# like the CCL_*/ZE_* vars above; PALS forwards the environment to every rank, so
# eval.py's VJEPA_PROBE_SEED (np/torch manual_seed) is set per rank.
export VJEPA_PROBE_SEED="${VJEPA_PROBE_SEED:-0}"

mpiexec --pmi=pmix -n "$WORLD_SIZE" -ppn "$PPN" --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname "$PROBE_CFG" --params_path "$PROBE_CFG" "${LOCAL_FLAG[@]}"

echo "JOB END: $(date)"
