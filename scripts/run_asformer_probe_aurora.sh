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
# 2 nodes x 12 tiles = 24 ranks (FULL node utilization; global batch 96, batch
# 4/rank). Earlier 16-rank (-ppn 8) config left 4 tiles/node idle to match
# Leo's Polaris global batch 64 exactly; we standardize on full-node here and
# accept global batch 96 — still within run-to-run F1 noise vs the 75.12 anchor.
export WORLD_SIZE=24

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

# VideoDataset (CSV-of-paths) reads clips directly from flare — no webdataset
# /tmp staging. Do NOT set WDS_LOCAL_SLICING here.

mpiexec --pmi=pmix -n 24 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname "$PROBE_CFG" --params_path "$PROBE_CFG"

echo "JOB END: $(date)"
