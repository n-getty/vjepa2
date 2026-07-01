#!/usr/bin/env bash
# Verification job: 2 nodes on debug queue. Tests:
#   1. stage_node_shards.py distributes shards correctly across 2 nodes
#   2. WebDataset loader picks up local /tmp paths via --local_data_root
#   3. Training step completes ~60 iters with no flare I/O during steady-state
#
# Outputs to /flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_verify_n2g12_resharded/
# so it does NOT interfere with the 16n production chain.
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/verify_staging_pbs.sh
#
#PBS -N vjepa_verify_stage
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
PARAMS=/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_verify_n2g12_resharded/params-pretrain.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID"
echo "Nodes: 2 Total ranks: 24 ppn: 12"

cd $ROOT

ulimit -c unlimited
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

# Tell WebDataset loader to slice by local rank/world_size, not global.
# This is correct because each node's local data root holds exactly the
# union of its LOCAL_WORLD_SIZE ranks' shards, so disjoint slicing across
# local ranks gives full coverage with no cross-node opens.
export WDS_LOCAL_SLICING=1

if [[ -f "${PBS_NODEFILE:-}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=29501
export WORLD_SIZE=24

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

# --- Per-node WebDataset shard staging ---
# 2 nodes x 12 ranks = world_size 24. Each rank reads urls[r::24].
# Node N stages union of slices for r in [12N..12N+11]: indices where
# (i % 24) // 12 == N. Each node gets exactly half the shards of every
# dataset that has >= 24 shards; tiny datasets stage in full.
export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging shards to $LOCAL_DATA_ROOT (per-node disjoint, n=2) ---"
mpiexec -n 2 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS \
        --local-root $LOCAL_DATA_ROOT \
        --num-nodes 2 --local-world-size 12 --workers 8
echo "--- staging complete ---"

# Sanity: each node reports /tmp usage and shard count for the first dataset
echo "--- post-staging sanity (per node) ---"
mpiexec -n 2 -ppn 1 --cpu-bind none bash -c '
echo "[$HOSTNAME] /tmp free: $(df -h /tmp | tail -1 | awk "{print \$4}")"
echo "[$HOSTNAME] LOCAL_DATA_ROOT contents:"
ls -1 $LOCAL_DATA_ROOT 2>/dev/null | sed "s/^/  /"
for ds in surgvu24 sitl kinetics400; do
  if [[ -d $LOCAL_DATA_ROOT/$ds ]]; then
    n=$(ls $LOCAL_DATA_ROOT/$ds/*.tar 2>/dev/null | wc -l)
    sz=$(du -sh $LOCAL_DATA_ROOT/$ds 2>/dev/null | awk "{print \$1}")
    echo "[$HOSTNAME] $ds: $n shards, $sz"
  fi
done
'
echo "--- end sanity ---"

mpiexec --pmi=pmix -n 24 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT

echo "JOB END: $(date)"
