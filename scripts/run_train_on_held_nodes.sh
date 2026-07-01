#!/usr/bin/env bash
# Run V-JEPA 2.1 trainer interactively on a held N-node Aurora allocation.
# Usage:
#   bash scripts/run_train_on_held_nodes.sh <jobid> <nnodes> <label> [<bucket_mb>] [<pred_static>]
#
# e.g.
#   bash scripts/run_train_on_held_nodes.sh 8527134 2 baseline    25 0
#   bash scripts/run_train_on_held_nodes.sh 8527134 2 bucket100  100 0
#   bash scripts/run_train_on_held_nodes.sh 8527134 2 pred_static 25 1
#
# Writes outputs under /flare/ModCon/ngetty/checkpoints/opt_test_v1/<label>/
# Lives in the repo so multiple opt runs are reproducible.
set -euo pipefail
jobid=${1:?jobid required}
nnodes=${2:?nnodes required}
label=${3:?label required}
bucket=${4:-25}
pred_static=${5:-0}

# Pick first allocated host from qstat (PBS_NODEFILE lives on compute nodes, not login)
host=$(qstat -f $jobid 2>&1 | awk '/exec_vnode/' | head -1 | tr -d '()' | cut -d= -f2 | tr '+' '\n' | head -1 | cut -d: -f1 | tr -d ' ')
[[ -z "$host" ]] && { echo "could not resolve host for $jobid"; exit 2; }
echo "ssh -> $host"

ssh -o BatchMode=yes "$host" bash -lc "'
source /etc/profile >/dev/null 2>&1
module -q load frameworks
export PBS_NODEFILE=\$(ls /var/spool/pbs/aux/${jobid}* 2>/dev/null | head -1)
[[ -z \"\$PBS_NODEFILE\" ]] && { echo \"no PBS_NODEFILE for ${jobid} on \$(hostname)\"; exit 2; }
export MASTER_ADDR=\$(head -n1 \"\$PBS_NODEFILE\")
export MASTER_PORT=29503
echo \"hosts:\"; sort -u \"\$PBS_NODEFILE\" | sed \"s/^/  /\"

# Aurora env block
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
# Critical: per-rank NIC pinning. Without this, each of 12 ranks/node opens
# all 8 NICs and exhausts CXI PTE entries (\"CXI alloc failed: request
# exceeds PTEs limits\"). Set by PBS prolog on production jobs but not
# inherited through ssh.
export MPIR_CVAR_CH4_OFI_MAX_NICS=1
export OMP_NUM_THREADS=16
export TMPDIR=/tmp
export PYTHONFAULTHANDLER=1
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# Variables we vary
export VJEPA_DDP_BUCKET_MB=$bucket
export VJEPA_PRED_STATIC=$pred_static
echo \"label=$label bucket=$bucket pred_static=$pred_static\"

cd /lus/flare/projects/ModCon/ngetty/vjepa2

cfg=configs/vitl16_surg_vid_webdataset_single4/pretrain-256px-16f.yaml
runtime_cfg=\$(python scripts/prepare_runtime_config.py \\
    \"\$cfg\" \\
    --root /lus/flare/projects/ModCon/ngetty/vjepa2 \\
    --num-gpus 12 --num-nodes $nnodes --weak-scale \\
    --folder-base /flare/ModCon/ngetty/checkpoints/opt_test_v1/$label)
echo \"runtime: \$runtime_cfg\"

folder=\$(python -c \"import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))[\\\"folder\\\"])\" \"\$runtime_cfg\")
echo \"folder: \$folder\"
mkdir -p \"\$folder\"
params=\$folder/params-pretrain.yaml
cp \"\$runtime_cfg\" \"\$params\"

total_ranks=\$(( $nnodes * 12 ))
mpiexec --pmi=pmix -n \$total_ranks -ppn 12 --cpu-bind depth --depth 16 \\
    python -m app.main_dist_aurora --train_mode \\
        --fname \"\$params\" --params_path \"\$params\"
'"
