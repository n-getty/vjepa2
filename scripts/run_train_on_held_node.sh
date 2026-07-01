#!/usr/bin/env bash
# Run the V-JEPA 2.1 trainer on the held node directly via mpiexec.
# Usage:  bash scripts/run_train_on_held_node.sh <host> <jobid> [<config>]
set -euo pipefail
host=${1:?host required}
jobid=${2:?jobid required}
cfg=${3:-configs/vitl16_surg_vid_webdataset_single4/pretrain-256px-16f.yaml}

ssh -o BatchMode=yes "$host" bash -lc "'
source /etc/profile >/dev/null 2>&1
module -q load frameworks
export PBS_NODEFILE=\$(ls /var/spool/pbs/aux/${jobid}* 2>/dev/null | head -1)
export MASTER_ADDR=\$(head -n1 \"\$PBS_NODEFILE\")
export MASTER_PORT=29502
echo \"PBS_NODEFILE=\$PBS_NODEFILE MASTER_ADDR=\$MASTER_ADDR\"

# Aurora env block (production multi-node from torchtune CLAUDE.md)
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
export OMP_NUM_THREADS=16
export TMPDIR=/tmp
export PYTHONFAULTHANDLER=1
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

cd /lus/flare/projects/ModCon/ngetty/vjepa2

# Generate runtime cfg (weak scaling: keep per-rank bs=4, global = 12)
runtime_cfg=\$(python scripts/prepare_runtime_config.py \\
    \"$cfg\" \\
    --root /lus/flare/projects/ModCon/ngetty/vjepa2 \\
    --num-gpus 12 --num-nodes 1 --weak-scale \\
    --folder-base /flare/ModCon/ngetty/checkpoints/dryrun)
echo \"runtime: \$runtime_cfg\"

# Make folder, dump params there for the trainer
folder=\$(python -c \"import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))[\\\"folder\\\"])\" \"\$runtime_cfg\")
echo \"folder: \$folder\"
mkdir -p \"\$folder\"
params=\$folder/params-pretrain.yaml
cp \"\$runtime_cfg\" \"\$params\"

mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 \\
    python -m app.main_dist_aurora --train_mode \\
        --fname \"\$params\" --params_path \"\$params\"
'"
