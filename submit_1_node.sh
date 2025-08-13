#!/bin/bash -l
#PBS -l walltime=0:30:00
#PBS -l select=1
#PBS -N vjepa2_1_node_test
#PBS -k doe
#PBS -j oe
#PBS -A AuroraGPT
#PBS -l filesystems=home:flare
#PBS -q debug

# Set up environment
module load frameworks

# Set oneCCL environment variables
export CCL_ALLREDUCE=topo
export CCL_ALLREDUCE_SCALEOUT=direct
export CCL_ALLGATHERV=topo
export CCL_ALLGATHERV_SCALEOUT=direct

# Set affinity
export CCL_WORKER_AFFINITY=1,9,17,25,33,41,53,61,69,77,85,93
export CPU_BIND="list:2-8:10-16:18-24:26-32:34-40:42-48:54-60:62-68:70-76:78-84:86-92:94-100"
export NUMEXPR_MAX_THREADS=7
export OMP_NUM_THREADS=7

# Set distributed training parameters
export NGPU=12
export NNODES=1
export WORLD_SIZE=$((NGPU*NNODES))
export MASTER_ADDR=$(head -n 1 ${PBS_NODEFILE})
export MASTER_PORT=29500

# Go to the submission directory
cd $PBS_O_WORKDIR

# Launch the training
mpiexec -n ${NGPU} --ppn ${NGPU} --cpu-bind=${CPU_BIND} \
    python -u train_xpu.py --config configs/train/vitg16/droid-256px-8f.yaml
