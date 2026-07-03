#!/usr/bin/env bash
# 16-NODE SPIKE TEST — LEVER A: the ALLOCATOR / MR-CACHE env fix.
#
# CORRECTED DIAGNOSIS (2026-07-03): the 2B backward is not "fabric AR volume".
# Measured: 1B (bs=1) backward is FLAT 9.2s for 4519 iters; 2B (bs=2) backward
# CLIMBS monotonically 13s->328s by iter 42 while max_memory_allocated stays
# FLAT at 57.6GB (=90% of the 64GB tile). That is torchtune's documented Aurora
# bug "CCL external memory growth" (docs/reports/ccl_external_memory_growth_32b.md):
# CCL IPC-handle + OFI fabric-registration memory grows ~10-30 MiB/step OUTSIDE
# PyTorch's pool (invisible to max_memory_allocated). The 1B had L0 headroom to
# absorb it; the 2B at 90% occupancy starves L0 -> CCL stalls every backward,
# worsening. Torchtune's VALIDATED production block for this exact bug was set in
# ZERO of our 7 prior tests. This adds it.
#
# ISOLATION vs Test C (vitG384_spiketest_C_pt213_nohook.sh): byte-identical EXCEPT
# the four allocator/MR/IPC env vars below. Same torch 2.13 venv, no bf16 hook,
# pmix/mpi, fpcs16. So a flat-vs-climbing backward isolates the allocator env as
# the fix.
#
# PASS criterion: max backward-ms across all 192 ranks stays ~flat (no monotonic
# climb) through ipe120. FAIL: same climb as Test C -> Lever B (grad accum) needed.
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/vitG384_spiketest_G_alloc.sh
#
#PBS -N vitG_spkG
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=16
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/vitG384_cleandata.yaml
VENV=/flare/ModCon/ngetty/venvs/torchtune-pt213-xpu
PY_STAGE=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/SPIKETEST_vitG384/allocG_n16g12
RUNTIME_CFG=$ROOT/.runtime_configs/n16g12_weak/configs/vitg16_surg_vid_webdataset_single4/vitG384_cleandata.yaml
PARAMS=$CKPT_DIR/params-pretrain.yaml
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID  [LEVER A: allocator/MR env]"

$PY_STAGE $ROOT/scripts/prepare_runtime_config.py \
    $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 16 --weak-scale > /dev/null
$PY_STAGE - "$RUNTIME_CFG" "$PARAMS" "$CKPT_DIR" <<'PY'
import sys, yaml
src, dst, folder = sys.argv[1], sys.argv[2], sys.argv[3]
d = yaml.safe_load(open(src))
d["folder"] = folder
d["optimization"]["ipe"] = 120           # short: enough to see the climb (Test C wedged by ~42)
d["optimization"]["epochs"] = 1
d["meta"]["save_every_freq"] = 1000000000 # no checkpoint saving
yaml.safe_dump(d, open(dst, "w"), sort_keys=False)
print(f"patched test params -> {dst} (ipe120, epochs1, no-save, folder={folder})")
PY

cd $ROOT
module load frameworks
export PYTHONNOUSERSITE=1
source $VENV/bin/activate
python -c "import torch; print('torch', torch.__version__)" 2>&1 | grep -viE "UserWarning|warn" | head -1

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
export VJEPA_BF16_COMM=0  # Test C base: no IPC-handle-churning comm hook
export VJEPA_DDP_BUCKET_MB=50

# ================= LEVER A: the four validated allocator/MR/IPC vars =================
# Source: torchtune docs/features/allocator_strategy.md "Production Config (32B multi-node)"
# + docs/bugs/ccl_ipc_handle_cache.md. These were set in NONE of our 7 prior tests.
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95  # reclaim suballocs (stable L0 VAs), NOT segments -> no stale IPC handles
export FI_MR_CACHE_MONITOR=disabled                          # userfaultfd makes banned:1 WORSE (doc-tested); disable MR cache monitor
unset XPU_USM_ALLOC_SO                                       # ensure DEFAULT allocator (pluggable allocs OOM/banned:1 at scale)
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536         # prevent IPC-handle eviction -> stale-handle access at scale
echo "LEVER A env: PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF FI_MR_CACHE_MONITOR=$FI_MR_CACHE_MONITOR CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=$CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD XPU_USM_ALLOC_SO=${XPU_USM_ALLOC_SO:-unset}"
# ====================================================================================

MASTER_ADDR=$(head -n1 "$PBS_NODEFILE"); export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=192
echo "MASTER_ADDR=$MASTER_ADDR WORLD_SIZE=$WORLD_SIZE"

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging ---"
mpiexec -n 16 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS --local-root $LOCAL_DATA_ROOT \
        --num-nodes 16 --local-world-size 12 --workers 8
echo "--- staging complete ---"

mpiexec --pmi=pmix -n 192 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
echo "JOB END: $(date)"
