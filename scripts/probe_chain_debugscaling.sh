#!/usr/bin/env bash
# Self-resubmitting asformer-probe chain (debug queue, 1h slices). Drives ONE
# probe config to completion (num_epochs or early-stop) across multiple slices,
# since each 20-epoch Leo-exact probe (~7min/epoch, re-encodes frozen backbone
# every epoch with random aug) needs ~3 slices. Resumes via resume_checkpoint:
# true (loads latest.pt, continues from saved epoch).
#
# Dependency-free resubmit (NOT afterany — that sticks in system-hold when a
# slice is walltime-killed, see v2_final chain). LOCK guard serializes slices.
#
# Submit with:
#   qsub -v PROBE_CFG=configs/heads/sarrarp50/v2_probe/bs2_v2_e9.yaml \
#        scripts/probe_chain_debugscaling.sh
#
#PBS -N probe_chain
#PBS -A ModCon
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
: "${PROBE_CFG:?must pass -v PROBE_CFG=<probe yaml>}"
[[ "$PROBE_CFG" != /* ]] && PROBE_CFG="$ROOT/$PROBE_CFG"
[[ -f "$PROBE_CFG" ]] || { echo "config not found: $PROBE_CFG" >&2; exit 2; }
SELF=$ROOT/scripts/probe_chain_debugscaling.sh

module load frameworks
mkdir -p /flare/ModCon/ngetty/logs
echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID  PROBE_CFG=$PROBE_CFG"

# Resolve the probe's run folder + tag to find its CSV / lock.
FOLDER=$($PY -c "import yaml;print(yaml.safe_load(open('$PROBE_CFG'))['folder'])")
TAG=$($PY -c "import yaml;print(yaml.safe_load(open('$PROBE_CFG')).get('tag',''))")
NUM_EPOCHS=$($PY -c "import yaml;print(yaml.safe_load(open('$PROBE_CFG'))['experiment']['optimization']['num_epochs'])")
PATIENCE=$($PY -c "import yaml;print(yaml.safe_load(open('$PROBE_CFG')).get('early_stop_patience',6))")
CSV="$FOLDER/video_classification_frozen/$TAG/log_r0.csv"
LOCK="$FOLDER/.probe.lock"
mkdir -p "$FOLDER"

# Done? (reached num_epochs OR early-stop fired). Parse CSV if present.
if [[ -f "$CSV" ]]; then
  read LAST_EP BEST_EP < <($PY - "$CSV" <<'PYEOF'
import sys, csv
last_ep, best_ep, best_f1 = 0, 0, -1.0
with open(sys.argv[1]) as f:
    for row in csv.reader(f):
        if not row or not row[0].isdigit():
            continue
        ep = int(row[0]); f1 = float(row[14]) if len(row) > 14 and row[14] else -1
        last_ep = ep
        if f1 > best_f1:
            best_f1 = f1; best_ep = ep
print(last_ep, best_ep)
PYEOF
)
  echo "progress: last_ep=$LAST_EP best_ep=$BEST_EP / num_epochs=$NUM_EPOCHS patience=$PATIENCE"
  if (( LAST_EP >= NUM_EPOCHS )); then
    echo "probe complete (reached num_epochs). chain stops."; rm -f "$LOCK"; exit 0
  fi
  if (( LAST_EP - BEST_EP >= PATIENCE )); then
    echo "probe complete (early-stop: $((LAST_EP-BEST_EP)) >= $PATIENCE). chain stops."; rm -f "$LOCK"; exit 0
  fi
fi

# Keep chain alive: dependency-free successor unless one already queued.
DQ=$(qstat -u $USER 2>/dev/null | awk '$3=="debug" && $5=="Q"' | wc -l)
if (( DQ >= 1 )); then echo "skip resubmit: a debug job already queued"; else
  NEXT=$(qsub -v PROBE_CFG="$PROBE_CFG" $SELF); echo "chained next (no dep): $NEXT"; fi

# Lock guard.
if [[ -f "$LOCK" ]]; then
  H=$(cat "$LOCK" 2>/dev/null || echo "")
  if [[ -n "$H" ]] && qstat "$H" 2>/dev/null | awk 'NR>2{print $5}' | grep -q '^R$'; then
    echo "LOCK held by running $H — skip this slice."; echo "JOB END (skipped): $(date)"; exit 0
  else echo "stale lock from $H — taking over."; fi
fi
echo "$PBS_JOBID" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd $ROOT
export ZE_FLAT_DEVICE_HIERARCHY=FLAT MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_KVS_CONNECTION_TIMEOUT=600 CCL_OP_SYNC=1 CCL_WORKER_COUNT=1
export FI_PROVIDER=cxi PYTHONFAULTHANDLER=1 TMPDIR=/tmp OMP_NUM_THREADS=16
export http_proxy="http://proxy.alcf.anl.gov:3128" https_proxy="http://proxy.alcf.anl.gov:3128" ftp_proxy="http://proxy.alcf.anl.gov:3128"
if [[ -f "${PBS_NODEFILE:-}" ]]; then MASTER_ADDR=$(head -n1 "$PBS_NODEFILE"); else MASTER_ADDR=$(hostname); fi
export MASTER_ADDR MASTER_PORT=29610 WORLD_SIZE=24
echo "MASTER_ADDR=$MASTER_ADDR WORLD_SIZE=$WORLD_SIZE"

mpiexec --pmi=pmix -n 24 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname "$PROBE_CFG" --params_path "$PROBE_CFG"
echo "JOB END: $(date)"
