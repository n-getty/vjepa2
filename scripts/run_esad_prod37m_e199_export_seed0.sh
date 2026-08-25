#!/usr/bin/env bash
# ESAD double-head probe -- export (train/val/test) + seed-0 head-train for
# vitG384_prod_37M's FINAL checkpoint (e199, 200 epochs, run complete
# 2026-08-21). Same pattern as prod37m_e54/prod9M_e29/prod18M_e59
# (2B ViT-gigantic@384, same architecture/geometry) -- only the checkpoint
# and output paths differ. Seeds 1/2 are trained separately by
# run_esad_prod37m_e199_seed_train.sh, reusing this job's export cache (frozen
# encoder -> identical features across seeds).
#
# Uses presence_mode=rep (the 2026-08-17 fix) directly at build time via
# --presence-override flags in the seed-train step -- see
# run_esad_frozen_presrep.sh for the pattern this borrows. §4h measured a
# clean null between union/rep (all |delta| <= 0.0088, an order of magnitude
# inside the probe's ~0.06 resolution limit), so rep is free correctness with
# no cost; used here since e199 is the lineage's final checkpoint.
#
#   ssh polaris.alcf.anl.gov
#   qsub run_esad_prod37m_e199_export_seed0.sh
#
#PBS -N esad_prod37m_e199_export
#PBS -A ModCon
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o /eagle/projects/ModCon/ngetty/esad_probe/logs/

set -eo pipefail

WS="${WS:-/eagle/projects/tpc/leonardo_borgioli/esad_probe}"
REPO="${VJEPA_REPO:-/eagle/projects/ModCon/ngetty/repos/vjepa2}"
ESAD_ROOT="${ESAD_ROOT:-/eagle/projects/tpc/leonardo_borgioli/surg_vid/esad}"
CACHE_ROOT="${CACHE_ROOT:-/eagle/projects/ModCon/ngetty/esad_probe}"
CACHE_DIR="$CACHE_ROOT/cache_prod37m_e199"
EXPORT_CFG="${EXPORT_CFG:-$WS/esad_double_export_polaris_prod37m_e199.yaml}"
PROBE_CFG="${PROBE_CFG:-$WS/esad_double_probe_prod37m_e199.yaml}"
RUN_DIR="$CACHE_ROOT/runs/esad_double_prod37m_e199_frozen_presrep_s0"
REP_DIR="${REP_DIR:-$CACHE_ROOT/manifests_presrep_m1}"
REP_TRAIN="$REP_DIR/esad_windows_train_m1_presrep.pt"
REP_VAL="$REP_DIR/esad_windows_val_m1_presrep.pt"
EXPORT_BS="${EXPORT_BS:-2}"
VENV="${VJEPA_VENV:-/eagle/projects/tpc/leonardo_borgioli/venvs/vjepa_polaris}"
PPN="${VJEPA_PPN:-4}"

echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-none}"
[[ -f "$EXPORT_CFG" ]] || { echo "FATAL: missing $EXPORT_CFG"; exit 2; }
[[ -f "$PROBE_CFG" ]] || { echo "FATAL: missing $PROBE_CFG"; exit 2; }
[[ -d "$REPO" ]] || { echo "REPO not found: $REPO" >&2; exit 2; }
[[ -f "$REP_TRAIN" ]] || { echo "FATAL: missing rep manifest $REP_TRAIN"; exit 2; }
[[ -f "$REP_VAL" ]] || { echo "FATAL: missing rep manifest $REP_VAL"; exit 2; }

cd "$WS"
module use /soft/modulefiles
module load cudatoolkit-standalone/13.0.1 PrgEnv-gnu cray-mpich
export LD_LIBRARY_PATH="/home/ngetty/.local/lib_shim:${LD_LIBRARY_PATH:-}"
export PATH="$VENV/bin:$PATH"

export MPICH_GPU_SUPPORT_ENABLED=1
export FI_PROVIDER=cxi
export PYTHONFAULTHANDLER=1
export TMPDIR=/tmp
export OMP_NUM_THREADS=8
export VJEPA_REPO="$REPO"
export PYTHONPATH="$WS:$REPO:${PYTHONPATH:-}"

mkdir -p "$RUN_DIR" "$CACHE_DIR" /eagle/projects/ModCon/ngetty/esad_probe/logs

RANKWRAP="$WS/_polaris_rankwrap_${PBS_JOBID%%.*}.sh"
cat > "$RANKWRAP" <<'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES="${PALS_LOCAL_RANKID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}"
exec "$@"
EOF
chmod +x "$RANKWRAP"
trap 'rm -f "$RANKWRAP"' EXIT

if [[ -f "${PBS_NODEFILE:-}" ]]; then
  NUM_NODES=$(sort -u "$PBS_NODEFILE" | wc -l)
else
  NUM_NODES=1
fi
WORLD_SIZE=$(( NUM_NODES * PPN ))
echo "NUM_NODES=$NUM_NODES PPN=$PPN WORLD_SIZE=$WORLD_SIZE"

export_split () {
  local split="$1"
  local cache_dir="$CACHE_DIR/$split"
  local marker="$cache_dir/rank_0000/manifest.json"
  if [[ -f "$marker" ]] && grep -q '"status": "completed"' "$marker" 2>/dev/null; then
    echo "[EXPORT] $split cache already complete -> skip"
    return 0
  fi
  echo "[EXPORT] $split -> $cache_dir"
  mpiexec -envall -n "$WORLD_SIZE" -ppn "$PPN" --cpu-bind depth -d 8 \
    "$RANKWRAP" python "$WS/export_esad_cache.py" \
      --config "$EXPORT_CFG" \
      --windows "$WS/esad_windows_${split}.pt" \
      --esad-root "$ESAD_ROOT" \
      --output-dir "$cache_dir" \
      --repo-root "$REPO" \
      --batch-size "$EXPORT_BS"
}

export_split train
export_split val
export_split test

echo "[TRAIN] prod37M_e199 seed=0 joint head training (presence_mode=rep) -> $RUN_DIR"
mpiexec -envall -n 1 -ppn 1 --cpu-bind depth -d 8 \
  "$RANKWRAP" python "$WS/train_esad_double_head.py" \
    --config "$PROBE_CFG" \
    --train-cache "$CACHE_DIR/train" \
    --val-cache "$CACHE_DIR/val" \
    --test-cache "$CACHE_DIR/test" \
    --train-presence-override "$REP_TRAIN" \
    --val-presence-override "$REP_VAL" \
    --output-dir "$RUN_DIR" \
    --seed 0 2>&1 | tee "$RUN_DIR/train_log.txt"

# The tee above is load-bearing: without it nothing writes train_log.txt, the
# grep fails open on a missing file, and this guard passes vacuously -- which
# is what it did until 2026-08-25.
if ! grep -q "presence override ACTIVE" "$RUN_DIR/train_log.txt" 2>/dev/null; then
  echo "FATAL: 'presence override ACTIVE' absent -- trained on UNION labels"
  exit 4
fi

echo "JOB END: $(date)"
