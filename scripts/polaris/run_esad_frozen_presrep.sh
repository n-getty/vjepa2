#!/usr/bin/env bash
# Frozen 3-seed head-training + scoring on PRESENCE-MODE=REP caches.
#
# This is run_esad_frozen_batch.sh with exactly two changes:
#   1. train/val presence labels are overridden at LOAD TIME from the
#      --presence-mode rep manifests. The caches themselves are reused
#      unchanged (features are bit-identical under both modes).
#   2. results land in runs/esad_double_<arm>_frozen_presrep_s<seed>/ so the
#      union-mode numbers on record are never overwritten.
#
# TEST-side caches are deliberately NOT patched: presence_mode changes only the
# TRAINING/SELECTION target. The scorer reads ground truth from the label files
# via --full-gt-label-dir, so the test population, gt_full/gt_cov/frames and the
# metric definition are byte-identical to the union runs. That is what makes
# this an apples-to-apples A/B: same test set, same denominator, only the
# training/selection labels differ.
#
# Submit (one arm's 3 seeds + one extra, 4 GPUs):
#   qsub -v COMBOS="v1:0 v1:1 v1:2 meta2b:0" run_esad_frozen_presrep.sh
#
#PBS -N esad_frozen_presrep
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o /eagle/projects/ModCon/ngetty/esad_probe/logs/

set -eo pipefail

COMBOS="${COMBOS:?set COMBOS=\"name:seed name:seed ...\" (up to 4)}"
read -ra COMBO_ARR <<< "$COMBOS"
NCOMBO=${#COMBO_ARR[@]}
if [[ $NCOMBO -gt 4 ]]; then echo "FATAL: at most 4 combos (1 node = 4 GPUs)"; exit 2; fi

WS="${WS:-/eagle/projects/tpc/leonardo_borgioli/esad_probe}"
REPO="${VJEPA_REPO:-/eagle/projects/ModCon/ngetty/repos/vjepa2}"
ESAD_ROOT="${ESAD_ROOT:-/eagle/projects/tpc/leonardo_borgioli/surg_vid/esad}"
CACHE_ROOT="${CACHE_ROOT:-/eagle/projects/ModCon/ngetty/esad_probe}"
MANIFEST_DIR="${MANIFEST_DIR:-/eagle/projects/ModCon/ngetty/esad_probe/manifests_cov}"
LABEL_DIR="${LABEL_DIR:-$ESAD_ROOT/test_labels}"
VENV="${VJEPA_VENV:-/eagle/projects/tpc/leonardo_borgioli/venvs/vjepa_polaris}"

echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-none} COMBOS=$COMBOS NCOMBO=$NCOMBO host=$(hostname)"

cd "$WS"
module use /soft/modulefiles
module load conda
conda activate
source "$VENV/bin/activate"

export PYTHONFAULTHANDLER=1
export TMPDIR=/tmp
export VJEPA_REPO="$REPO"
export PYTHONPATH="$WS:$REPO:${PYTHONPATH:-}"
unset PMI_RANK PMIX_RANK OMPI_COMM_WORLD_RANK PALS_RANKID RANK 2>/dev/null || true

WORKER="$WS/_presrep_worker_${PBS_JOBID%%.*}.sh"
cat > "$WORKER" <<'WEOF'
#!/bin/bash
set -eo pipefail
GPU="$1"
COMBO="$2"
export CUDA_VISIBLE_DEVICES="$GPU"
unset PMI_RANK PMIX_RANK OMPI_COMM_WORLD_RANK PALS_RANKID RANK 2>/dev/null || true

NAME="${COMBO%%:*}"
SEED="${COMBO##*:}"
echo "[$COMBO] GPU=$CUDA_VISIBLE_DEVICES starting (presence_mode=rep)"

# The ORIGINAL caches are reused unchanged -- presence_mode does not touch
# pixels, so the exported features are bit-identical. Only the presence LABEL
# is swapped, at load time, from the rep manifests (see
# train_esad_double_head.py --train/--val-presence-override). Copying the
# caches to change a [24,21] label would have duplicated ~123 GB per arm.
if [[ "$NAME" == "v1" ]]; then
  CACHE_DIR="$CACHE_ROOT/cache"
  PROBE_CFG="$WS/esad_double_probe.yaml"
  CACHE_TAG="v1"
else
  CACHE_DIR="$CACHE_ROOT/cache_${NAME}"
  PROBE_CFG="$WS/esad_double_probe_${NAME}.yaml"
  CACHE_TAG="$NAME"
fi

REP_DIR="$CACHE_ROOT/manifests_presrep_m1"
REP_TRAIN="$REP_DIR/esad_windows_train_m1_presrep.pt"
REP_VAL="$REP_DIR/esad_windows_val_m1_presrep.pt"

# Hard gate: refuse to run unless BOTH override manifests exist and actually
# declare presence_mode=rep. Without this, a missing manifest would silently
# reproduce the union baseline under a "presrep" run name -- a plausible number
# answering the wrong question, which is the exact failure class this whole
# investigation was opened by.
for f in "$REP_TRAIN" "$REP_VAL"; do
  if [[ ! -f "$f" ]]; then
    echo "[$COMBO] FATAL: missing rep manifest $f"; exit 3
  fi
  MODE=$(python3 -c "
import torch,sys
g=torch.load('$f',map_location='cpu',weights_only=False)['geometry']
print(g.get('presence_mode','union'))")
  if [[ "$MODE" != "rep" ]]; then
    echo "[$COMBO] FATAL: $f presence_mode=$MODE (expected rep)"; exit 3
  fi
done
echo "[$COMBO] rep-manifest gate passed"

RUN_DIR="$CACHE_ROOT/runs/esad_double_${NAME}_frozen_presrep_s${SEED}"
mkdir -p "$RUN_DIR"

RESULT_JSON="$RUN_DIR/test_detection_ap_cov_fulldenom.json"
if [[ -f "$RESULT_JSON" ]]; then
  echo "[$COMBO] already scored -> skip"
  exit 0
fi

echo "[$COMBO] training ..."
python "$WS/train_esad_double_head.py" \
  --config "$PROBE_CFG" \
  --train-cache "$CACHE_DIR/train" \
  --val-cache "$CACHE_DIR/val" \
  --test-cache "$CACHE_DIR/test" \
  --train-presence-override "$REP_TRAIN" \
  --val-presence-override "$REP_VAL" \
  --output-dir "$RUN_DIR" \
  --seed "$SEED" 2>&1 | tee "$RUN_DIR/train_log.txt"

# Prove the override actually engaged. The trainer prints this once per
# dataset; if it is absent the run silently trained on union labels.
# NOTE: the tee above is load-bearing. Without it nothing writes train_log.txt,
# the grep fails open on a missing file, and this guard passes vacuously --
# which is exactly what it did until 2026-08-25.
if ! grep -q "presence override ACTIVE" "$RUN_DIR/train_log.txt" 2>/dev/null; then
  echo "[$COMBO] FATAL: 'presence override ACTIVE' absent -- trained on UNION labels"
  exit 4
fi

# Scoring is IDENTICAL to the union runs: same coverage test caches, same
# manifests, same --full-gt-label-dir. Only the trained head differs.
echo "[$COMBO] scoring ..."
python "$WS/score_esad_detection_ap.py" \
  --config "$PROBE_CFG" \
  --best-ckpt "$RUN_DIR/best.pt" \
  --test-cache "$CACHE_ROOT/cache_${CACHE_TAG}_cov_p0/test" \
  --test-windows "$MANIFEST_DIR/esad_windows_test_cov_p0.pt" \
  --test-cache "$CACHE_ROOT/cache_${CACHE_TAG}_cov_p1/test" \
  --test-windows "$MANIFEST_DIR/esad_windows_test_cov_p1.pt" \
  --full-gt-label-dir "$LABEL_DIR" \
  --output "$RESULT_JSON"

python3 -c "
import json
d = json.load(open('$RESULT_JSON'))
b = d['combined_not_isolated']
v = b['variants_full_denominator']['maxpick']['ap_mean']
print('[$COMBO] REP full-denominator maxpick ap_mean:', v)
print('[$COMBO] gt_full', b['gt_instances_full'], 'gt_cov', b['gt_instances_covered'], 'frames', b['num_frames_scored'])
"
echo "[$COMBO] DONE"
WEOF
chmod +x "$WORKER"
trap 'rm -f "$WORKER"' EXIT

export CACHE_ROOT MANIFEST_DIR LABEL_DIR WS

PIDS=()
for i in "${!COMBO_ARR[@]}"; do
  echo "[batch] launching GPU=$i combo=${COMBO_ARR[$i]}"
  "$WORKER" "$i" "${COMBO_ARR[$i]}" > "$WS/_presrep_${PBS_JOBID%%.*}_gpu${i}.log" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[batch] combo ${COMBO_ARR[$i]} (pid ${PIDS[$i]}) exited OK"
  else
    echo "[batch] combo ${COMBO_ARR[$i]} (pid ${PIDS[$i]}) FAILED"
    FAIL=1
  fi
done

echo "=== per-combo logs ==="
for i in "${!COMBO_ARR[@]}"; do
  echo "--- GPU $i (${COMBO_ARR[$i]}) ---"
  cat "$WS/_presrep_${PBS_JOBID%%.*}_gpu${i}.log"
  rm -f "$WS/_presrep_${PBS_JOBID%%.*}_gpu${i}.log"
done

echo "JOB END: $(date)"
exit $FAIL
