#!/usr/bin/env bash
#PBS -A ModCon
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o /eagle/projects/ModCon/ngetty/esad_probe/logs/
#
# Frozen + AUGMENTATION-ONLY arm for one checkpoint, 3 seeds, on POLARIS.
# One node = 4x A100 (Polaris is force_exclhost; there is no per-GPU queue),
# so this does: 4-way parallel export -> 3 seeds in PARALLEL -> 3 scores in
# parallel. ~40 min export + ~70 min training, all three seeds at once.
#
# WHY POLARIS AND NOT SOPHIA. Sophia's by-gpu allows max_run=5 / max_queued=20
# per PROJECT and the FT campaign already holds 17 of those slots. Polaris
# `preemptable` allows 10 concurrent single-node jobs at Priority 155 (the
# highest on the machine) with a 72h walltime -- so the six checkpoints fit at
# once instead of drip-feeding behind FT. Note `debug` is NOT usable here: its
# 1h cap already walltime-killed two of these runs at epoch 18/20 (7551194,
# 7551483).
#
# WHY augonly AND NOT aug. `aug` passes --train-cache twice (deterministic +
# augmented), DOUBLING windows/epoch 2468 -> 4936, so its gain conflates
# augmentation with 2x data. `augonly` passes only the augmented cache: same
# 2468 windows, same optimiser steps, only the pixels differ. That is the arm
# whose delta is attributable. See ledger 4b-frozctl.
#
# EACH ARM MIRRORS ITS OWN BASELINE, which differ by family:
#   vjepa (meta2b/meta1b/ours1b_e19) -> frozen_presrep: presence overrides,
#     cache_<n>, scored cov_p0+cov_p1 -> combined_not_isolated (5903 frames).
#   ext (lemonfm/snx/endovit) -> <n>_t1: tubelet-1 manifests, cache_<n>_t1, NO
#     presence override, scored fairness-masked to V-JEPA's covered stems.
# Mixing the two yields a number comparable to nothing.
set -uo pipefail
NAME="${NAME:?set NAME=}"
FAMILY="${FAMILY:?set FAMILY=vjepa|ext}"

WS=/eagle/projects/tpc/leonardo_borgioli/esad_probe
REPO=/eagle/projects/ModCon/ngetty/repos/vjepa2
CR=/eagle/projects/ModCon/ngetty/esad_probe
MAN=$CR/manifests_cov
REP=$CR/manifests_presrep_m1
ESAD_ROOT=/eagle/projects/tpc/leonardo_borgioli/surg_vid/esad
LABEL_DIR=$ESAD_ROOT/test_labels
CFG=$WS/esad_double_probe_${NAME}.yaml
ECFG=$WS/esad_double_export_polaris_${NAME}.yaml

if [ "$FAMILY" = "vjepa" ]; then
  BASE=$CR/cache_${NAME}; AUGC=$CR/cache_${NAME}_augtrain
  TRAIN_WIN=$WS/esad_windows_train.pt; TAG=frozen_augonly
else
  BASE=$CR/cache_${NAME}_t1; AUGC=$CR/cache_${NAME}_t1_augtrain
  TRAIN_WIN=$MAN/esad_windows_train_tubelet1_m1.pt; TAG=t1_augonly
fi

# ---- Polaris env. NOT `module load conda`: that is broken site-wide (unknown
# gcc-native/14.2, cray-hdf5-parallel) and Lmod EXITS 0 anyway, so the next
# `conda activate` dies rc=127 and under `set -e` kills the job before the venv
# is ever sourced. Source conda.sh directly, as run_esad_double_probe_polaris.sh
# does -- that is the one script of 57 that actually works here.
module use /soft/modulefiles
source /soft/applications/conda/2025-09-28/mconda3/etc/profile.d/conda.sh
conda activate base
source /eagle/projects/tpc/leonardo_borgioli/venvs/vjepa_polaris/bin/activate
_SHIM="$HOME/.local/polaris/cuda_shim13"
_CU13=/soft/compilers/cudatoolkit/cuda-13.0.1
_CU12=/soft/compilers/cudatoolkit/cuda-12.9.1
export LD_LIBRARY_PATH="$_SHIM/mpishim:$_CU13/lib64:$_CU13/extras/CUPTI/lib64:$_CU12/lib64:$_CU12/extras/CUPTI/lib64:/opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3/lib:/opt/cray/libfabric/2.3.1/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="$_CU12" CUDA_PATH="$_CU12"
export PYTHONPATH="$_SHIM:$WS:$REPO" VJEPA_REPO="$REPO"
export TMPDIR=/tmp OMP_NUM_THREADS=8 PYTHONFAULTHANDLER=1
# Stale ~/.local can shadow the venv's torch; this bit us before.
export PYTHONNOUSERSITE=1
PY=python
cd "$WS" || exit 2

echo "JOB START $(date -u) host=$(hostname) NAME=$NAME FAMILY=$FAMILY TAG=$TAG"
echo "  base=$BASE aug=$AUGC win=$TRAIN_WIN"
for f in "$CFG" "$ECFG" "$TRAIN_WIN"; do [ -f "$f" ] || { echo "FATAL: missing $f"; exit 2; }; done
grep -q '"status": "completed"' "$BASE/val/rank_0000/manifest.json" 2>/dev/null \
  || { echo "FATAL: baseline val cache incomplete at $BASE"; exit 2; }
$PY -c "import torch;print('[env] torch',torch.__version__,'cuda',torch.cuda.is_available(),'ndev',torch.cuda.device_count())" || exit 2

# ---------------------------------------------------------------- 1. EXPORT
# Shards windows[r::4] across 4 GPUs. The trainer globs rank_*/ so shard count
# is free -- but a MISSING shard is not, and each rank writes its own manifest
# marked completed, so per-rank status is NOT sufficient. The gate below sums
# num_samples across every shard and demands the full window count.
if $PY -c "
import glob,json,sys,torch
ms=sorted(glob.glob('$AUGC/train/rank_*/manifest.json'))
if not ms: sys.exit(1)
if not all(json.load(open(m)).get('status')=='completed' for m in ms): sys.exit(1)
n=sum(json.load(open(m))['num_samples'] for m in ms)
tot=len(torch.load('$TRAIN_WIN',map_location='cpu',weights_only=False)['windows'])
sys.exit(0 if n==tot else 1)" 2>/dev/null; then
  echo "[export] augmented train cache already complete -> skip"
else
  echo "[export] augmented TRAIN cache -> $AUGC/train  $(date -u)"
  rm -rf "$AUGC/train"
  for r in 0 1 2 3; do
    RANK=$r WORLD_SIZE=4 LOCAL_RANK=$r CUDA_VISIBLE_DEVICES=$r PMI_RANK=$r PMI_SIZE=4 \
      $PY "$WS/export_esad_cache.py" --config "$ECFG" \
        --windows "$TRAIN_WIN" --esad-root "$ESAD_ROOT" \
        --output-dir "$AUGC/train" --repo-root "$REPO" --batch-size 2 \
        --augment --aug-seed 0 > "$AUGC.export.r$r.log" 2>&1 &
  done
  wait
  $PY -c "
import glob,json,sys,torch
ms=sorted(glob.glob('$AUGC/train/rank_*/manifest.json'))
bad=[m for m in ms if json.load(open(m)).get('status')!='completed']
n=sum(json.load(open(m))['num_samples'] for m in ms)
tot=len(torch.load('$TRAIN_WIN',map_location='cpu',weights_only=False)['windows'])
print(f'[export] gate shards={len(ms)} exported={n} expected={tot} incomplete={bad}')
sys.exit(0 if (len(ms)==4 and not bad and n==tot) else 3)" || {
    echo "FATAL: augmented export sharded/short -- see gate line"; tail -15 "$AUGC.export.r0.log"; exit 3; }
  echo "[export] done $(date -u)"
fi

# ------------------------------------------------- 2. TRAIN x3, IN PARALLEL
# One seed per GPU. Head training is single-process and cache-fed, so three
# seeds on three GPUs of the same node do not contend for anything but PCIe.
train_seed() {
  local SEED=$1 GPU=$2
  local OUT=$CR/runs/esad_double_${NAME}_${TAG}_s${SEED}
  # best.pt is written from epoch 0, so it is NOT a completion signal -- gating
  # on it is how a killed 17/20-epoch run gets published as finished. Gate on
  # the CSV epoch count; the trainer resumes from latest.pt.
  if [ -f "$OUT/log_r0.csv" ] && [ "$(( $(wc -l < "$OUT/log_r0.csv") - 1 ))" -ge 20 ]; then
    echo "[train] skip s$SEED: already 20 epochs"; return 0
  fi
  if grep -q "early stop at epoch" "$OUT/stdout.log" 2>/dev/null; then
    echo "[train] skip s$SEED: converged (early stop)"; return 0
  fi
  mkdir -p "$OUT"
  echo "[train] $TAG seed=$SEED gpu=$GPU $(date -u)"
  if [ "$FAMILY" = "vjepa" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY "$WS/train_esad_double_head.py" --config "$CFG" \
      --train-cache "$AUGC/train" --val-cache "$BASE/val" --test-cache "$BASE/test" \
      --train-presence-override "$REP/esad_windows_train_m1_presrep.pt" \
      --val-presence-override "$REP/esad_windows_val_m1_presrep.pt" \
      --output-dir "$OUT" --seed "$SEED" > "$OUT/stdout.log" 2>&1
    # Prove the override engaged. Without this line the run trained on UNION
    # labels under a presrep name: a plausible answer to the wrong question.
    if grep -q "presence override ACTIVE" "$OUT/stdout.log" 2>/dev/null; then
      rm -f "$OUT/OVERRIDE_UNCONFIRMED"
    else
      # A warn here cannot stop anything -- this runs backgrounded and `wait`
      # discards the return code. Leave a sentinel the score pass honours, so
      # a union-label run cannot be published under a presrep name.
      echo "[train] FATAL s$SEED: 'presence override ACTIVE' not in stdout.log"
      date -u > "$OUT/OVERRIDE_UNCONFIRMED"
    fi
  else
    CUDA_VISIBLE_DEVICES=$GPU $PY "$WS/train_esad_double_head.py" --config "$CFG" \
      --train-cache "$AUGC/train" --val-cache "$BASE/val" --test-cache "$BASE/test" \
      --output-dir "$OUT" --seed "$SEED" > "$OUT/stdout.log" 2>&1
  fi
}
for S in 0 1 2; do train_seed "$S" "$S" & done
wait
for S in 0 1 2; do
  echo "--- s$S tail ---"; tail -3 "$CR/runs/esad_double_${NAME}_${TAG}_s${S}/stdout.log" 2>/dev/null
done

# ------------------------------------------------- 3. SCORE x3, IN PARALLEL
echo "=== SCORE PASS $(date -u) ==="
score_seed() {
  local SEED=$1 GPU=$2
  local OUT=$CR/runs/esad_double_${NAME}_${TAG}_s${SEED}
  [ -f "$OUT/best.pt" ] || { echo "[score] skip s$SEED: no best.pt"; return 0; }
  [ -f "$OUT/OVERRIDE_UNCONFIRMED" ] && { echo "[score] REFUSE s$SEED: presence override unconfirmed"; return 0; }
  local NEP; NEP=$([ -f "$OUT/log_r0.csv" ] && echo $(( $(wc -l < "$OUT/log_r0.csv") - 1 )) || echo 0)
  # Early stopping is COMPLETION, not truncation. The trainer prints
  # "[TRAIN] early stop at epoch N" and breaks; a PBS kill prints nothing.
  # Gating on epoch count alone silently dropped every converged seed that
  # stopped before 19 -- historically ~38% of frozen runs.
  # Second witness, from the CSV alone: stdout.log can be absent even for a
  # complete run (prod37m s2 had all 20 epochs in log_r0.csv and no log at
  # all), and grep-on-a-missing-file fails CLOSED -- safe, but it discards a
  # finished seed. The trainer stops when (last_epoch - best_epoch) >= patience,
  # so that condition is itself proof it stopped on its own rather than by kill.
  # Find best_epoch BY HEADER NAME, never by position. The two ESAD trainers
  # write different columns -- frozen has `lr` and ends on best_epoch; the FT
  # trainer (train_esad_unfreeze.py) has no `lr` and ends on `secs`. Both are
  # 11 fields wide and both land in a file called log_r0.csv, so $NF silently
  # reads seconds on an FT run. That mistake has already been made once here
  # (it read best_epoch as a timing column and concluded head-training was
  # free). This script only scores frozen arms today, but the failure is silent
  # and the fix is one awk clause.
  local CONVERGED=no
  if [ -f "$OUT/log_r0.csv" ]; then
    CONVERGED=$(awk -F, '
      NR==1 { for (i=1;i<=NF;i++) if ($i=="best_epoch") bi=i; next }
      !bi   { next }
            { le=$1; be=$bi; n++ }
      END   { if (n && bi && (le-be)>=6) print "yes"; else print "no" }' "$OUT/log_r0.csv")
  fi
  if [ "$NEP" -lt 19 ] && [ "$CONVERGED" = no ] \
     && ! grep -q "early stop at epoch" "$OUT/stdout.log" 2>/dev/null; then
    echo "[score] skip s$SEED: only $NEP epochs, no early-stop line, not converged (killed?)"; return 0
  fi
  if [ "$FAMILY" = "vjepa" ]; then
    local RJ=$OUT/test_detection_ap_cov_fulldenom.json
    [ -f "$RJ" ] || CUDA_VISIBLE_DEVICES=$GPU $PY "$WS/score_esad_detection_ap.py" --config "$CFG" \
        --best-ckpt "$OUT/best.pt" \
        --test-cache "$CR/cache_${NAME}_cov_p0/test" --test-windows "$MAN/esad_windows_test_cov_p0.pt" \
        --test-cache "$CR/cache_${NAME}_cov_p1/test" --test-windows "$MAN/esad_windows_test_cov_p1.pt" \
        --full-gt-label-dir "$LABEL_DIR" --output "$RJ" > "$OUT/score.log" 2>&1
    [ -f "$RJ" ] && $PY -c "
import json
b=json.load(open('$RJ'))['combined_not_isolated']
# Four population nodes live in this file and differ by ~0.08 AP. source0_only
# scores 3467 frames and reads ~0.10 for a checkpoint published at ~0.18.
assert b['num_frames_scored']==5903 and b['gt_instances_full']==11207, ('WRONG POPULATION',b['num_frames_scored'])
v=b['variants_full_denominator']
print('[$NAME $TAG s$SEED] maxpick',round(v['maxpick']['ap_mean'],4),
      'wbf',round(v['wbf_meanconf']['ap_mean'],4),'frames',b['num_frames_scored'])"
  else
    local RJ=$OUT/test_detection_ap_masked_fulldenom.json
    [ -f "$RJ" ] || CUDA_VISIBLE_DEVICES=$GPU $PY "$WS/score_esad_detection_ap.py" --config "$CFG" \
        --best-ckpt "$OUT/best.pt" --test-cache "$BASE/test" \
        --test-windows "$MAN/esad_windows_test_cov_tubelet1_m1.pt" \
        --full-gt-label-dir "$LABEL_DIR" \
        --restrict-frame-stems "$MAN/vjepa_covered_frame_stems.pt" \
        --output "$RJ" > "$OUT/score.log" 2>&1
    local UJ=$OUT/test_detection_ap_unmasked_fulldenom.json
    [ -f "$UJ" ] || CUDA_VISIBLE_DEVICES=$GPU $PY "$WS/score_esad_detection_ap.py" --config "$CFG" \
        --best-ckpt "$OUT/best.pt" --test-cache "$BASE/test" \
        --test-windows "$MAN/esad_windows_test_cov_tubelet1_m1.pt" \
        --full-gt-label-dir "$LABEL_DIR" --output "$UJ" > "$OUT/score_unmasked.log" 2>&1
    [ -f "$RJ" ] && $PY -c "
import json
b=json.load(open('$RJ'))['single_source']; v=b['variants_full_denominator']
print('[$NAME $TAG s$SEED] MASKED maxpick',round(v['maxpick']['ap_mean'],4),
      'wbf',round(v['wbf_meanconf']['ap_mean'],4),'frames',b['num_frames_scored'])"
  fi
}
for S in 0 1 2; do score_seed "$S" "$S" & done
wait
echo "JOB END $(date -u)"
