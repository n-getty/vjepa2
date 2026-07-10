#!/bin/bash
# run.sh for the calibration hold job: run all calib_*.yaml cells SEQUENTIALLY on the held node
# (12 tiles each), OOM-tolerant, then print the read_calib summary. Executed by the hold loop's
# `bash run.sh` on the head node — env (module load, CCL_*) is already set by the hold job preamble.
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
CTRL=/flare/ModCon/ngetty/experiments/_calib_ctrl
CFG_GLOB="$ROOT/configs/scaling/calib/calib_*.yaml"
TILES=12
cd "$ROOT"
HOST=$(head -1 "$CTRL/nodefile.full")
echo "CALIB(hold) START $(date) on $HOST"
i=0
for cfg in $CFG_GLOB; do
  slug=$(basename "$cfg" .yaml)
  port=$((29760 + i)); i=$((i+1))
  echo "===== [$slug] $(date) ====="
  (
    export PBS_NODEFILE="$CTRL/nodefile.full"
    export MASTER_ADDR="$HOST"; export MASTER_PORT=$port; export WORLD_SIZE=$TILES
    timeout 360 mpiexec --pmi=pmix -n "$TILES" -ppn "$TILES" \
        --hostfile "$CTRL/nodefile.full" --cpu-bind depth --depth 16 \
        python -m app.main_dist_aurora --train_mode \
            --fname "$cfg" --params_path "$cfg"
  ) > "$CTRL/calib_${slug}.log" 2>&1
  rc=$?
  fldr=$(python3 -c "import yaml;print(yaml.safe_load(open('$cfg'))['folder'])" 2>/dev/null)
  if [ -f "$fldr/log_r0.csv" ]; then
    n=$(($(wc -l < "$fldr/log_r0.csv") - 1))
    echo "[$slug] rc=$rc iters_logged=$n"
  else
    echo "[$slug] rc=$rc NO CSV (OOM/fail); tail:"; tail -3 "$CTRL/calib_${slug}.log"
  fi
done
echo "CALIB(hold) DONE $(date)"
echo "=== summary ==="
python3 -m scaling.read_calib --exp-root /flare/ModCon/ngetty/experiments/scaling_calib || true
