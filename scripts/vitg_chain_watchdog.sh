#!/usr/bin/env bash
# Watchdog: ensure exactly one vitg_chain job is running or queued, UNLESS the
# CPT run is already complete (epoch >= num_epochs). Decoupled from any slice so
# co-tenant Q-slot contention only DELAYS the next submit, never permanently
# breaks the chain. Intended to be polled (e.g. CronCreate every ~10min).
set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/surg_2_1_vitg384_cleandata/vitg384_n16g12_weak
SELF=$ROOT/scripts/vitg384_chain_debugscaling.sh
PARAMS=$CKPT_DIR/params-pretrain.yaml

# 1. Is the run already complete?
if [[ -f "$PARAMS" && -f "$CKPT_DIR/latest.pth.tar" ]]; then
  NE=$($PY -c "import yaml;print(yaml.safe_load(open('$PARAMS'))['optimization']['epochs'])" 2>/dev/null || echo 999)
  CE=$($PY -c "import torch;print(torch.load('$CKPT_DIR/latest.pth.tar',map_location='cpu',weights_only=False).get('epoch',0))" 2>/dev/null || echo 0)
  if [[ "$CE" =~ ^[0-9]+$ ]] && (( CE >= NE )); then
    echo "$(date -u +%FT%TZ) watchdog: complete (epoch $CE/$NE) — no action"
    exit 0
  fi
  echo "$(date -u +%FT%TZ) watchdog: progress epoch ${CE}/${NE}"
fi

# 2. Is a vitg_chain already R or Q? (col4=jobname, col10=state)
ALIVE=$(qstat -u "$USER" 2>/dev/null | awk '$4=="vitg_chain" && ($10=="R"||$10=="Q")' | wc -l)
if (( ALIVE >= 1 )); then
  echo "$(date -u +%FT%TZ) watchdog: $ALIVE vitg_chain alive — no action"
  exit 0
fi

# 3. None alive and not complete -> resubmit (retry a few times for transient slot contention)
for attempt in 1 2 3; do
  OUT=$(qsub "$SELF" 2>&1)
  if [[ $? -eq 0 ]]; then
    echo "$(date -u +%FT%TZ) watchdog: RESUBMITTED chain -> $OUT"
    exit 0
  fi
  echo "$(date -u +%FT%TZ) watchdog: qsub attempt $attempt failed: $OUT"
  sleep 20
done
echo "$(date -u +%FT%TZ) watchdog: all qsub attempts failed (likely co-tenant holds slot) — will retry next poll"
exit 0
