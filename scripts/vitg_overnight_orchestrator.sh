#!/usr/bin/env bash
# Overnight autonomous orchestrator for the ViT-g next-stage decision.
#
# Waits for the e19 probe result, then per the decision rule launches EITHER a
# resume (16f) chain OR a cooldown (64f) chain from the e19 checkpoint. Idempotent
# and safe to call repeatedly (from cron): it does nothing until e19 result is in,
# does nothing once a next-stage chain is already launched/running.
#
# DECISION RULE:
#   e19_f1 vs e14_f1 (74.23):
#     e19 >= 74.43 (clear +0.2 continued trend) -> RESUME (16f, ride the trend)
#     e19 <  74.43 (plateaued/declined)         -> COOLDOWN (64f, temporal lever)
#
# The cooldown branch is gated by a 1-node MEMORY SMOKE first (64f/bs1 is only an
# estimate ~84% tile); if the smoke OOMs, it falls back to 32f/bs2 (also ~84% but
# verified-shape-safe) — NOT to lower resolution (resolution stays 384).
set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
E19_LOG=/flare/ModCon/ngetty/logs/8572924.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov.OU
STATE=/flare/ModCon/ngetty/logs/vitg_overnight.state
E14_F1=74.23
THRESH=74.43   # e14 + 0.2
ts(){ date -u +%FT%TZ; }

# --- 0. already decided/launched? then ensure the chosen chain stays alive (re-arm) ---
if [[ -f "$STATE" ]]; then
  DECISION=$(cat "$STATE")
  case "$DECISION" in
    RESUME)        NAME=vitg_rs; CHAIN=$ROOT/scripts/vitg384_resume_16f_chain.sh ;;
    COOLDOWN_64F)  NAME=vitg_cd; CHAIN=$ROOT/scripts/vitg384_cooldown_64f_chain.sh ;;
    COOLDOWN_32F)  NAME=vitg_cd; CHAIN=$ROOT/scripts/vitg384_cooldown_32f_chain.sh ;;
    COOLDOWN)      echo "$(ts) orchestrator: COOLDOWN pending smoke gate, re-evaluating"; rm -f "$STATE"; exec "$0" ;;
    *) echo "$(ts) orchestrator: transient state '$DECISION', re-evaluating"; rm -f "$STATE"; exec "$0" ;;
  esac
  ALIVE=$(qstat -u "$USER" 2>/dev/null | awk -v n="$NAME" '$4==n && ($10=="R"||$10=="Q")' | wc -l)
  if (( ALIVE >= 1 )); then echo "$(ts) orchestrator: $DECISION chain alive ($ALIVE) — ok"; exit 0; fi
  # keep the chosen chain alive (watchdog role)
  for a in 1 2 3; do OUT=$(qsub "$CHAIN" 2>&1) && { echo "$(ts) orchestrator: re-armed $DECISION -> $OUT"; exit 0; }; sleep 20; done
  echo "$(ts) orchestrator: re-arm $DECISION qsub failed: $OUT"; exit 0
fi

# --- 1. e19 result in yet? ---
if [[ ! -f "$E19_LOG" ]]; then echo "$(ts) orchestrator: e19 log not present yet"; exit 0; fi
# best val_macro_f1 across all saves (early-stop prints best; take max of Saved-best lines)
E19_F1=$(grep -oE "val_macro_f1=[0-9.]+" "$E19_LOG" 2>/dev/null | sed 's/val_macro_f1=//' | sort -g | tail -1)
DONE=$(grep -cE "\[e19\] DONE|Early stop" "$E19_LOG" 2>/dev/null)
if [[ -z "$E19_F1" ]]; then echo "$(ts) orchestrator: e19 probe running, no F1 yet"; exit 0; fi
if (( DONE == 0 )); then echo "$(ts) orchestrator: e19 probe in progress, best-so-far=$E19_F1 (waiting for convergence)"; exit 0; fi

# --- 2. decide ---
CMP=$($PY -c "print('RESUME' if float('$E19_F1') >= $THRESH else 'COOLDOWN')")
echo "$(ts) orchestrator: e19=$E19_F1 vs e14=$E14_F1 (thresh $THRESH) -> $CMP"

# --- 3. launch chosen branch ---
if [[ "$CMP" == "RESUME" ]]; then
  echo "RESUME" > "$STATE"
  for a in 1 2 3; do OUT=$(qsub "$ROOT/scripts/vitg384_resume_16f_chain.sh" 2>&1) && { echo "$(ts) orchestrator: LAUNCHED RESUME -> $OUT"; exit 0; }; sleep 20; done
  echo "$(ts) orchestrator: RESUME qsub failed: $OUT"; exit 0
else
  # cooldown: pick 64f if the memory smoke PASSED, else 32f fallback (both @384, never lower res)
  GATE=/flare/ModCon/ngetty/logs/vitg_cd_smoke.done
  if [[ ! -f "$GATE" ]]; then
    echo "$(ts) orchestrator: COOLDOWN chosen but 64f mem smoke not done yet — waiting (will retry)."
    rm -f "$STATE"; exit 0   # re-evaluate next poll once smoke gate appears
  fi
  if grep -q "SMOKE_PASS" "$GATE"; then
    CHAIN=$ROOT/scripts/vitg384_cooldown_64f_chain.sh; KIND=COOLDOWN_64F
  else
    CHAIN=$ROOT/scripts/vitg384_cooldown_32f_chain.sh; KIND=COOLDOWN_32F
  fi
  echo "$KIND" > "$STATE"
  for a in 1 2 3; do OUT=$(qsub "$CHAIN" 2>&1) && { echo "$(ts) orchestrator: LAUNCHED $KIND -> $OUT"; exit 0; }; sleep 20; done
  echo "$(ts) orchestrator: $KIND qsub failed: $OUT"; exit 0
fi
