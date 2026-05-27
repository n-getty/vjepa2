#!/usr/bin/env bash
# Multi-node Polaris launcher for V-JEPA 2.1 phase sequences.
#
# Env vars:
#   VJEPA_NUM_NODES     (required, e.g. 1, 2, 4)
#   VJEPA_NUM_GPUS      per-node GPU count (default 4 for Polaris)
#   VJEPA_STRONG_SCALE  if "1", per-rank batch stays fixed; else global batch preserved
#   VJEPA_PYTHON        (required) path to python (inside the venv used inside PBS)
#   VJEPA_ACCOUNT       (required) PBS account
#   VJEPA_PARTITION     PBS queue (default prod)
#   VJEPA_TIME_MIN      walltime in minutes per phase (default 720)
#   VJEPA_FILESYSTEMS   PBS filesystems (default home:eagle)
#   VJEPA_STAGE_DATA    if "1", pass --stage_data (only valid for WebDataset dirs)
#   VJEPA_DRY_RUN       if "1", build PBS scripts but don't qsub
#
# Phases are chained via PBS afterok dependencies.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VJEPA_PYTHON:?must set VJEPA_PYTHON to your venv python}"
RUNTIME_CFG_TOOL="$ROOT/scripts/prepare_runtime_config.py"

NUM_NODES="${VJEPA_NUM_NODES:?must set VJEPA_NUM_NODES}"
NUM_GPUS="${VJEPA_NUM_GPUS:-4}"
STRONG_SCALE="${VJEPA_STRONG_SCALE:-0}"
ACCOUNT="${VJEPA_ACCOUNT:?must set VJEPA_ACCOUNT to your PBS allocation}"
PARTITION="${VJEPA_PARTITION:-prod}"
TIME_MIN="${VJEPA_TIME_MIN:-720}"
FILESYSTEMS="${VJEPA_FILESYSTEMS:-home:eagle}"
STAGE_DATA="${VJEPA_STAGE_DATA:-0}"
DRY_RUN="${VJEPA_DRY_RUN:-0}"
FOLDER_BASE="${VJEPA_FOLDER_BASE:-}"
DISABLE_AWS_OFI="${VJEPA_DISABLE_AWS_OFI:-0}"

case "$NUM_GPUS" in
  2|4|8) ;;
  *) echo "VJEPA_NUM_GPUS must be 2, 4, or 8 (got $NUM_GPUS)" >&2; exit 2 ;;
esac

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 <phase1.yaml> [phase2.yaml ...]" >&2
  exit 2
fi

PREP_ARGS=(--root "$ROOT" --num-gpus "$NUM_GPUS" --num-nodes "$NUM_NODES")
if [[ "$STRONG_SCALE" == "1" ]]; then
  PREP_ARGS+=(--strong-scale)
fi
if [[ -n "$FOLDER_BASE" ]]; then
  PREP_ARGS+=(--folder-base "$FOLDER_BASE")
fi

LAUNCH_ARGS=(
  -m app.main_dist_polaris
  --account "$ACCOUNT"
  --partition "$PARTITION"
  --time "$TIME_MIN"
  --filesystems "$FILESYSTEMS"
  --master_port 29500
)
if [[ "$STAGE_DATA" == "1" ]]; then
  LAUNCH_ARGS+=(--stage_data)
fi
if [[ "$DISABLE_AWS_OFI" == "1" ]]; then
  LAUNCH_ARGS+=(--disable_aws_ofi)
fi

# Phase status: read effective config, look for latest checkpoint, decide fresh/resume/complete/skip.
phase_status() {
  local cfg="$1"
  "$PYTHON_BIN" - "$cfg" <<'PY'
import os, sys, yaml, torch
cfg_path = sys.argv[1]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
folder = cfg["folder"]
num_epochs = int(cfg["optimization"]["epochs"])
for cand in ("latest.pth.tar", "latest.pt"):
    p = os.path.join(folder, cand)
    if os.path.exists(p):
        try:
            ck = torch.load(p, map_location="cpu")
            if int(ck.get("epoch", 0)) >= num_epochs:
                print("complete"); sys.exit(0)
        except Exception:
            pass
        print("resume"); sys.exit(0)
print("fresh")
PY
}

cd "$ROOT"

PREV_JOB=""
for cfg in "$@"; do
  echo
  echo "== Preparing $cfg [nodes=$NUM_NODES gpus/node=$NUM_GPUS strong=$STRONG_SCALE] =="
  runtime_cfg="$("$PYTHON_BIN" "$RUNTIME_CFG_TOOL" "${PREP_ARGS[@]}" "$cfg")"
  echo "  runtime cfg: $runtime_cfg"

  status="$(phase_status "$runtime_cfg")"
  case "$status" in
    complete) echo "  status: complete (skipping)"; continue ;;
    resume)   echo "  status: resume" ;;
    fresh)    echo "  status: fresh" ;;
    *) echo "  unknown status: $status" >&2; exit 1 ;;
  esac

  phase_launch_args=("${LAUNCH_ARGS[@]}" --fname "$runtime_cfg")
  if [[ -n "$PREV_JOB" ]]; then
    phase_launch_args+=(--depends_on "$PREV_JOB")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    phase_launch_args+=(--dry_run)
  fi

  job_out="$("$PYTHON_BIN" "${phase_launch_args[@]}" | tail -n1)"
  PREV_JOB="$job_out"
  echo "  -> $PREV_JOB"
done

echo
echo "Done. Phase jobs chained via PBS afterok dependencies."
echo "Track with: qstat -u \$USER"
