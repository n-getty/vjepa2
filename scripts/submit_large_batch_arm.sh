#!/usr/bin/env bash
# Submit one LARGE-BATCH arm (lbA / lbB) through the proven capacity launcher.
#
# The arms exist to answer ONE question before any prod time is spent: does this
# recipe survive the global batch that 256 nodes forces on it? See
# scripts/gen_large_batch_configs.py for the derivation, and the plan file for
# the gate. VJEPA_TRUE_ACCUM=16 makes 16 nodes produce a gradient-exact 16x
# batch (train.py:1179-1198), so gb=6144 is reproduced on the capacity queue.
#
# This wrapper exists because the arm env must be set CONSISTENTLY -- config,
# checkpoint dir, accum and jobtag all have to agree, and a partial set would
# quietly train the wrong recipe into an arm's directory.
#
#   ./scripts/submit_large_batch_arm.sh lbA
#   ./scripts/submit_large_batch_arm.sh lbB
#
# Capacity allows 5 queued/running and 2 running per user -- check `qstat -u $USER`
# before submitting both.

set -o pipefail
ARM="${1:-}"
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2

case "$ARM" in
  lbA|lbB) ;;
  *) echo "usage: $0 {lbA|lbB}"; exit 1 ;;
esac

CFG="$ROOT/configs/vitg16_surg_vid_webdataset_single4/vitG384_${ARM}.yaml"
if [[ ! -f "$CFG" ]]; then
  echo "FATAL: $CFG missing. Run: python scripts/gen_large_batch_configs.py"
  exit 1
fi

# Must match VJEPA_TRUE_ACCUM to the multiplier the config was generated for --
# the EMA/warmup/lambda constants in the YAML are only correct at that batch.
ACCUM=16
CKPT_DIR="/flare/ModCon/ngetty/checkpoints/surg_2_1_vitG384_${ARM}/vitG384_n16g12_${ARM}"

echo "arm      : $ARM"
echo "config   : $CFG"
echo "ckpt dir : $CKPT_DIR"
echo "accum    : $ACCUM  (global batch 384 x $ACCUM = $((384 * ACCUM)))"

mkdir -p "$CKPT_DIR"
qsub -N "$ARM" \
     -v "VJEPA_CFG_NAME=vitG384_${ARM},VJEPA_CKPT_DIR=${CKPT_DIR},VJEPA_TRUE_ACCUM=${ACCUM},VJEPA_JOBTAG=${ARM}" \
     "$ROOT/scripts/vitG384_capacity_v2.sh"
