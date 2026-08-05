#!/usr/bin/env bash
# Minimal 1-tile Aurora job to run the flash-SDPA correctness gate
# (scripts/gate_flash_sdpa_xpu.py) as a FRESH current artifact. A/Bs the native
# SYCL-TLA fused flash kernel vs a math reference at the ViT-gigantic encoder
# shape. PASS (exit 0) = bf16 cos>0.999 & max|Δ|<0.03; FAIL = do not trust flash.
#
#   qsub -A ModCon -q debug -l select=1 scripts/run_flash_gate_aurora.sh
#PBS -N flash_gate
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:15:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
cd "$ROOT"
module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
# Pin to a single tile; force the flag ON (the script also sets it, belt+braces).
export ZE_AFFINITY_MASK=0
export VJEPA_USE_XPU_FLASH=1
echo "=== flash gate @ $(hostname) ==="
python scripts/gate_flash_sdpa_xpu.py
echo "=== gate exit=$? ==="
