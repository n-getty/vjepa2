#!/usr/bin/env bash
# PBS launcher for the surgical WebDataset reshard + metadata generation.
# Submit on Aurora with:
#   qsub -A ModCon -q debug -l select=2 -l walltime=01:00:00 \
#        -l filesystems=home:flare scripts/reshard_pbs_aurora.sh
#
# Step 5 of the V-JEPA 2.1 surgical Aurora plan (16-node pretraining).
#
#PBS -N vjepa_reshard
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/vjepa_reshard.${PBS_JOBID}.log

set -euo pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
INPUT_ROOT=/flare/ModCon/ngetty/data/surg_vid_webdataset
OUTPUT_ROOT=/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded
PYTHON=/usr/bin/python3.10

mkdir -p "$OUTPUT_ROOT"
mkdir -p /flare/ModCon/ngetty/logs

cd "$ROOT"

echo "=== reshard: under-sharded surgical datasets ==="
INPUT_ROOT="$INPUT_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" PYTHON="$PYTHON" \
  TARGET_SHARDS=32 PARALLEL=8 INCLUDE_SITL=0 \
  bash "$ROOT/scripts/reshard_all_aurora.sh"

echo
echo "=== metadata: un-resharded datasets ==="
for ds in sitl surgtoolloc2022 surgvu24; do
  dir="$INPUT_ROOT/$ds"
  if [[ ! -d "$dir" ]]; then
    echo "  skip $ds (missing)" >&2
    continue
  fi
  if [[ -f "$dir/metadata.json" ]]; then
    echo "  $ds: metadata.json already present, skipping"
    continue
  fi
  echo "  $ds: generating metadata.json"
  "$PYTHON" "$ROOT/scripts/generate_wds_metadata.py" --input "$dir"
done

echo
echo "=== summary ==="
for ds in crcd endovis15 jigsaw kinetics400 miccai_2017 miccai_endoseg sitl \
         surgenet_robotic surgtoolloc2022 surgvisdom surgvu24; do
  for root in "$OUTPUT_ROOT" "$INPUT_ROOT"; do
    if [[ -d "$root/$ds" ]]; then
      mf="$root/$ds/metadata.json"
      if [[ -f "$mf" ]]; then
        echo "  [OK]   $mf"
      else
        echo "  [MISS] $root/$ds (no metadata.json)"
      fi
      break
    fi
  done
done

echo "done"
