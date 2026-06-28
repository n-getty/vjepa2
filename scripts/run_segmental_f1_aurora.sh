#!/usr/bin/env bash
# Compute F1@{10,25,50} + edit-score for a trained ASFormer probe on SAR-RARP50
# val, on Aurora (single tile, XPU, inference only). Lets us compare against
# external SOTA F1@10=84.10 (our macro-F1 alone can't).
#
# Usage: qsub -v TAG=e19 scripts/run_segmental_f1_aurora.sh
#   TAG -> uses configs/heads/sarrarp50/full_cached_384/${TAG}_probe.yaml for the
#   encoder + data spec, and the matching probe run's best.pt for the head.
#
#PBS -N vitg_segf1
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=1
#PBS -l walltime=00:60:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
TAG="${TAG:?must qsub with -v TAG=...}"
YAML=$ROOT/configs/heads/sarrarp50/full_cached_384/${TAG}_probe.yaml
# best.pt: probe folder is the yaml's `folder` + video_classification_frozen/<tag>
FOLDER=$(grep -E "^folder:" "$YAML" | awk '{print $2}')
PROBE_TAG=$(grep -E "^tag:" "$YAML" | awk '{print $2}')
BEST=$FOLDER/video_classification_frozen/$PROBE_TAG/best.pt
OUT=/flare/ModCon/ngetty/logs/segf1_${TAG}.json

cd $ROOT && module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TMPDIR=/tmp OMP_NUM_THREADS=16
# single tile is enough (inference only); pin to tile 0
export ZE_AFFINITY_MASK=0.0
export http_proxy="http://proxy.alcf.anl.gov:3128" https_proxy="http://proxy.alcf.anl.gov:3128"

PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
echo "[segf1 $TAG] yaml=$YAML best=$BEST out=$OUT $(date)"
[ -f "$BEST" ] || { echo "[segf1 $TAG] best.pt NOT FOUND: $BEST"; exit 1; }
$PY $ROOT/scripts/eval_segmental_f1.py --yaml "$YAML" --checkpoint "$BEST" --out "$OUT"
echo "[segf1 $TAG] DONE $(date)"
