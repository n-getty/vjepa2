#!/usr/bin/env bash
# Pack the 4 surgical IMAGE sets into image-WebDataset tars, then reshard each
# into a final <name>_img/ source dir (metadata.json included) for the vjepa_2_1
# image branch. CPU-only (Pillow decode/re-encode). One dataset per job:
#   qsub -v DATASET=hyperkvasir scripts/pack_images_pbs.sh
#   qsub -v DATASET=dsad        scripts/pack_images_pbs.sh
#   qsub -v DATASET=esad        scripts/pack_images_pbs.sh
#   qsub -v DATASET=psi_ava     scripts/pack_images_pbs.sh
# (CholecSeg8k intentionally excluded — Cholec80-frame redundancy; masks-only new
#  signal the image branch never consumes.)
#
#PBS -N vjepa_pack_img
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1:ncpus=104
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail   # NOT -u: `module load frameworks` references unbound vars
module load frameworks 2>/dev/null || module load frameworks

DATASET="${DATASET:?set -v DATASET=hyperkvasir|dsad|esad|psi_ava}"
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
RESHARD_ROOT=/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded
INCOMING=/flare/ModCon/ngetty/data/incoming_robotic
PYTHON="$(command -v python3)"
echo "using python: $PYTHON"
"$PYTHON" -c "import PIL; print('Pillow', PIL.__version__)"

mkdir -p /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-} DATASET=$DATASET"

STAGING="$RESHARD_ROOT/${DATASET}_img_staging"
FINAL="$RESHARD_ROOT/${DATASET}_img"
mkdir -p "$STAGING"

# Per-dataset archive selection (see pack_images_to_wds.py for the keep-filters).
case "$DATASET" in
  hyperkvasir)
    ARCHIVES=( "$INCOMING/hyperkvasir/train.zip" "$INCOMING/hyperkvasir/valid.zip" "$INCOMING/hyperkvasir/test.zip" )
    ;;
  dsad)
    ARCHIVES=( "$INCOMING/dsad/DSAD.zip" )
    ;;
  esad)
    # ESAD train + val (test_images too — all are real RARP frames).
    ARCHIVES=( "$INCOMING/esad/train.zip" "$INCOMING/esad/val.zip" "$INCOMING/esad/test_images.zip" )
    ;;
  psi_ava)
    ARCHIVES=( "$INCOMING/psi_ava/PSI-AVA.tar.gz" )
    ;;
  *) echo "unknown DATASET=$DATASET" >&2; exit 2 ;;
esac

echo "packing $DATASET from ${#ARCHIVES[@]} archive(s) -> $STAGING"
"$PYTHON" "$ROOT/scripts/pack_images_to_wds.py" \
    --dataset "$DATASET" --archives "${ARCHIVES[@]}" \
    --output-staging "$STAGING" --force

# Count packed images (member = <key>.image.jpg) to size the reshard.
NIMG=$("$PYTHON" -c "import glob,tarfile
tot=0
for t in glob.glob('$STAGING/*.tar'):
    with tarfile.open(t) as tf:
        tot+=sum(1 for n in tf.getnames() if n.endswith('.image.jpg'))
print(tot)")
echo "packed images: $NIMG"
# ~512 images/shard density; floor 8 shards (image sets are small).
TARGET=$(( NIMG / 512 )); (( TARGET < 8 )) && TARGET=8
echo "resharding $STAGING -> $FINAL (target shards: $TARGET)"
"$PYTHON" "$ROOT/scripts/reshard_webdataset.py" \
    --input "$STAGING" --output "$FINAL" \
    --prefix "${DATASET}_img" --target-shards "$TARGET" --seed 0 --force

echo "JOB END: $(date)"
echo "=== final metadata ==="
"$PYTHON" -c "import json;d=json.load(open('$FINAL/metadata.json'));print({k:d[k] for k in ['name','shard_count','sample_count']})"
