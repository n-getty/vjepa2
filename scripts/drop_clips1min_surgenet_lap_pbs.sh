#!/usr/bin/env bash
# Drop the clips1min clips from surgenet_lap. Leonardo's clips_1min subset was a
# TEST set, not real training data; it was merged into surgenet_lap at ingest and
# is 1859/5843 (31.8%) of the source. Their source-video IDs all start with
# "clips1min_" (real clips are procedure-named: appendectomy_/cholestectomy_/...),
# so a source-ID substring match cleanly separates them.
#
# Reuses the tested scripts/drop_sources_reshard.py (same tool that scrubbed
# eval-leaked videos from surgenet_robotic).
#
# SAFETY: the non-resharded source for surgenet_lap is GONE — the resharded dir
# is the ONLY copy. So: build drop-list from the current dir, drop -> temp dir,
# verify (kept == total-clips1min AND shard_count>=192), then atomic-swap:
#   surgenet_lap -> surgenet_lap_bak_clips1min, temp -> surgenet_lap.
# Backup kept; delete after a training run confirms the new shards load.
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/drop_clips1min_surgenet_lap_pbs.sh
#
#PBS -N vjepa_drop_c1m
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
DATA=/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded
DS=surgenet_lap
IN="$DATA/$DS"
TMP="$DATA/${DS}_dropc1m_tmp"
BAK="$DATA/${DS}_bak_clips1min"
DROP_JSON="$DATA/${DS}.clips1min_drop.json"
PYTHON=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python

mkdir -p /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-}"

# --- Build the drop-list: every source-video id starting with 'clips1min'. ---
"$PYTHON" - "$IN" "$DROP_JSON" <<'PY'
import glob, json, os, sys, tarfile
sys.path.insert(0, "scripts")
import reshard_webdataset as rw
in_dir, out_json = sys.argv[1], sys.argv[2]
drop = {}
for t in sorted(glob.glob(os.path.join(in_dir, "*.tar"))):
    with tarfile.open(t, "r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            km = rw.SAMPLE_KEY_RE.match(m.name)
            if not km:
                continue
            sm = rw.SOURCE_VIDEO_RE.match(km.group("key"))
            if sm and sm.group("source").startswith("clips1min"):
                drop[sm.group("source")] = 1
json.dump(drop, open(out_json, "w"))
print(f"drop-list: {len(drop)} clips1min source videos -> {out_json}")
PY

NDROP_SRC=$("$PYTHON" -c "import json;print(len(json.load(open('$DROP_JSON'))))")
if (( NDROP_SRC == 0 )); then
  echo "ERROR: drop-list empty — refusing to run (source parse changed?)." >&2
  exit 1
fi

# --- Drop -> temp (source dir read-only; nothing destroyed). ---
rm -rf "$TMP"
echo "dropping clips1min ($NDROP_SRC sources) from $IN -> $TMP"
"$PYTHON" "$ROOT/scripts/drop_sources_reshard.py" \
    --input "$IN" --output "$TMP" \
    --drop-json "$DROP_JSON" --target-shards 256 --seed 0

# --- Verify + atomic swap. ---
read -r old_n new_n new_sh drop_clips < <("$PYTHON" - "$IN/metadata.json" "$TMP/metadata.json" "$TMP/reshard_summary.json" <<'PY'
import json, sys
old = json.load(open(sys.argv[1])); new = json.load(open(sys.argv[2])); summ = json.load(open(sys.argv[3]))
print(old["sample_count"], new["sample_count"], new["shard_count"], summ.get("dropped_clips", -1))
PY
)
echo "verify: old=$old_n new=$new_n dropped=$drop_clips new_shards=$new_sh"
# Expected: dropped == 1859, new == old - 1859, shards >= 192.
if (( new_n + drop_clips != old_n )); then
  echo "FAIL: kept($new_n)+dropped($drop_clips) != old($old_n) — NOT swapping, source kept." >&2
  exit 1
fi
if (( drop_clips == 0 )); then
  echo "FAIL: dropped 0 clips — NOT swapping." >&2
  exit 1
fi
if (( new_sh < 192 )); then
  echo "FAIL: new shard_count $new_sh < 192 staging floor — NOT swapping." >&2
  exit 1
fi

rm -rf "$BAK"
mv "$IN" "$BAK"
mv "$TMP" "$IN"
echo "[OK] $DS: $old_n -> $new_n samples (dropped $drop_clips clips1min), shards -> $new_sh. backup: $BAK"
echo "JOB END: $(date)"
