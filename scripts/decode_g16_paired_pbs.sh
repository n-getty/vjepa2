#!/bin/bash
#PBS -N g16_paired
#PBS -l select=1
#PBS -l walltime=00:50:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
#
# PAIRED decode benchmark, all four re-encoded sources.
#
# WHY THIS RE-RUNS WORK JOB 8740990 ALREADY DID. That job's arms were UNPAIRED:
# decode_per_source_profile.py used wds.WebDataset(resampled=True), and
# create_url_iterator() builds ResampledShardList(urls) with NO seed forwarded --
# ResampledShards.__iter__ then mixes time_ns(), getpid() and os.urandom(4) into
# its own. So a source and its _g16 twin streamed DIFFERENT VIDEOS, and every
# reported ratio was a difference of medians over two independent draws rather
# than a within-pair effect.
#
# That is not a hypothetical defect. It presented as a CORPUS-CONTENT change:
# surgvu24 came back frac<1.0 = 0.015 -> 0.000 and clip-std min 0.00 -> 28.72,
# which reads exactly like "the re-encode dropped the black clips" -- the one
# thing the intervention promised not to do -- and was in fact 3 different clips
# out of 200. Fixed in 520e0ce (resampled=False + np seed); a source and its twin
# hold the same samples in the same shard order, so in-order iteration makes the
# comparison paired. The fix also moved cholec80's ratio 2.30x -> ~3.0x, so the
# defect was UNDERSTATING the win, not inflating it.
#
# lemon and sitl_2026 have NEVER been benchmarked at all -- only count-verified.
#
# READ THE sitl_2026 ROW DIFFERENTLY FROM THE OTHER THREE. Its twin is
# _g16_512: reencode_gop_pbs.sh:106 sets SHORT=512 for that source alone, so it
# is GOP *and* a downscale. Its ratio is therefore not a GOP measurement and is
# not comparable to the other three. The resolution line in the output makes the
# confound visible; report it as "GOP+downscale" or not at all.
#
# Expect ~1.6-3x, NOT the 7-8x on record. Those figures are per-clip arms with a
# 16-frame scatter across the whole video; the trainer seeks ONCE into a
# 64-frame window (src/datasets/webdataset.py:375-383), so at most ~2 keyframe
# backfills are ever paid per sample instead of ~16. See the "At the corpus
# level the win is 1.6-3x" subsection in docs/THROUGHPUT_RECIPE_AURORA.md.
# A 2-3x corpus number here is the PREDICTED result, not a shortfall.
#
# Submit:  qsub -A AuroraGPT -q debug-scaling scripts/decode_g16_paired_pbs.sh
#
# Read-only: streams .tar shards, decodes in-process, writes nothing.

set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
PY=/flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/python
cd "$ROOT" || exit 1

N=${N:-200}
SEED=${SEED:-0}

echo "=== paired g16 decode benchmark  (n=$N, seed=$SEED, in-order/paired) ==="
echo "commit: $(git rev-parse --short HEAD 2>/dev/null)"
echo

for pair in "cholec80 cholec80_g16" \
            "surgvu24_clean surgvu24_clean_g16" \
            "lemon lemon_g16" \
            "sitl_2026 sitl_2026_g16_512"; do
  set -- $pair
  echo "##### PAIR: $1  vs  $2 #####"
  for s in "$1" "$2"; do
    echo "----- $s -----"
    # Serial, one source at a time. These are single-threaded decodes and the
    # measurement IS decode latency, so concurrent arms would contend and
    # inflate the exact number being reported.
    "$PY" scripts/decode_per_source_profile.py --source "$s" --n "$N" --seed "$SEED" 2>&1 | tail -12
    echo
  done
done

cat <<'EOF'
=== HOW TO READ THIS ===
Judge on the p50 raw-decode line within each pair. Mean is dominated by the
tail, and keyframe spacing is not what owns the tail
([[dataload-tail-is-keyframe-spacing]] is CORRECTED on exactly this point: GOP
owns the BODY of the distribution; the owner of the tail is still unknown).
Report the max line too -- in the unpaired run the tails improved MORE than the
medians (cholec80 max 2362 -> 790), and if that holds under pairing it is worth
saying, because the dataload cost that hurts at scale is a max-over-ranks order
statistic ([[scaling-loss-is-a-straggler-order-statistic]]).

  frames/video ~equal within a pair -> the pairing worked (same clips).
  frames/video differs             -> the arms are still not paired; the ratio
                                      is a difference of draws. Do not report.
  frac<1.0(DROP) differs           -> under pairing this would be a REAL content
                                      change, not the sampling artifact it was
                                      before. Stop and investigate before ingest.
  sitl_2026 row                    -> GOP + downscale, see header. Not a GOP
                                      number; check the resolution line.

A 2-3x here is the predicted corpus-level result. Do not restate the 7-8x
per-clip prediction as if this fell short of it.
EOF
