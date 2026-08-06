#!/bin/bash
#PBS -N verify_g16
#PBS -l select=1
#PBS -l walltime=00:40:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
#
# TWO OWED VERIFICATIONS, ONE NODE. Both are small, both are blocked only on a
# queue slot, and debug/debug-scaling are max_run=1 -- so they share one.
#
# PART 1 -- the g16 decode benchmark.
#   scripts/reencode_gop_pbs.sh ends by naming its own acceptance criteria:
#     "1. sample count matches the original metadata.json
#      2. re-run the offline decode benchmark and confirm the predicted drop"
#   surgvu24_clean_g16 has passed (1) -- 40356/40356 samples, 2000/2000 shards,
#   enc_fail=0 across 1024 workers -- and has NOT been through (2). Sample count
#   only proves nothing was lost; it says nothing about whether the re-encode
#   bought the decode time it was run for. The whole point of spending 32 nodes
#   was the predicted 3.38 s -> 0.41 s (8.2x) from the per-clip intervention arm
#   in docs/THROUGHPUT_RECIPE_AURORA.md. Until this runs, that 8.2x is a
#   prediction from ONE hand-picked clip, extrapolated to 40 k.
#
#   cholec80 is re-run as a CONTROL, not out of caution: it already has a
#   corpus-level number (7.1x). If this harness reproduces ~7x there, its
#   surgvu24 number is trustworthy; if it does not, the harness is the problem
#   and the surgvu24 number should be discarded rather than believed.
#
#   n=200, not the default 60. The quantity of interest is a p50 ratio and the
#   per-clip spread within a source is large; 60 draws puts a wide interval on
#   the very digit being reported.
#
# PART 2 -- does the landed TMPDIR fix reach the ranks (scripts/probe_tmpdir_pals.sh).
#   Runs FIRST because it is under a minute and gates whether nw=2 is usable at
#   all. See that script's header for why the fix's validation layer and its
#   application layer are not the same claim.
#
# Submit:  qsub -A AuroraGPT -q debug scripts/verify_g16_and_tmpdir_pbs.sh
#
# Neither part touches the training path, writes to a corpus, or needs >1 node.

set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
PY=/flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/python
cd "$ROOT" || exit 1

echo "########## PART 2 (first: it gates nw=2) -- TMPDIR through PALS ##########"
bash scripts/probe_tmpdir_pals.sh
echo
echo "########## PART 1 -- g16 decode benchmark ##########"
echo "Predicted from the per-clip intervention arm (THROUGHPUT_RECIPE_AURORA.md):"
echo "  surgvu24_clean  3.38 s -> 0.41 s  (8.2x)   <- UNDER TEST"
echo "  cholec80        3.04 s -> 0.35 s  (8.8x); corpus-level 7.1x already measured"
echo "                                              <- CONTROL for this harness"
echo

N=${N:-200}
for pair in "surgvu24_clean surgvu24_clean_g16" "cholec80 cholec80_g16"; do
  set -- $pair
  for s in "$1" "$2"; do
    echo "----- $s (n=$N) -----"
    # Serially, one source at a time: these are single-threaded decodes and the
    # measurement IS decode latency, so running them concurrently would have
    # them contend and inflate exactly the number being reported.
    "$PY" scripts/decode_per_source_profile.py --source "$s" --n "$N" 2>&1 | tail -12
    echo
  done
done

cat <<'EOF'
=== HOW TO READ THIS ===
Compare the p50 raw-decode line within each pair; that is the number the
re-encode was bought for. Judge on p50, not mean -- mean is dominated by the
tail, and the tail is explicitly NOT what a re-encode addresses (see
"…but keyframe spacing does not own the TAIL").

  cholec80 ratio ~7x   -> harness agrees with the known corpus-level result;
                          the surgvu24 ratio beside it can be trusted.
  cholec80 ratio off   -> the HARNESS is wrong, not the corpus. Discard the
                          surgvu24 number too; do not report either.
  surgvu24 ratio <<8x  -> a real shortfall. Note it against the prediction
                          rather than restating the prediction: the 8.2x came
                          from one clip, and one clip is not a corpus.

Also check `frac<1.0(DROP)` is unchanged across each pair. A re-encode that
altered the black-clip drop rate would have changed the corpus CONTENT, not
just its cost -- which is the one thing this intervention promised not to do.
EOF
