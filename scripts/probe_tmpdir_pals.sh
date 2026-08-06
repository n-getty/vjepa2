#!/bin/bash
#PBS -N tmpdirprobe
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -l filesystems=flare
#PBS -j oe
#
# DOES A JOB-SHELL `export TMPDIR=/tmp` ACTUALLY REACH THE RANKS?
#
# Job 8740830 proved the AF_UNIX overflow and proved that TMPDIR=/tmp fixes it
# -- but it set TMPDIR from INSIDE python (--short-tmpdir), one layer BELOW
# mpiexec. The fix that landed in scripts/lib/aurora_hsdp_env.sh sets it in the
# job shell, one layer ABOVE. Those are not the same claim, because PALS does
# not merely inherit TMPDIR, it REWRITES it: the job shell's 68-char
# /var/tmp/pbs.<jobid>.<server> arrived in the ranks as a 109-char
# .../<uuid>/tmp. Something in the launch path composed that value.
#
# Two derivation rules are consistent with what we observed, and they differ in
# whether the landed fix does anything at all:
#
#   (1) PALS appends <uuid>/tmp to $TMPDIR      -> ranks see /tmp/<uuid>/tmp
#                                                  = 45 chars, +32 = 77 < 107. FIXED.
#   (2) PALS composes from $PBS_TMPDIR, ignoring
#       whatever TMPDIR the job shell exported  -> ranks see the 109 again. INERT.
#
# Under (2) the one-line fix is a no-op that LOOKS like a fix, which is the same
# ${VAR:-default} failure mode that put the bug there in the first place: a
# guard that never fires reads exactly like a guard that fired.
#
# This probe costs one node for well under a minute and settles it by printing
# the resolved value from inside the ranks -- the only place it is evidence.
# Do NOT re-derive any of these paths from a login or job shell: both miss the
# UUID and report a comfortable ~68.
#
# Submit:  qsub -A <account> -q debug scripts/probe_tmpdir_pals.sh
# Read:    the ARMS table at the end. Arm A is the landed fix.

cd "${PBS_O_WORKDIR:-$PWD}" || exit 1

# No `set -u` anywhere near Lmod.
module load frameworks 2>/dev/null
source /flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/activate 2>/dev/null

echo "=== job shell (NOT evidence -- no UUID exists yet at this layer) ==="
echo "  TMPDIR      = '$TMPDIR' (${#TMPDIR})"
echo "  PBS_TMPDIR  = '$PBS_TMPDIR' (${#PBS_TMPDIR})"
echo

# Printed from inside a rank. TMPDIR is what PALS handed us; gettempdir() is
# what python will actually build sockets under (it can diverge -- tempfile
# silently falls back to /tmp when TMPDIR does not exist, which is how an
# off-node probe once "showed" a passing 36 while every rank was failing).
PROBE='
import os, tempfile, multiprocessing.util as mu, multiprocessing.connection as mc
t = os.environ.get("TMPDIR", "")
try:
    a = mc.arbitrary_address("AF_UNIX")
except Exception as e:
    a = "<%s>" % type(e).__name__
print("  rank TMPDIR=%r (%d)  gettempdir=%r  listener=%r (%d/107)%s"
      % (t, len(t), tempfile.gettempdir(), a, len(a),
         "  <-- OVER, nw>0 WILL HANG" if len(a) > 107 else "  ok"),
      flush=True)
'

run_arm () {
    echo "=== ARM $1: $2 ==="
    shift 2
    # -n 2 is enough: the value is per-node, not per-rank, and two ranks show
    # whether the per-launch UUID is shared (it is) without any log volume.
    "$@" mpiexec -n 2 -ppn 2 --cpu-bind depth --depth 16 --no-vni \
        python -c "$PROBE"
    echo "  (rc=$?)"
    echo
}

# A: exactly what scripts/lib/aurora_hsdp_env.sh now does. This is the arm
#    under test -- if it prints OVER, the landed fix is inert.
( export TMPDIR=/tmp; run_arm A "job-shell export TMPDIR=/tmp  [the landed fix]" )

# B: the other form ALCF's known-issues #7 offers. If A fails and B passes,
#    PALS reads its own env rather than the job shell's, and every launcher
#    needs the flag rather than the export. Written out rather than passed
#    through run_arm because the flag belongs to mpiexec, not to its prefix.
echo "=== ARM B: mpiexec --env TMPDIR=/tmp ==="
mpiexec -n 2 -ppn 2 --cpu-bind depth --depth 16 --no-vni \
    --env TMPDIR=/tmp python -c "$PROBE"
echo "  (rc=$?)"
echo

# C: baseline, untouched. Reproduces the 141 and proves the probe can SEE the
#    failure -- without it, two passes are not evidence that anything was fixed.
run_arm C "baseline, TMPDIR untouched  [expect OVER]"

cat <<'EOF'
=== HOW TO READ THIS ===
  A ok,  C OVER   -> the landed aurora_hsdp_env.sh fix works. Nothing to do.
  A OVER, B ok    -> PALS ignores the job-shell export; the export is INERT and
                     every mpiexec needs `--env TMPDIR=/tmp` instead. Revert the
                     one-liner to a comment and patch the launchers.
  A OVER, B OVER  -> neither layer wins; PALS composes from PBS_TMPDIR. Next
                     lever is exporting PBS_TMPDIR itself -- test it, do not
                     assume it, and check nothing else depends on that path.
  C ok            -> the probe is not reproducing the bug at all. Ignore A and B
                     entirely; a pass against a baseline that also passes says
                     nothing. Suspect the node already had a short TMPDIR.
EOF
