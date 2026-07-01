#!/usr/bin/env bash
# Hold an Aurora node for iterative debugging. Sleeps so we can ssh in.
#
# Submit with:
#   qsub -A ModCon -q debug -l select=1 -l walltime=01:00:00 \
#        -l filesystems=home:flare scripts/hold_node_aurora.sh
#
# Then: qstat -u $USER to find the assigned node, ssh <node> for an
# interactive shell with full PBS env intact.
#PBS -N vjepa_hold
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/vjepa_hold.${PBS_JOBID}.log

set -eo pipefail
mkdir -p /flare/ModCon/ngetty/logs

echo "HOLD START: $(date)"
echo "PBS_JOBID=$PBS_JOBID"
echo "PBS_NODEFILE=$PBS_NODEFILE"
echo "Allocated nodes:"
cat "$PBS_NODEFILE"
echo
echo "ssh into the first node above and use the repo at"
echo "  /lus/flare/projects/ModCon/ngetty/vjepa2"
echo "to run experiments. Sleeping until walltime."
sleep infinity
