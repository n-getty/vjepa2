#!/usr/bin/env bash
# Hold a single Aurora debug node for interactive investigation (per
# hpc-iteration-discipline skill). SSH in, drive tests directly, observe
# live iostat/top state instead of guessing from one-shot batch logs.
#
#   qsub -A ModCon -q debug -l select=1 scripts/hold_debug_node.sh
#PBS -N hold_debug
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:55:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

echo "holding $(hostname) for interactive use"
sleep 3300
