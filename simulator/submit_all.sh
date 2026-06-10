#!/bin/bash
# Submit ONE SLURM array job per analysis in campaigns/. Each job is named
# gnss_<analysis>, so `squeue -u $USER` (or sacct) immediately shows which
# analysis is running/finished. Resumable: re-run to fill missing mc_index.
#
# Usage:
#   ./submit_all.sh                 # submit every campaigns/*.yaml
#   ./submit_all.sh eps snr         # submit only the named analyses
set -euo pipefail
cd "$(dirname "$0")"
source ../.venv/bin/activate 2>/dev/null || true

if [ "$#" -gt 0 ]; then
    grids=(); for a in "$@"; do grids+=("campaigns/$a.yaml"); done
else
    grids=(campaigns/*.yaml)
fi

for g in "${grids[@]}"; do
    name=$(basename "$g" .yaml)
    n=$(GNSS_FAST_GRID="$g" python runner_fast.py --count)
    cmd=(sbatch --job-name="gnss_${name}" --array="0-$((n-1))"
         --export=ALL,GNSS_FAST_GRID="$g" run_one.sbatch)
    echo "gnss_${name}: ${n} scenarios (array 0-$((n-1)))"
    if [ "${DRYRUN:-0}" = "1" ]; then echo "  DRYRUN: ${cmd[*]}"; else "${cmd[@]}"; fi
done
echo "done. track with:  squeue -u \$USER   (job names gnss_<analysis>)"
