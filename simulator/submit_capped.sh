#!/bin/bash
# Submit ALL campaigns (campaigns/*.yaml except smoke) as ONE SLURM array with a
# GLOBAL concurrency cap MAXP (default 28). With cpus-per-task=1, at most MAXP
# scenarios run at once => at most MAXP threads busy (the cluster has 32; we
# leave a few free). Resumable: re-run any time.
#
# Usage:
#   ./submit_capped.sh            # cap at 28 threads (default)
#   MAXP=24 ./submit_capped.sh    # cap at 24
#   DRYRUN=1 ./submit_capped.sh   # print the sbatch command, submit nothing
set -euo pipefail
cd "$(dirname "$0")"
source ../.venv/bin/activate 2>/dev/null || true

MAXP=${MAXP:-28}
N=$(GNSS_FAST_ALL=1 python runner_fast.py --count)
cmd=(sbatch --job-name=gnss_campaign --array="0-$((N-1))%${MAXP}"
     --export=ALL,GNSS_FAST_ALL=1 run_all.sbatch)
echo "ALL campaigns: ${N} scenarios, concurrency cap %${MAXP}"
if [ "${DRYRUN:-0}" = "1" ]; then echo "DRYRUN: ${cmd[*]}"; else "${cmd[@]}"; fi
echo "track: squeue -u \$USER (job gnss_campaign); outputs in results_fast/<analysis>/"
