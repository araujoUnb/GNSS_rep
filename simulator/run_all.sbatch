#!/bin/bash
# Task script for the COMBINED single-array campaign (GNSS_FAST_ALL=1): one task
# = one scenario across ALL campaigns/. Submit via submit_capped.sh, which sets
# the array range and the %MAXP global concurrency cap (default 28).
#SBATCH --output=slurm_logs/%x_%a.out
#SBATCH --error=slurm_logs/%x_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p slurm_logs
source ../.venv/bin/activate

# 1 thread per task (bayes_opt GP is single-threaded) -> #tasks == #threads
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GNSS_FAST_ALL=1

python runner_fast.py "${SLURM_ARRAY_TASK_ID}"
