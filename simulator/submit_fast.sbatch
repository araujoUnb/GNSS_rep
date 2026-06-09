#!/bin/bash
#SBATCH --job-name=gnss_fast
#SBATCH --output=slurm_logs/%x_%a.out
#SBATCH --error=slurm_logs/%x_%a.err
#SBATCH --array=0-899          # = scenarios - 1; run `python runner_fast.py --list` to recount after editing grid_fast.yaml
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00

# One array task = ONE scenario (its full Monte-Carlo sequence). Scenarios run in
# parallel across tasks. Resumable: re-submitting skips completed mc_index values.
set -euo pipefail
cd "$(dirname "$0")"               # simulator/ dir
mkdir -p slurm_logs

# project venv with torch + bayesian-optimization==1.4.3
source ../.venv/bin/activate

# bayes_opt's GP is the bottleneck and is single-threaded; keep BLAS modest.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

python runner_fast.py "${SLURM_ARRAY_TASK_ID}"
