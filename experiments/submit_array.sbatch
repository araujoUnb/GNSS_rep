#!/bin/bash
#SBATCH --job-name=gnss_delay
#SBATCH --output=slurm_logs/%x_%a.out
#SBATCH --error=slurm_logs/%x_%a.err
#SBATCH --array=0-83          # = number of scenarios - 1 (see: python scenarios.py)
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00

# --- environment ---
# Each array task runs ONE scenario (its full Monte-Carlo sequence) on one node.
# Different scenarios run in parallel across array tasks. Resumable: re-submitting
# the same array skips already-completed mc_index values.

set -euo pipefail
cd "$(dirname "$0")/.."           # repo root
mkdir -p experiments/slurm_logs

# activate the project venv (created with: python -m venv .venv; pip install -e .)
source .venv/bin/activate

# keep BLAS single-threaded per task unless you raise --cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd experiments
python run_scenario.py "${SLURM_ARRAY_TASK_ID}"
