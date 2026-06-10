#!/bin/bash
# One SLURM array job = one analysis (campaigns/<name>.yaml). The job-name, the
# --array range and GNSS_FAST_GRID are set by submit_all.sh (or pass them by hand:
#   sbatch --job-name=gnss_eps --array=0-5 --export=ALL,GNSS_FAST_GRID=campaigns/eps.yaml run_one.sbatch
# Each array task runs ONE scenario's full Monte-Carlo (resumable).
#SBATCH --output=slurm_logs/%x_%a.out
#SBATCH --error=slurm_logs/%x_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p slurm_logs
source ../.venv/bin/activate

# bayes_opt GP is single-threaded; keep BLAS modest.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "analysis=${GNSS_FAST_GRID:-grid_fast.yaml} task=${SLURM_ARRAY_TASK_ID}"
python runner_fast.py "${SLURM_ARRAY_TASK_ID}"
