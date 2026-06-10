# Runbook for Claude on the cluster — GNSS delay-estimation campaigns

You are running on an HPC cluster with **SLURM**. Your job: run the Monte-Carlo
simulation campaigns for the GNSS time-delay paper revision, monitor them, and
aggregate the results into the figure CSVs. Everything lives in `simulator/`.

## What this is (read first)
- The faithful simulator is `simulator/sim_fast.py` (`FastDelaySim`): an efficient
  replica of the ORIGINAL `simulator/DelayEstimation.py` (PyTorch + the
  `bayes_opt` library). It is validated to reproduce the original **seed-by-seed**
  (LSKRF/BO identical, BO+Ref ~0.013 m). Methods per run: LSKRF, LSKRF+Ref, BO,
  BO+Ref.
- **Do NOT change the BO engine.** Use `bo_engine: bayesopt` and
  `objective: exact` (the defaults in the campaign grids). `botorch` was tested
  and does NOT match `bayes_opt` — do not use it for paper data.
- **Do NOT commit** the big shelves (`*.dat`) or `results_fast/` (already
  git-ignored). The published `results.dat` is stale; we regenerate everything
  with `sim_fast`.

## One-time setup
```bash
cd <repo>            # the GNSS_rep checkout
python -m venv .venv
source .venv/bin/activate
pip install -e .     # installs numpy scipy pandas tensorly scikit-learn pyyaml
                     # torch  bayesian-optimization==1.4.3   (pinned API!)
python -c "import torch, bayes_opt, tensorly; print('env ok')"
```

## Run the campaigns (one SLURM array job per analysis)
Each analysis is `simulator/campaigns/<name>.yaml` (one varying axis at a fixed
operating point): `eps, dphi, snr, dtau, imax, xi` (+ `smoke` for testing).
```bash
cd simulator
DRYRUN=1 ./submit_all.sh smoke     # 1) print the sbatch command, submit nothing
./submit_all.sh smoke              # 2) submit the tiny smoke array; confirm it finishes
./submit_all.sh                    # 3) submit ALL analyses (or: ./submit_all.sh eps snr)
```
- One array job per analysis, **named `gnss_<analysis>`** → `squeue -u $USER`
  (or `sacct`) tells you which analysis is running/finished.
- **Resumable & idempotent:** re-run `./submit_all.sh <name>` any time; each task
  skips already-done `mc_index` (results flush every 10 MC). Safe after timeouts.
- Cost: ~4–7 s per Monte-Carlo iteration × `n_mc` (1000) per scenario. Each
  analysis has 4–6 scenarios. Bump `--time`/`--cpus-per-task` in `run_one.sbatch`
  if needed (the `bayes_opt` GP is single-threaded; more CPUs won't speed one task).

## Output layout
`simulator/results_fast/<analysis>/<hash>.csv` + identically-named `<hash>.yaml`.
The `<hash>` is a digest of ALL simulation parameters (system + scenario +
estimator + base_seed); the `.yaml` stores those parameters for audit. `n_mc` is
NOT in the hash, so extending `n_mc` resumes the same files.

## Aggregate when done
```bash
cd simulator
python aggregate_fast.py results_fast      # -> summary_fast.csv + figure_csv/*.csv
```
This writes `summary_fast.csv` (per-scenario, per-method mean/median/outlier) and
the wide figure CSVs in `figure_csv/` (columns match the paper's
`paper/plots/data_revision/*.csv`). **Bring back** `summary_fast.csv` and
`figure_csv/*.csv` (e.g. scp / git on a branch) — those feed the paper's pgfplots.
(The `--to-paper` flag only works on the author's laptop where the paper repo is.)

## If something fails
- Import errors → the `.venv` is missing deps; redo setup (note the pinned
  `bayesian-optimization==1.4.3`).
- A task timed out → just resubmit that analysis; it resumes.
- Verify a single scenario quickly:
  `GNSS_FAST_GRID=campaigns/eps.yaml python runner_fast.py --list` then
  `... python runner_fast.py 0`.
- More detail: `simulator/README.md`.
