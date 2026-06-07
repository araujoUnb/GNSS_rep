# GNSS time-delay estimation — Monte-Carlo simulator

Modular, reproducible and SLURM-ready simulator for GNSS time-delay estimation
under multipath and antenna-array calibration errors. Refactored from the
original scripts into a layered design where the **forward model** is built once
and the **estimator** is a swappable layer.

## Install (any machine, no path edits)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

All internal paths (C/A-code cache, default results dir) are resolved relative to
the package via `__file__`, so the simulator runs unchanged on a laptop or a
SLURM node.

## Architecture (layers)

| Layer | File | Role |
|-------|------|------|
| Config | `gnss_func/config.py` | `SystemConfig`: every scenario parameter (incl. `epsilon` calibration error) |
| Forward model | `gnss_func/system.py` | `GNSSSystem`: builds heavy operators (`Q`, `Qw`, …) **once**; `simulate()` per realization |
| Estimators | `gnss_func/estimators.py` | swappable `DelayEstimator` layer: `BSL`, `LSKRF`, `BO`, `BO+Ref`, `LSKRF+Ref` (registry `ESTIMATORS`) |
| Orchestrator | `gnss_func/simulation.py` | `ScenarioRunner`: config hash, YAML audit, CSV checkpoint, resume, `seed = base_seed + mc_index` |
| Metrics | `gnss_func/metrics.py` | RMSE, percentiles, outlier rate |
| CRLB | `gnss_func/crlb.py` | conditional Cramér-Rao bound (letter Appendix A) |

The original `model.py` / `singlePolModel_estimator` are kept for back-compat.

### Why it is fast
The delay-independent operators (`Q`, `Qw`, the delay dictionary) used to be
rebuilt on every Monte-Carlo iteration. They are now precomputed once per
process, cutting the per-iteration cost from ~15 s to ~0.6 s (~25×). On SLURM
each array task pays the build once and streams its Monte-Carlo slice.

## Reproducibility & auditing
- Each realization uses `seed = base_seed + mc_index` (the global NumPy RNG is
  seeded so the estimator's internal randomness is reproducible too).
- Each scenario gets a content **hash** of its full configuration; results go to
  `results/<hash>/` with `config.yaml` (full audit) and `results.csv`.
- `results.csv` columns: `mc_index, seed, tau_los, tau_nlos, tau_los_est,
  error_m, estimator_time_s, theta_los, theta_nlos, delta_tau_frac, cn0_db,
  estimator, config_hash`.
- CSV is flushed every `checkpoint_every` (default 10) realizations; re-running a
  scenario **resumes** from the first missing `mc_index` (no rework).

## Running the experiments (complement the response letter)

Scenarios are the cartesian product defined in `experiments/grid.yaml`
(estimators × Δτ × ε × C/N₀).

```bash
cd experiments && python scenarios.py     # list scenarios + SLURM array size
python run_scenario.py 0                   # run one scenario locally
sbatch submit_array.sbatch                 # run the whole grid on SLURM
python compute_crlb.py                     # CRLB curves -> crlb_curve.csv (no MC)
python aggregate.py                        # -> summary.csv, outlier_table.csv
```

> **Bandwidth.** `grid.yaml` uses `B = 1023e6` (⇒ `N = 2·B·T = 2,046,000`),
> matching the reference data — **heavy**, run on SLURM. For quick local code
> checks use a light grid (`bandwidth: 1.023e6`, `N = 2046`):
> `GNSS_GRID=light_grid.yaml python run_scenario.py 0`.

## Validation status
- **Reproduction of the legacy pipeline**: ✅ the refactor reproduces the stored
  `deltaTau` RMSE means (same scale and trend; identical `tau_los_est` on
  point 0).
- **Orchestrator** (hash/YAML/CSV/resume/seed): ✅ validated.
- **New estimators (LSKRF/BO/+Ref), epsilon model, CRLB, metrics**: ✅
  code-validated on a *light* grid (they run, refinement improves, CRLB grows
  with epsilon). Their **production numbers at `B = 1023e6` must be generated on
  SLURM** (too heavy for a laptop).
