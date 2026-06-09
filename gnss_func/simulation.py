"""High-level Monte-Carlo simulation orchestrator.

Ties together the forward model (:class:`GNSSSystem`) and a swappable estimator
layer (:class:`gnss_func.estimators.DelayEstimator`). The heavy operators are
built once in the constructor, so a SLURM array task creates ONE runner and then
streams its slice of Monte-Carlo iterations through cheap per-realization calls.

Design points (auditability + resumability):
  * Every scenario is fully described by a :class:`SystemConfig` plus the
    experiment parameters; a **content hash** of all of these identifies the
    output directory, and the full configuration is dumped to ``config.yaml``.
  * Each Monte-Carlo realization uses ``seed = base_seed + mc_index`` so every
    sample is independently reproducible and the generating seed is recorded.
  * Results are flushed to ``results.csv`` every ``checkpoint_every`` iterations,
    storing the estimate, the estimation error, the estimator runtime, the seed
    and the scenario data. On restart the runner reads the CSV and resumes from
    the first missing ``mc_index`` (no rework).
"""

import os
import json
import time
import hashlib
from dataclasses import asdict

import numpy as np
import pandas as pd
import yaml

from gnss_func.config import SystemConfig
from gnss_func.system import GNSSSystem
from gnss_func.estimators import BSLDelayEstimator
from gnss_func.paths import default_results_dir

LIGHT_VELOCITY = 299792458.0


class MonteCarloSimulation:
    """Forward model + estimator with the heavy operators precomputed once."""

    def __init__(self, cfg: SystemConfig, estimator_cls=BSLDelayEstimator,
                 correlator_type="Qw", theta_deg_space=None, n_qw=None,
                 estimator_kwargs=None):
        self.cfg = cfg
        self.estimator_cls = estimator_cls
        self.correlator_type = correlator_type
        self.n_qw = n_qw
        self.estimator_kwargs = estimator_kwargs or {}
        self.system = GNSSSystem(cfg, correlator_type, n_qw)
        self.estimator = estimator_cls(self.system, theta_deg_space,
                                       **self.estimator_kwargs)

    def draw_geometry(self, delta_tau_frac):
        Tc = self.cfg.chip_period
        tau_los = np.random.rand() * Tc
        tau_vec = np.array([tau_los, tau_los + delta_tau_frac * Tc])
        theta_los = np.random.rand() * 360.0
        theta_vec = np.array([theta_los, theta_los + self.cfg.delta_phi_deg])
        return tau_vec, theta_vec, tau_los

    def run_realization(self, delta_tau_frac, cn0_db, seed):
        """One fully reproducible realization. ``seed`` determines everything.

        The global NumPy RNG is seeded here so that BOTH the forward model AND
        the estimator's internal randomness (e.g. the EM initialization inside
        ``eMM_BSL``, which draws from ``np.random``) are reproducible from this
        single seed. Returns the estimate, error (m), estimator runtime and seed.
        """
        np.random.seed(int(seed))
        tau_vec, theta_vec, tau_los = self.draw_geometry(delta_tau_frac)
        rx = self.system.simulate(tau_vec, theta_vec, cn0_db)

        t0 = time.perf_counter()
        tau_los_est = self.estimator.estimate(rx)
        estimator_time_s = time.perf_counter() - t0

        error_m = LIGHT_VELOCITY * abs(tau_los - tau_los_est)
        return {
            "seed": int(seed),
            "tau_los": tau_los,
            "tau_nlos": tau_vec[1],
            "tau_los_est": tau_los_est,
            "error_m": error_m,
            "estimator_time_s": estimator_time_s,
            "theta_los": theta_vec[0],
            "theta_nlos": theta_vec[1],
        }


class ScenarioRunner:
    """Auditable, resumable Monte-Carlo runner for a single scenario.

    A scenario = (SystemConfig, delta_tau_frac, cn0_db, base_seed, estimator).
    Output goes to ``<out_root>/<hash>/`` containing ``config.yaml`` (full audit)
    and ``results.csv`` (one row per Monte-Carlo realization).
    """

    def __init__(self, cfg: SystemConfig, delta_tau_frac, cn0_db=None,
                 base_seed=0, estimator_cls=BSLDelayEstimator,
                 correlator_type="Qw", theta_deg_space=None, n_qw=None,
                 estimator_kwargs=None, out_root=None, label=None):
        if out_root is None:
            out_root = default_results_dir()
        self.cfg = cfg
        self.delta_tau_frac = float(delta_tau_frac)
        self.cn0_db = float(cfg.cn0_db if cn0_db is None else cn0_db)
        self.base_seed = int(base_seed)
        self.label = label
        self.estimator_kwargs = estimator_kwargs or {}

        self.mc = MonteCarloSimulation(
            cfg, estimator_cls, correlator_type, theta_deg_space, n_qw,
            estimator_kwargs=self.estimator_kwargs,
        )
        if theta_deg_space is None:
            theta_deg_space = self.mc.estimator.theta_deg_space
        self._theta_summary = [
            float(np.min(theta_deg_space)),
            float(np.max(theta_deg_space)),
            int(np.size(theta_deg_space)),
        ]
        self.estimator_name = estimator_cls.__name__
        self.correlator_type = correlator_type
        self.n_qw = n_qw

        self.config = self._scenario_config()
        self.hash = self._config_hash(self.config)
        self.out_dir = os.path.join(out_root, self.hash)
        self.csv_path = os.path.join(self.out_dir, "results.csv")
        self.yaml_path = os.path.join(self.out_dir, "config.yaml")

    # ------------------------------------------------------------------ config
    def _scenario_config(self):
        """Every parameter that defines the scenario (used for hash + audit)."""
        return {
            "system": asdict(self.cfg),
            "experiment": {
                "delta_tau_frac": self.delta_tau_frac,
                "cn0_db": self.cn0_db,
                "base_seed": self.base_seed,
                "seed_rule": "seed = base_seed + mc_index",
            },
            "estimator": {
                "name": self.estimator_name,
                "correlator_type": self.correlator_type,
                "n_qw": self.n_qw,
                "theta_deg_space_min_max_n": self._theta_summary,
                "kwargs": {k: (list(v) if isinstance(v, (np.ndarray,)) else v)
                           for k, v in self.estimator_kwargs.items()},
            },
            "label": self.label,
        }

    @staticmethod
    def _config_hash(config):
        blob = json.dumps(config, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def _write_yaml(self):
        os.makedirs(self.out_dir, exist_ok=True)
        if not os.path.exists(self.yaml_path):
            doc = dict(self.config)
            doc["config_hash"] = self.hash
            with open(self.yaml_path, "w") as f:
                yaml.safe_dump(doc, f, sort_keys=False)

    # ------------------------------------------------------------------ resume
    def _completed_indices(self):
        if not os.path.exists(self.csv_path):
            return set()
        try:
            done = pd.read_csv(self.csv_path)["mc_index"].astype(int).tolist()
            return set(done)
        except Exception:
            return set()

    def _flush(self, rows):
        if not rows:
            return
        df = pd.DataFrame(rows)
        header = not os.path.exists(self.csv_path)
        df.to_csv(self.csv_path, mode="a", header=header, index=False)

    # --------------------------------------------------------------------- run
    def run(self, n_mc, checkpoint_every=10, verbose=True):
        self._write_yaml()
        done = self._completed_indices()
        buffer = []
        n_new = 0
        for j in range(n_mc):
            if j in done:
                continue
            row = self.mc.run_realization(
                self.delta_tau_frac, self.cn0_db, self.base_seed + j
            )
            row.update({
                "mc_index": j,
                "config_hash": self.hash,
                "delta_tau_frac": self.delta_tau_frac,
                "delta_phi_deg": float(self.cfg.delta_phi_deg),
                "epsilon": float(getattr(self.cfg, "epsilon", 0.0)),
                "xi": float(self.estimator_kwargs.get("xi", float("nan"))),
                "i_max": int(self.estimator_kwargs["i_max"])
                if "i_max" in self.estimator_kwargs else -1,
                "cn0_db": self.cn0_db,
                "estimator": self.estimator_name,
            })
            buffer.append(row)
            n_new += 1
            if len(buffer) >= checkpoint_every:
                self._flush(buffer)
                buffer = []
                if verbose:
                    print(f"[{self.hash}] checkpoint at mc_index={j} "
                          f"({n_new} new)", flush=True)
        self._flush(buffer)
        if verbose:
            print(f"[{self.hash}] done: {n_new} new realizations "
                  f"(total target {n_mc}) -> {self.csv_path}", flush=True)
        return self.csv_path
