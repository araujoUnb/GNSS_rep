"""Run one scenario (one SLURM array task).

Usage:
    python experiments/run_scenario.py <scenario_index>
    # or, under SLURM:
    python experiments/run_scenario.py          # uses $SLURM_ARRAY_TASK_ID

Each task builds the (heavy) forward operators once, then streams its
Monte-Carlo realizations through the resumable, auditable ScenarioRunner.
"""

import os
import sys

from gnss_func.config import SystemConfig
from gnss_func.simulation import ScenarioRunner
from gnss_func.estimators import ESTIMATORS
from scenarios import load_grid, build_scenarios


def estimator_kwargs(name, grid, xi, i_max=None):
    p = grid["estimator_params"]
    if name in ("BO", "BO+Ref"):
        return {"n_grid": p["n_grid"],
                "i_max": int(i_max) if i_max is not None else p["bo_i_max"],
                "n_init": p["bo_n_init"], "xi": float(xi)}
    if name in ("LSKRF", "LSKRF+Ref"):
        return {"n_grid": p["n_grid"]}
    return {}


def main():
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    else:
        idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    grid = load_grid()
    scenarios = build_scenarios(grid)
    if idx < 0 or idx >= len(scenarios):
        raise SystemExit(f"index {idx} out of range (0..{len(scenarios)-1})")
    sc = scenarios[idx]
    print(f"[scenario {idx}] {sc}", flush=True)

    sysp = grid["system"]
    cfg = SystemConfig(
        sat_id=sysp["sat_id"], bandwidth=sysp["bandwidth"], fc=sysp["fc"],
        time_period=sysp["time_period"], n_antennas=sysp["n_antennas"],
        n_epochs=sysp["n_epochs"],
        n_correlators=sysp.get("n_correlators", 11),
        delay_granularity=sysp["delay_granularity"],
        cn0_db=sc["cn0_db"], delta_phi_deg=sc["delta_phi_deg"],
        epsilon=sc["epsilon"], smr_db=sysp.get("smr_db", 5.0),
    )

    est_cls = ESTIMATORS[sc["estimator"]]
    runner = ScenarioRunner(
        cfg, delta_tau_frac=sc["delta_tau_frac"], cn0_db=sc["cn0_db"],
        base_seed=sc["base_seed"], estimator_cls=est_cls,
        estimator_kwargs=estimator_kwargs(sc["estimator"], grid, sc["xi"],
                                          sc.get("i_max")),
        out_root=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", grid.get("out_root", "results")),
        label=sc["estimator"],
    )
    runner.run(n_mc=sc["n_mc"], checkpoint_every=sc["checkpoint_every"])


if __name__ == "__main__":
    main()
