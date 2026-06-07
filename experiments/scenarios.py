"""Build the list of Monte-Carlo scenarios from ``grid.yaml``.

A *scenario* is one (estimator, delta_tau_frac, epsilon, cn0) combination. The
ordering is deterministic, so a SLURM array index maps to a fixed scenario.
"""

import os
import itertools
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load_grid(path=None):
    if path is None:
        path = os.environ.get("GNSS_GRID", os.path.join(HERE, "grid.yaml"))
    with open(path) as f:
        return yaml.safe_load(f)


def build_scenarios(grid=None):
    if grid is None:
        grid = load_grid()
    exp = grid["experiment"]
    combos = itertools.product(
        exp["estimators"], exp["cn0_db"], exp["delta_tau_frac"], exp["epsilon"]
    )
    scenarios = []
    for est, cn0, dtau, eps in combos:
        scenarios.append({
            "estimator": est,
            "cn0_db": float(cn0),
            "delta_tau_frac": float(dtau),
            "epsilon": float(eps),
            "base_seed": int(exp["base_seed"]),
            "n_mc": int(exp["n_mc"]),
            "checkpoint_every": int(exp["checkpoint_every"]),
        })
    return scenarios


if __name__ == "__main__":
    sc = build_scenarios()
    print(f"{len(sc)} scenarios (SLURM array 0-{len(sc) - 1})")
    for i, s in enumerate(sc):
        print(i, s["estimator"], "dtau=%.2f" % s["delta_tau_frac"],
              "eps=%.3f" % s["epsilon"], "cn0=%.0f" % s["cn0_db"])
