"""Build the list of Monte-Carlo scenarios from ``grid.yaml``.

A *scenario* is one (estimator, cn0, delta_tau_frac, delta_phi_deg, epsilon, xi)
combination. The ordering is deterministic, so a SLURM array index maps to a
fixed scenario.
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
    # default the optional sweep axes so older grids still load
    delta_phi_list = exp.get("delta_phi_deg", [grid["system"].get("delta_phi_deg", 5.0)])
    xi_list = exp.get("xi", [grid.get("estimator_params", {}).get("bo_xi", 0.1)])
    # i_max axis only affects BO/BO+Ref; None -> use estimator_params default
    i_max_list = exp.get("i_max", [None])
    combos = itertools.product(
        exp["estimators"], exp["cn0_db"], exp["delta_tau_frac"],
        delta_phi_list, exp["epsilon"], xi_list, i_max_list,
    )
    scenarios = []
    for est, cn0, dtau, dphi, eps, xi, imax in combos:
        # i_max is meaningless for LSKRF; collapse to one (None) to avoid dup runs
        if est in ("LSKRF", "LSKRF+Ref") and imax is not None:
            continue
        scenarios.append({
            "estimator": est,
            "cn0_db": float(cn0),
            "delta_tau_frac": float(dtau),
            "delta_phi_deg": float(dphi),
            "epsilon": float(eps),
            "xi": float(xi),
            "i_max": (None if imax is None else int(imax)),
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
              "dphi=%.1f" % s["delta_phi_deg"], "eps=%.3f" % s["epsilon"],
              "xi=%.3f" % s["xi"], "cn0=%.0f" % s["cn0_db"])
