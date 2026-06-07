"""Compute the CRLB ranging-bound curves over the (delta_tau, epsilon) grid.

This is a deterministic formula evaluation (no Monte-Carlo): it builds the heavy
forward operators ONCE and then evaluates the conditional CRB of Appendix A at
each grid point. Output: experiments/crlb_curve.csv, ready to overlay on the
RMSE figures of the paper.

Usage:
    python experiments/compute_crlb.py
"""

import os
import numpy as np
import pandas as pd

from gnss_func.config import SystemConfig
from gnss_func.system import GNSSSystem
from gnss_func.crlb import crlb_tau
from scenarios import load_grid

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    grid = load_grid()
    sysp = grid["system"]
    exp = grid["experiment"]
    cn0 = float(exp["cn0_db"][0])

    base_cfg = dict(
        sat_id=sysp["sat_id"], bandwidth=sysp["bandwidth"], fc=sysp["fc"],
        time_period=sysp["time_period"], n_antennas=sysp["n_antennas"],
        n_epochs=sysp["n_epochs"],
        n_correlators=sysp.get("n_correlators", 11),
        delay_granularity=sysp["delay_granularity"],
        cn0_db=cn0, delta_phi_deg=sysp["delta_phi_deg"],
    )
    # operators do not depend on epsilon (epsilon enters the CRB noise model),
    # so build the system once.
    system = GNSSSystem(SystemConfig(**base_cfg))
    Tc = system.cfg.chip_period

    rows = []
    theta = np.array([100.0, 100.0 + sysp["delta_phi_deg"]])
    for dtau in exp["delta_tau_frac"]:
        tau0 = 0.4 * Tc
        tau_vec = np.array([tau0, tau0 + dtau * Tc])
        for eps in exp["epsilon"]:
            _, los_m = crlb_tau(system, tau_vec, theta, cn0_db=cn0,
                                epsilon=(eps if eps > 0 else None))
            rows.append({"delta_tau_frac": dtau, "epsilon": eps,
                         "cn0_db": cn0, "crlb_los_m": los_m})
            print(f"  dtau={dtau:.2f} eps={eps:.3f} -> CRLB {los_m:.4f} m",
                  flush=True)

    out = os.path.join(HERE, "crlb_curve.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
