"""CRLB curves for the paper figures: LOS ranging bound versus C/N0 and versus
the multipath delay separation.

Both use the mode-2 conditional bound of ``crlb.py``. The bound depends on the
delay separation through the projection that removes the whole delay subspace,
so it must be evaluated point by point along the Delta-tau axis instead of being
carried over from the reference operating point.

Writes ``figure_csv/R4C5_crlb.csv`` and ``figure_csv/R1C4_dtau_crlb.csv``.
"""

import os

import numpy as np
import pandas as pd

from sim_fast import FastDelaySim
from crlb import crlb_los_m, _sigma2_eff

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figure_csv")

DPHI, EPS = 5.0, 5e-3
CN0_GRID = [38.0, 43.0, 48.0, 53.0, 58.0]
DTAU_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
DTAU_REF, CN0_REF = 0.5, 48.0


def main():
    np.random.seed(0)

    rows = []
    for cn0 in CN0_GRID:
        sim = FastDelaySim(cn0=cn0)
        rows.append({"cn0_db": cn0,
                     "crlb_m": crlb_los_m(sim, DPHI, DTAU_REF, EPS)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "R4C5_crlb.csv"), index=False)
    print("R4C5_crlb.csv")
    print(pd.DataFrame(rows).to_string(index=False))

    sim = FastDelaySim(cn0=CN0_REF)
    sigma2 = _sigma2_eff(sim)          # one noise calibration for the whole sweep
    rows = [{"delay_diff": dt,
             "crlb_m": crlb_los_m(sim, DPHI, dt, EPS, sigma2=sigma2)}
            for dt in DTAU_GRID]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "R1C4_dtau_crlb.csv"),
                              index=False)
    print("\nR1C4_dtau_crlb.csv")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
