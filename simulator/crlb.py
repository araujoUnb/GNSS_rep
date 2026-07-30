"""Cramer-Rao lower bound for the LOS time-delay, consistent with sim_fast.

Mode-2 (delay-subspace) conditional bound. The estimators work on
Y_(2) = U(tau) G + N, with U = Qw^H C the calibration-free delay signatures and
the array response absorbed into the per-snapshot gain G. The LOS bound is

    CRB(tau0) = sigma2 / ( 2 * G00 * || Pi_U^perp u0' ||^2 ),

with G00 = K * P_LOS * ||a0||^2 and sigma2 the effective noise variance of the
mode-2 data (measured from a noise-only realization, so it matches the exact
forward of sim_fast). The calibration error enters only through
||a0||^2 -> ||a0||^2 + (epsilon/2) M (the sqrt(eps/2) phasor model), so the delay
bound is essentially flat in epsilon. Analytic -> instantaneous, no Monte Carlo.
"""

import numpy as np
import torch as th
import tensorly as tl

from sim_fast import FastDelaySim
from DelayEstimation import array_lin

LIGHT = 299792458.0


def _sigma2_eff(sim, n_rep=40):
    """Effective per-element noise variance of the mode-2 data, measured from the
    exact noise term of sim_fast (Zf = mode_dot(Z, Qw^H, 2))."""
    m, k, n, q = sim.m, sim.k, sim.n, sim.q
    vs = []
    for _ in range(n_rep):
        Z = tl.tensor(1 / np.sqrt(2) * th.randn(m, k, n)
                      + 1j * th.randn(m, k, n), dtype=th.complex64)
        Zf = tl.tenalg.mode_dot(Z, sim.QwH, 2)
        Y2 = tl.unfold(Zf, 2).numpy()
        vs.append(np.mean(np.abs(Y2) ** 2))
    return float(np.mean(vs))


def crlb_los_m(sim, angle_diff_deg, delay_diff, epsilon, doa_deg=10.0,
               tau0_frac=0.0, sigma2=None):
    """LOS ranging CRLB in metres for one operating point."""
    Tc = sim.Tc
    if sigma2 is None:
        sigma2 = _sigma2_eff(sim)
    tau0 = tau0_frac * Tc
    tau = th.tensor([tau0, tau0 + delay_diff * Tc])
    U = (sim.QwH @ sim._build_C(tau)).numpy()                 # (q, 2)
    h = Tc * 1e-3
    up = (sim.QwH @ sim._build_C(th.tensor([tau0 + h, float(tau[1])])))[:, 0]
    um = (sim.QwH @ sim._build_C(th.tensor([tau0 - h, float(tau[1])])))[:, 0]
    du0 = ((up - um) / (2 * h)).numpy()
    Q, _ = np.linalg.qr(U)
    PiU = np.eye(U.shape[0], dtype=complex) - Q @ Q.conj().T
    uperp2 = float(np.real(np.vdot(PiU @ du0, PiU @ du0)))
    P_LOS = 10 ** (sim.SNR_dB / 10)
    A = array_lin(th.tensor([doa_deg, doa_deg + angle_diff_deg]), sim.m).numpy()
    a0_energy = float(np.sum(np.abs(A[:, 0]) ** 2) + (epsilon / 2.0) * sim.m)
    G00 = sim.k * P_LOS * a0_energy
    crb = sigma2 / (2.0 * G00 * uperp2)
    return LIGHT * np.sqrt(crb)
