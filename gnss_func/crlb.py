"""Cramer-Rao Lower Bound for the GNSS time-delay (conditional model).

Implements Appendix A of the response letter, specialized to the compressed
correlator-subspace model used in this repository: each snapshot is

    y[k] = B(tau, theta) gamma[k] + n[k],   n[k] ~ CN(0, sigma^2 I),

with manifold columns  b_l = u_l (x) a_l,  where
    u_l = proj^T c(tau_l)   (delay signature, length n_qw)
    a_l = array_lin(theta_l) (steering vector, length M).

The bound is a deterministic formula evaluation (no Monte-Carlo). It returns the
LOS ranging bound in meters,  c * sqrt( CRB(tau_0) ), directly comparable to the
RMSE curves. Set ``epsilon>0`` to use the effective-noise (calibration-error)
covariance of Appendix A.5.
"""

import numpy as np

from gnss_func.gnss_function import build_signal_C
from gnss_func.array import array_lin

LIGHT_VELOCITY = 299792458.0


def _delay_signature(system, tau_vec):
    C = build_signal_C(system.cfg.bandwidth, system.cfg.chip_period,
                       system.cfg.time_period, np.asarray(tau_vec, float),
                       system.CA_FFT)
    return system.proj.conj().T @ C            # (n_qw, L)


def _delay_signature_grad(system, tau_vec, h=None):
    """d u_l / d tau_l by central finite differences (per path)."""
    tau_vec = np.asarray(tau_vec, float)
    if h is None:
        h = system.cfg.chip_period * 1e-3
    U = _delay_signature(system, tau_vec)
    dU = np.zeros_like(U)
    for l in range(tau_vec.size):
        tp = tau_vec.copy(); tp[l] += h
        tm = tau_vec.copy(); tm[l] -= h
        up = _delay_signature(system, tp)[:, l]
        um = _delay_signature(system, tm)[:, l]
        dU[:, l] = (up - um) / (2 * h)
    return U, dU


def _steering_grad(theta_deg_vec, n_antennas, h_deg=1e-3):
    A = array_lin(np.asarray(theta_deg_vec, float), n_antennas)
    tp = np.asarray(theta_deg_vec, float) + h_deg
    tm = np.asarray(theta_deg_vec, float) - h_deg
    dA = (array_lin(tp, n_antennas) - array_lin(tm, n_antennas)) / (2 * h_deg)
    return A, dA


def crlb_tau(system, tau_vec, theta_deg_vec, cn0_db=None, amp_gram=None,
             angles_known=False, epsilon=None):
    """Conditional CRB for the delay vector. Returns the full CRB matrix (s^2)
    over the geometric parameters and the LOS ranging bound in meters.

    Parameters
    ----------
    amp_gram : (L,L) array or None
        Amplitude Gram \hat P = (1/K) sum gamma gamma^H. Defaults to I_L
        (the model uses unit-power CN(0,1) taps per path).
    angles_known : bool
        If True, only delays are estimated (angles assumed known).
    epsilon : float or None
        If given (>0), uses the effective-noise covariance R = sigma^2 I +
        eps^2 sum_l (u_l u_l^H (x) I_M) of Appendix A.5 instead of white noise.
    """
    cfg = system.cfg
    M = cfg.n_antennas
    L = np.asarray(tau_vec).size
    K = cfg.n_epochs
    sigma2 = system.noise_var(cfg.cn0_db if cn0_db is None else cn0_db)

    U, dU = _delay_signature_grad(system, tau_vec)        # (nqw,L)
    A, dA = _steering_grad(theta_deg_vec, M)              # (M,L)

    # manifold and its derivatives, columns b_l = u_l (x) a_l
    def kr(Umat, Amat):
        return np.stack([np.kron(Umat[:, l], Amat[:, l])
                         for l in range(L)], axis=1)

    B = kr(U, A)                       # (nqw*M, L)
    Dt = kr(dU, A)                     # d/dtau
    Dp = kr(U, dA)                     # d/dtheta

    nqw = U.shape[0]
    QM = nqw * M

    # effective noise covariance (white, or calibration-error inflated)
    if epsilon:
        R = sigma2 * np.eye(QM, dtype=complex)
        IM = np.eye(M, dtype=complex)
        for l in range(L):
            R += (epsilon ** 2) * np.kron(np.outer(U[:, l], U[:, l].conj()), IM)
        Rinv = np.linalg.inv(R)
    else:
        Rinv = np.eye(QM, dtype=complex) / sigma2

    # R-weighted noise-subspace operator:
    #   M = R^{-1} - R^{-1} B (B^H R^{-1} B)^{-1} B^H R^{-1}
    # (reduces to Pi_B^perp / sigma^2 in the white-noise case)
    RB = Rinv @ B
    M_op = Rinv - RB @ np.linalg.solve(B.conj().T @ RB, RB.conj().T)

    if amp_gram is None:
        amp_gram = np.eye(L)
    Pt = np.asarray(amp_gram).T

    if angles_known:
        Delta = Dt
        W = Pt
    else:
        Delta = np.hstack([Dt, Dp])
        W = np.block([[Pt, Pt], [Pt, Pt]])

    Jbar = 2 * K * np.real((Delta.conj().T @ M_op @ Delta) * W)
    crb = np.linalg.inv(Jbar)
    los_rmse_m = LIGHT_VELOCITY * np.sqrt(np.real(crb[0, 0]))
    return crb, los_rmse_m
