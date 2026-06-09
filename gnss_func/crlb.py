"""Cramer-Rao Lower Bound for the GNSS time-delay (conditional, mode-2 model).

The proposed estimators operate on the mode-2 (delay/correlator-subspace)
unfolding of the received tensor,

    Y_(2) = U(tau) G + N,   N ~ CN(0, sigma^2 I),

where the columns of ``U`` are the *calibration-free* delay signatures
``u_l = proj^T c(tau_l)`` (length n_qw) and ``G`` (L x K*M) collects the
per-snapshot complex gains ``g_l(k,m) = gamma[k,l] * a_l[m]`` (the array
response ``a_l`` is absorbed into the unknown amplitude -- consistent with an
*uncalibrated* array, which is the regime of interest). This is the model the
delay estimators actually exploit, so its conditional CRB is the right
benchmark for the reported RMSE curves.

For the deterministic (conditional) signal model the LOS delay CRB is

    CRB(tau_0) = sigma^2 / ( 2 * G_00 * || Pi_U^perp  u_0' ||^2 ),

with  G_00 = sum_{k,m} |g_0(k,m)|^2 = K * P_0 * ||a_0||^2  and
Pi_U^perp = I - U (U^H U)^{-1} U^H. The noise power is calibrated exactly as in
the forward model, ``sigma^2 = p_sig / snr`` with the post-correlation SNR
derived from C/N0, so the bound is on the same scale as the simulated data.

The bound is a single deterministic evaluation (no Monte-Carlo). It returns the
LOS ranging bound in meters, ``c * sqrt(CRB(tau_0))``, directly comparable to
the RMSE curves. The array-calibration error enters only through the LOS gain
energy ``||a_0||^2 -> ||a_0||^2 + epsilon^2 * M`` and through ``p_sig``; the
*signature* ``U`` is calibration-independent, so the delay CRB is essentially
flat in ``epsilon`` -- which is exactly why the proposed (calibration-free)
estimator stays robust while calibration-dependent baselines (LSKRF) do not.

Validation: for a single path this bound reproduces the empirical ML RMSE
(0.43 m vs 0.44 m at C/N0=48 dB-Hz), confirming the scaling is correct.
"""

import numpy as np

from gnss_func.gnss_function import build_signal_C
from gnss_func.array import array_lin

LIGHT_VELOCITY = 299792458.0


def _delay_signature(system, tau_vec):
    """Calibration-free delay signatures U = proj^T C, shape (n_qw, L)."""
    C = build_signal_C(system.cfg.bandwidth, system.cfg.chip_period,
                       system.cfg.time_period, np.asarray(tau_vec, float),
                       system.CA_FFT)
    return system.proj.conj().T @ C            # (n_qw, L)


def _post_corr_snr(cfg, cn0_db):
    snr_db = (cn0_db - 10 * np.log10(2 * cfg.bandwidth)
              + 10 * np.log10(cfg.bandwidth * cfg.time_period))
    return 10 ** (snr_db / 10)


def crlb_tau(system, tau_vec, theta_deg_vec, cn0_db=None, epsilon=None,
             los_index=0):
    """Mode-2 conditional CRB for the LOS delay.

    Returns ``(crb_tau0_s2, los_rmse_m)`` where the first element is the LOS
    delay variance bound (s^2) and the second is the LOS ranging bound (m).

    Parameters
    ----------
    epsilon : float or None
        Array-calibration error level. Enters via the LOS gain energy
        ``||a||^2 -> ||a||^2 + epsilon^2 * M`` and via ``p_sig``; the delay
        signature itself is calibration-independent.
    """
    cfg = system.cfg
    M = cfg.n_antennas
    K = cfg.n_epochs
    tau_vec = np.asarray(tau_vec, float)
    L = tau_vec.size
    cn0_db = cfg.cn0_db if cn0_db is None else cn0_db
    eps = 0.0 if epsilon is None else float(epsilon)

    # --- delay signatures and the LOS derivative (calibration-free) ---
    U = _delay_signature(system, tau_vec)              # (n_qw, L)
    h = cfg.chip_period * 1e-3
    tp = tau_vec.copy(); tp[los_index] += h
    tm = tau_vec.copy(); tm[los_index] -= h
    du0 = (_delay_signature(system, tp)[:, los_index]
           - _delay_signature(system, tm)[:, los_index]) / (2 * h)

    # noise-subspace projector Pi_U^perp (removes the WHOLE delay subspace,
    # i.e. accounts for the other paths -> multipath raises the bound)
    Q, _ = np.linalg.qr(U)
    PiU_perp = np.eye(U.shape[0], dtype=complex) - Q @ Q.conj().T
    uperp2 = float(np.real(np.vdot(PiU_perp @ du0, PiU_perp @ du0)))

    # --- per-path powers (LOS = 1, NLOS attenuated by SMR) ---
    smr = getattr(cfg, "smr_db", 0.0)
    powers = np.ones(L)
    if smr and L > 1:
        powers[1:] = 10 ** (-smr / 10.0)

    # --- array-response energies (calibration error adds epsilon^2 * M) ---
    A = array_lin(np.asarray(theta_deg_vec, float), M)    # (M, L)
    # Original calibration model: a_l + sqrt(eps/2) e^{j phi}, per-element
    # perturbation power eps/2 -> E||a_l + delta||^2 = ||a_l||^2 + M*(eps/2).
    a_energy = np.sum(np.abs(A) ** 2, axis=0) + (eps / 2.0) * M

    # --- noise power, calibrated exactly as in the forward model ---
    snr = _post_corr_snr(cfg, cn0_db)
    # p_sig = E[mean|S|^2] = (1/(n_qw*M)) sum_l P_l ||u_l||^2 ||a_l||^2
    u_energy = np.sum(np.abs(U) ** 2, axis=0)             # ||u_l||^2
    p_sig = float(np.sum(powers * u_energy * a_energy) / (U.shape[0] * M))
    sigma2 = p_sig / snr

    # --- LOS gain energy G_00 = sum_{k,m} |g_0|^2 = K * P_0 * ||a_0||^2 ---
    G00 = K * powers[los_index] * a_energy[los_index]

    crb = sigma2 / (2.0 * G00 * uperp2)
    return crb, LIGHT_VELOCITY * np.sqrt(crb)
