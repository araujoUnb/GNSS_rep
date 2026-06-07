"""GNSS forward model with precomputed (delay-independent) operators.

This is the "model layer". The heavy operators that do NOT depend on the random
per-realization parameters -- the correlator bank ``Q``, its whitened basis
``Qw``, the transmit power and the noise colouring ``Lnoise`` -- are built ONCE
in the constructor and reused for every Monte-Carlo realization produced by
:meth:`GNSSSystem.simulate`.

On a SLURM array task this means the expensive O(N) FFT/SVD work is paid once per
process and the per-realization cost collapses to a single small N x L delay
replica plus tiny operations in the compressed correlator space.

It is numerically equivalent to the legacy ``single_polarization`` model: it
reuses the exact same low-level builders (``build_correlator_bank`` /
``build_signal_C`` factored out of ``correlator_bank_Q``) and the same Qw, CQ
and noise definitions.
"""

import numpy as np
import tensorly as tl

from gnss_func.config import SystemConfig
from gnss_func.gnss_function import (
    frequecy_domain_CA,
    build_correlator_bank,
    build_signal_C,
)
from gnss_func.array import array_lin
from gnss_func.utils import normalise_columns


class GNSSSystem:

    def __init__(self, cfg: SystemConfig, correlator_type="Qw", n_qw=7):
        self.cfg = cfg
        self.correlator_type = correlator_type
        self.n_qw = n_qw

        B, Tc, T = cfg.bandwidth, cfg.chip_period, cfg.time_period

        # C/A code spectrum (cached on disk by frequecy_domain_CA)
        self.CA_FFT, self.CA_PSD = frequecy_domain_CA(B, T, cfg.sat_id)

        # --- precompute the delay-independent operators ONCE ---
        self.Q, self.tx_power = build_correlator_bank(
            B, Tc, T, self.bank_delay(), self.CA_FFT
        )

        if correlator_type == "Qw":
            U, _, _ = np.linalg.svd(self.Q, full_matrices=False)
            self.Qw = normalise_columns(U[:, :n_qw])
            self.proj = self.Qw
        else:
            self.Qw = None
            self.proj = self.Q

        Rnoise = np.conj(self.proj.T) @ self.proj
        self.Lnoise = np.linalg.cholesky(Rnoise)

    # ------------------------------------------------------------------ helpers
    def bank_delay(self):
        Tc = self.cfg.chip_period
        return np.linspace(-Tc, Tc, 2 * self.cfg.delay_granularity)

    def calc_snr_pre(self, cn0_db):
        return cn0_db - 10 * np.log10(2 * self.cfg.bandwidth)

    def noise_var(self, cn0_db):
        snr_db = self.calc_snr_pre(cn0_db) + 10 * np.log10(
            self.cfg.bandwidth * self.cfg.time_period
        )
        return self.tx_power / 10 ** (snr_db / 10)

    def delay_signature(self, tau_vec):
        """Compressed delay signature CQ = C(tau)^T @ proj for given delays."""
        C = build_signal_C(
            self.cfg.bandwidth, self.cfg.chip_period, self.cfg.time_period,
            np.asarray(tau_vec), self.CA_FFT,
        )
        return C.T @ self.proj

    # -------------------------------------------------------------- realization
    def simulate(self, tau_vec, theta_deg_vec, cn0_db=None, rng=None):
        """Generate one received (compressed) tensor for the given geometry.

        Mirrors ``single_polarization.rx_signal`` but reuses the precomputed
        operators. ``rng`` is a ``numpy.random.Generator`` (defaults to the
        legacy global RNG for backwards-compatible behaviour).
        """
        if cn0_db is None:
            cn0_db = self.cfg.cn0_db
        randn = np.random.randn if rng is None else (
            lambda *s: rng.standard_normal(s)
        )

        tau_vec = np.asarray(tau_vec)
        L = tau_vec.size
        n_epochs = self.cfg.n_epochs
        M = self.cfg.n_antennas

        CQ = self.delay_signature(tau_vec)              # (n_qw, L)
        A = array_lin(np.asarray(theta_deg_vec), M)     # (M, L)

        # Array calibration error: A = A_D + epsilon * A_P, A_P ~ CN(0,1).
        # epsilon = 0 leaves the perfectly-calibrated model untouched.
        eps = getattr(self.cfg, "epsilon", 0.0)
        if eps:
            Ap = (randn(M, L) + 1j * randn(M, L)) / np.sqrt(2)
            A = A + eps * Ap

        taps = 1 / np.sqrt(2) * (randn(n_epochs, L) + 1j * randn(n_epochs, L))
        S0 = taps @ tl.tenalg.khatri_rao([CQ.T, A]).T
        S = tl.tensor(S0.reshape(n_epochs, int(CQ.size / L), M))

        sigma = self.noise_var(cn0_db)
        a, b, c = S.shape
        N2 = 1 / np.sqrt(2 * sigma) * (randn(b, a * c) + 1j * randn(b, a * c))
        N2 = self.Lnoise @ N2
        noise = tl.tensor(N2.reshape(a, b, c))

        return S + noise
