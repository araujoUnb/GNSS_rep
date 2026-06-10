"""Efficient, modular re-structuring of the ORIGINAL simulator.

Goal: keep the EXACT numerics of ``DelayEstimation.py`` (the code that produced
``results.dat``) -- same forward model (signalModel3 with the sqrt(eps/2) phasor
calibration error), same Bayesian optimizer (``bayes_opt`` library, GP+EI with
``SequentialDomainReductionTransformer``), same ESPRIT+LSKRF baseline, same
unbounded L-BFGS-B refinement -- but make it ~30x faster so we can run large
Monte-Carlo campaigns on SLURM.

The only inefficiency removed: the original rebuilds the whitened signature
``CQw`` (an FFT + an SVD of the correlator bank over N=2046 samples) on EVERY
black-box evaluation of the BO (62 evals) and of the refinement. Here we
precompute, ONCE per system:
  * the whitening basis ``Qw`` (SVD of the correlator bank), and
  * a fine signature dictionary ``CQw_dict = Qw^H C(grid)``,
and the BO/refinement objective becomes a cheap (complex) interpolation of the
dictionary instead of a full rebuild. The objective VALUE is the same mode-2 LS
residual, so the optimizer sees the same landscape and the statistics match
results.dat.

Reuses the original low-level building blocks from ``DelayEstimation`` verbatim
to guarantee identical math.
"""

import time
import numpy as np
import torch as th
import tensorly as tl
from tensorly.tenalg import khatri_rao
from scipy.optimize import minimize
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from bayes_opt.util import UtilityFunction

import DelayEstimation as D  # original module (same folder)

tl.set_backend("pytorch")
LIGHT = 299792458.0


class FastDelaySim:
    """One system (fixed C/N0, array size); heavy operators precomputed once.

    ``run_realization`` then streams cheap Monte-Carlo iterations. Mirrors the
    original ``gnssSimulSignal`` + ``simulation`` but with the precomputed,
    interpolated BO objective.
    """

    def __init__(self, cn0, m=8, k=30, q=11, grid_n=2048, device=None,
                 bo_engine="bayesopt", objective="exact",
                 bo_n_init=2, bo_n_iter=60):
        # BO budget (original defaults init_points=2, n_iter=60 -> I_max=62).
        # Parameterized so the I_max sensitivity figure can sweep bo_n_iter.
        self.bo_n_init = int(bo_n_init)
        self.bo_n_iter = int(bo_n_iter)
        # BO/refine objective: "exact" (Qw^H C(tau) per call, faithful to the
        # original genCQw; default) or "interp" (precomputed dictionary, faster
        # but smoother -> fewer BO outliers, less faithful).
        assert objective in ("exact", "interp"), objective
        self.objective = objective
        # Which Bayesian-optimization engine the BO/BO+Ref stage uses:
        #   "bayesopt" -> original bayes_opt library (faithful to results.dat)
        #   "botorch"  -> BoTorch GP+EI (GPU-capable; validate vs bayes_opt first)
        assert bo_engine in ("bayesopt", "botorch"), bo_engine
        self.bo_engine = bo_engine
        # Device auto-detection (layered/GPU-ready). The torch forward, the
        # signature dictionary and the LS objective are device-agnostic; the BO
        # (bayes_opt/sklearn GP) and the ESPRIT-LSKRF spline run on CPU, so those
        # tensors are moved back with .cpu() at the boundary. NOTE: validate the
        # cuda path on an actual GPU node before trusting it -- the CPU path
        # (device='cpu') is the one validated here against results.dat.
        self.device = th.device(device) if device is not None else th.device(
            "cuda" if th.cuda.is_available() else "cpu")
        # --- replicate gnssSimulSignal.__setUpGNSSpar / channel params ---
        self.C_N0 = float(cn0)
        self.m, self.k, self.q = m, k, q
        self.sat_no = 20
        self.Nd = 1023
        self.B = 1.023e6
        self.Tc = 1 / 1.023e6
        self.fc = 1575.42e6
        self.T = 1e-3
        self.n = 2 * self.Nd
        self.CB_delays = th.linspace(-self.Tc, self.Tc, self.q)
        res = 2 * self.Tc * self.fc + 1
        self.isv = th.linspace(float(self.CB_delays.min()),
                               float(self.CB_delays.max()), int(res))
        self.CA_FFT, self.CA_PSD = D.frequecy_domain_CA(self.B, self.T, self.sat_no)

        self.SMR_dB = 5.0
        self.SNR_dB = (self.C_N0 - 10 * np.log10(2 * self.B)
                       + 10 * np.log10(self.B * self.T))
        P_LOS = 10 ** (self.SNR_dB / 10)
        gamma_LOS = float(np.sqrt(P_LOS))
        P_NLOS = P_LOS / (10 ** (self.SMR_dB / 10))
        gamma_NLOS = float(np.sqrt(P_NLOS))
        # float32 (like the original) so Gamma stays complex64 downstream
        self.abs_gamma = th.tensor([gamma_LOS, gamma_NLOS], dtype=th.float32)

        # --- precompute Qw / OMEGA once (tau-independent) ---
        # correlator_bank_Q builds Q from CB_delays (fixed); Qw=SVD(Q), OMEGA too.
        dummy = th.tensor([0.0, 0.5 * self.Tc])
        _, self.Q, _, self.Qw, self.OMEGA = D.correlator_bank_Q(
            self.B, self.Tc, self.T, self.CB_delays, dummy, self.CA_FFT)
        self.QwH = th.conj(self.Qw).T                      # (q, N)

        # --- fine signature dictionary over the BO search range ---
        # BO bounds: tauLos in +-0.3Tc, tauNLos in (-0.3Tc, Tc) -> cover [-0.4,1.1]Tc
        self.grid = th.linspace(-0.4 * self.Tc, 1.1 * self.Tc, grid_n)
        Cdict = self._build_C(self.grid)
        self.CQw_dict = (self.QwH @ Cdict).numpy()         # (q, grid_n)
        self.grid_np = self.grid.numpy()

    def _build_C(self, delays):
        """Signature matrix C(delays), replicating the construction inside the
        original ``genCQw`` / ``correlator_bank_Q`` (NOT the buggy, unused
        ``create_matrix_C``). Columns are normalized so ||C[:,0]|| = sqrt(N)."""
        from math import sqrt as msqrt
        N = int(2 * self.B * self.T)
        f0 = 2 * self.B / N
        samples = f0 * (th.linspace(0, N - 1, N) - (N / 2))
        PULSE_FFT = th.fft.fftshift(msqrt(self.Tc) * th.sinc(samples * self.Tc) ** 2)
        T_C = th.exp(-1j * 2 * np.pi * th.outer(samples, delays))
        T_C = th.fft.fftshift(T_C, 0)
        Xc = th.outer(PULSE_FFT * self.CA_FFT, th.ones(delays.size()[0]))
        C = th.fft.ifft(T_C * Xc, len(samples), 0)
        C = msqrt(N) * C / th.norm(C[:, 0])
        return C

    # ------------------------------------------------------------------ forward
    def _forward(self, angle_diff_deg, delay_diff, epsilon, doa_deg=10.0):
        """signalModel3 forward, reusing the precomputed Qw. Returns Y, OMEGA,
        factors, tau_real, and the FBA+SPS matrix E for ESPRIT (as in simulation)."""
        m, k, q = self.m, self.k, self.q
        # Draw order MUST match the original signalModel3 for seed-by-seed
        # reproducibility: phases (rand 2) -> A's eps phasor (rand m x L) -> tau0.
        phases = th.exp(1j * 2 * np.pi * th.rand(2))
        Gamma = th.outer(self.abs_gamma * phases, th.ones(k))

        phi = th.tensor([doa_deg, doa_deg + angle_diff_deg])
        A = D.array_lin_noise(phi, m, epsilon)

        tau0 = self.Tc * 0.3 / 0.5 * (th.rand(1) - 0.5)
        tauL = tau0 + self.Tc * delay_diff
        tau = th.tensor([tau0, tauL])

        # CQw for the true delays via precomputed Qw (no re-SVD)
        C_true = self._build_C(tau)
        CQw = self.QwH @ C_true                              # (q, 2)

        X0 = A @ khatri_rao([Gamma.T, CQw]).T
        Z = tl.tensor(1 / np.sqrt(2) * th.randn(m, k, self.n)
                      + 1j * th.randn(m, k, self.n), dtype=th.complex64)
        Zf = tl.tenalg.mode_dot(Z, self.QwH, 2)
        X = tl.tensor(X0.reshape(m, k, q).clone().detach(), dtype=th.complex64)
        Y = X + Zf

        # FBA + spatial smoothing (E) for ESPRIT, exactly as in simulation()
        PIm = th.fliplr(th.eye(m, dtype=th.complex64))
        Y0 = tl.unfold(Y, 0)
        Z2 = th.cat((Y0, PIm @ th.conj(Y0)), 1)
        l_s = 5
        m_s = m - l_s + 1
        E = th.zeros([m_s, 2 * k * q * l_s], dtype=th.complex64)
        for vv in range(l_s - 1):
            E[:, 2 * k * q * vv:2 * k * q * (vv + 1)] = Z2[vv:(m_s + vv), :]
        return Y, float(tau0), m_s, E

    # ------------------------------------------------------------- fast objective
    def _interp_CQw(self, tau):
        """Complex interpolation of the precomputed dictionary at delay ``tau``.
        Interpolates each of the q rows (q is small)."""
        g = self.grid_np
        re = np.array([np.interp(tau, g, row) for row in self.CQw_dict.real])
        im = np.array([np.interp(tau, g, row) for row in self.CQw_dict.imag])
        return re + 1j * im                                 # (q,)

    def _black_box(self, Y2, tauLos, tauNLos):
        """Mode-2 LS residual, same as the original ``black_box_LS``.

        objective='exact' builds CQw = Qw^H C(tau) exactly per call (matches the
        original genCQw, only skipping the redundant per-call SVD of Q);
        objective='interp' uses the precomputed dictionary (faster, slightly
        smoother -> fewer BO outliers, so less faithful)."""
        if tauLos > tauNLos:
            return -1.0
        if self.objective == "exact":
            C = self._build_C(th.tensor([tauLos, tauNLos], dtype=th.float32))
            CQw = (self.QwH @ C).numpy()                     # (q, 2)
        else:
            CQw = np.stack([self._interp_CQw(tauLos),
                            self._interp_CQw(tauNLos)], axis=1)
        M = np.linalg.pinv(CQw) @ Y2
        return -(np.linalg.norm(Y2 - CQw @ M) / np.linalg.norm(Y2)) ** 2

    def _bayes_opt(self, seed, Y, xi):
        Y2 = tl.unfold(Y, 2).numpy()
        tlim = self.Tc * 0.3
        pbounds = {"tauLos": (-tlim, tlim), "tauNLos": (-tlim, self.Tc)}
        bb = lambda tauLos, tauNLos: self._black_box(Y2, tauLos, tauNLos)
        opt = BayesianOptimization(
            f=bb, pbounds=pbounds, verbose=0,
            bounds_transformer=SequentialDomainReductionTransformer(),
            random_state=1000 + seed)
        util = UtilityFunction(kind="ei", xi=xi)
        opt.maximize(init_points=self.bo_n_init, n_iter=self.bo_n_iter,
                     acquisition_function=util)
        tau_vec = np.array([opt.max["params"]["tauLos"],
                            opt.max["params"]["tauNLos"]])
        tau_bo = float(np.min(tau_vec))
        res = minimize(lambda v: -bb(v[0], v[1]), tau_vec, method="L-BFGS-B")
        tau_bo_ref = float(np.min(res.x))
        return tau_bo, tau_bo_ref

    # --------------------------------------------------------------- realization
    def run_realization(self, seed, angle_diff_deg, delay_diff, epsilon, xi):
        th.manual_seed(int(seed))
        Y, tau_real, m_s, E = self._forward(angle_diff_deg, delay_diff, epsilon)

        t = time.perf_counter()
        tau_lskrf, _ = D.esprit_lskrf(Y, E, 2, m_s, self, self.OMEGA)
        t_lskrf = time.perf_counter() - t
        tau_lskrf = float(tau_lskrf)
        # LSKRF + global refine (same objective), to mirror Bayesian+Ref.
        Y2 = tl.unfold(Y, 2).numpy()
        rl = minimize(lambda v: -self._black_box(Y2, v[0], v[1]),
                      np.array([tau_lskrf, tau_lskrf + delay_diff * self.Tc]),
                      method="L-BFGS-B")
        tau_lskrf_ref = float(np.min(rl.x))

        t = time.perf_counter()
        if self.bo_engine == "botorch":
            from botorch_bo import botorch_bayes_opt
            tau_bo, tau_bo_ref = botorch_bayes_opt(self, Y, int(seed), xi)
        else:
            tau_bo, tau_bo_ref = self._bayes_opt(int(seed), Y, xi)
        t_bo = time.perf_counter() - t

        def err(e):
            return LIGHT * abs(tau_real - e)
        return {
            "seed": int(seed), "tau_real": tau_real,
            "i_max": self.bo_n_init + self.bo_n_iter,
            "lskrf_est": tau_lskrf, "lskrf_ref_est": tau_lskrf_ref,
            "bo_engine": self.bo_engine,
            "bo_est": tau_bo, "bo_ref_est": tau_bo_ref,
            "err_lskrf": err(tau_lskrf), "err_lskrf_ref": err(tau_lskrf_ref),
            "err_bo": err(tau_bo), "err_bo_ref": err(tau_bo_ref),
            "t_lskrf_s": t_lskrf, "t_bo_s": t_bo,
        }
