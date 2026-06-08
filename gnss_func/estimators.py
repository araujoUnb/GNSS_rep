import numpy as np
import tensorly as tl
from scipy.signal import find_peaks
from scipy.optimize import minimize, minimize_scalar  # noqa: F401
from bayopt.gaussian import eMM_BSL as MM_BSL
from gnss_func.array import array_lin
from gnss_func.gnss_function import create_matrix_C, build_signal_C
import matplotlib.pyplot as plt


class singlePolModel_estimator:

    def __init__(self, model, delay_granularity, theta_deg_space):
        self.model = model
        self.delay_granularity = delay_granularity
        self.theta_deg_space = theta_deg_space
        self.light_velocity = 299792458

        self.Cbasis = None
        self.delay_Basis = None

        self.tau_los_est = None
        self.beta_tau =None
        self.tau_est =None

        self.angle_basis = None
        self.beta_angle = None

        self.create_tau_space()
        self.create_delay_basis()

        self.create_angle_basis()

    def create_tau_space(self):
        self.tau_space = np.linspace(-1, 1, 2 * self.delay_granularity + 1) * self.model.chip_period


    def system_bandwidth(self):
        return self.model.bandwidth

    def system_ID(self):
        return self.model.ID

    def system_time_period(self):
        return self.model.time_period

    def system_chip_period(self):
        return self.model.chip_period

    def system_Qw(self):
        return self.model.Qw

    def system_Q(self):
        return self.model.Q

    def system_CA_FFT(self):
        return self.model.CA_FFT

    def create_delay_basis(self):
        self.Cbasis = create_matrix_C(self.system_bandwidth(), self.system_chip_period(), self.system_time_period(),
                                      self.tau_space, self.system_CA_FFT())
        if self.model.correlatorType == 'Qw':
            self.delay_Basis = self.Cbasis.T @ self.system_Qw()
        elif self.model.correlatorType == 'Q':
            self.delay_Basis = self.Cbasis.T @ self.system_Q()

    def create_angle_basis(self):
        self.angle_basis = array_lin(self.theta_deg_space, self.model.nAntennas)


    def sparse_delay_estimation(self, a=0, b=0, c=0, d=0):
        B = self.delay_Basis.T
        Y = tl.unfold(self.model.rSignal, mode=1)
        beta, sigma, error = MM_BSL(B, Y, a, b, c, d)
        return beta

    def sparse_delay_angle_estimation(self, a=0, b=0, c=0, d=0):
        B = self.delay_Basis.T
        A = array_lin(self.sparse_angle_estimation(),self.model.nAntennas)
        Y_filtered = tl.tenalg.mode_dot(self.model.rSignal,np.linalg.pinv(A),mode=2)
        Y = tl.unfold(Y_filtered, mode=1)
        beta, sigma, error = MM_BSL(B, Y, a, b, c, d)
        return beta

    def sparse_angle_estimation(self, a=0, b=0, c=0, d=0):
        B = self.angle_basis
        Y = tl.unfold(self.model.rSignal, mode=2)
        R = Y @ np.conj(Y).T
        L, U = np.linalg.eig(R)
        Ru = U[:, 2::] @ np.conj(U[:, 2::]).T
        return 1 / (np.diag(np.conj(self.angle_basis).T @ Ru @ self.angle_basis))

    def theta_estimation(self):
        self.beta_angle = self.sparse_angle_estimation()
        idx_peaks = find_peaks(np.abs(self.beta_angle) ** 2)[0]
        idx_sort = np.flipud(np.argsort(np.abs(self.beta_angle)[idx_peaks]))
        return self.theta_deg_space[idx_peaks[idx_sort[0:2]]]

    def rmse_angle(self, tau_los):
        self.beta_tau = self.sparse_delay_angle_estimation()
        idx_peaks = find_peaks(np.fft.fftshift(np.sum(np.abs(self.beta_tau) ** 2, axis=1)))[0]
        idx_sort = np.flipud(np.argsort(np.fft.fftshift(np.sum(np.abs(self.beta_tau) ** 2, axis=1))[idx_peaks]))
        self.tau_est = np.abs(self.tau_space[idx_peaks[idx_sort]])
        self.tau_los_est = np.abs(self.tau_space[idx_peaks[idx_sort[0]]])

        return self.light_velocity * np.abs(tau_los - self.tau_los_est)

    def rmse(self, tau_los):
        self.beta_tau = self.sparse_delay_estimation()
        idx_peaks = find_peaks(np.fft.fftshift(np.sum(np.abs(self.beta_tau) ** 2, axis=1)))[0]
        idx_sort = np.flipud(np.argsort(np.fft.fftshift(np.sum(np.abs(self.beta_tau) ** 2, axis=1))[idx_peaks]))
        self.tau_est = np.abs(self.tau_space[idx_peaks[idx_sort]])
        self.tau_los_est = np.abs(self.tau_space[idx_peaks[idx_sort[0]]])

        return self.light_velocity * np.abs(tau_los - self.tau_los_est)

    def plot_beta_tau(self):
        self.beta_tau = self.sparse_delay_estimation()
        plt.plot(self.tau_space, np.fft.fftshift(np.sum(np.abs(self.beta_tau) ** 2,axis=1)))
        plt.show()

    def plot_beta_angle(self):
        self.beta_angle = self.sparse_angle_estimation()
        plt.plot(self.theta_deg_space, np.sum(np.abs(self.beta_angle) ** 2,axis=1))
        plt.show()


# ---------------------------------------------------------------------------
# Swappable estimator layer (operates on a precomputed GNSSSystem).
#
# These classes separate the *estimator* from the *forward model*: the delay
# dictionary and angle manifold are built ONCE from the system and reused for
# every received tensor, so swapping the estimation method (BSL, LSKRF, BO+Ref,
# ...) is just instantiating a different subclass. They reproduce exactly the
# numerics of singlePolModel_estimator above.
# ---------------------------------------------------------------------------
class DelayEstimator:
    """Base class: subclasses implement ``estimate(rx) -> tau_los_est``."""

    LIGHT_VELOCITY = 299792458.0

    def __init__(self, system, theta_deg_space=None):
        self.system = system
        cfg = system.cfg
        self.tau_space = np.linspace(
            -1, 1, 2 * cfg.delay_granularity + 1
        ) * cfg.chip_period
        # delay dictionary projected onto the correlator subspace (precomputed)
        Cbasis = create_matrix_C(
            cfg.bandwidth, cfg.chip_period, cfg.time_period,
            self.tau_space, system.CA_FFT,
        )
        self.delay_Basis = Cbasis.T @ system.proj
        if theta_deg_space is None:
            theta_deg_space = np.linspace(35, 75, 100)
        self.theta_deg_space = theta_deg_space
        self.angle_basis = array_lin(theta_deg_space, cfg.n_antennas)
        self.tau_los_est = None

    def estimate(self, rx):
        raise NotImplementedError

    def error_m(self, rx, tau_los_true):
        """Ranging error in meters for one realization."""
        return self.LIGHT_VELOCITY * np.abs(tau_los_true - self.estimate(rx))

    # shared peak-picking on the delay power profile
    def _peak_tau(self, beta):
        power = np.fft.fftshift(np.sum(np.abs(beta) ** 2, axis=1))
        idx_peaks = find_peaks(power)[0]
        idx_sort = np.flipud(np.argsort(power[idx_peaks]))
        self.tau_est = np.abs(self.tau_space[idx_peaks[idx_sort]])
        self.tau_los_est = np.abs(self.tau_space[idx_peaks[idx_sort[0]]])
        return self.tau_los_est


class BSLDelayEstimator(DelayEstimator):
    """Bayesian sparse-learning delay estimator (no angle pre-filtering)."""

    def estimate(self, rx, a=0, b=0, c=0, d=0):
        B = self.delay_Basis.T
        Y = tl.unfold(rx, mode=1)
        beta, _, _ = MM_BSL(B, Y, a, b, c, d)
        return self._peak_tau(beta)


class BSLDelayAngleEstimator(DelayEstimator):
    """BSL delay estimator with a spatial (MUSIC-like) angle pre-filter."""

    def _estimate_angles(self, rx):
        Y = tl.unfold(rx, mode=2)
        R = Y @ np.conj(Y).T
        _, U = np.linalg.eig(R)
        Ru = U[:, 2:] @ np.conj(U[:, 2:]).T
        return 1 / np.diag(np.conj(self.angle_basis).T @ Ru @ self.angle_basis)

    def estimate(self, rx, a=0, b=0, c=0, d=0):
        beta_angle = self._estimate_angles(rx)
        idx_peaks = find_peaks(np.abs(beta_angle) ** 2)[0]
        idx_sort = np.flipud(np.argsort(np.abs(beta_angle)[idx_peaks]))
        theta_est = self.theta_deg_space[idx_peaks[idx_sort[0:2]]]
        A = array_lin(theta_est, self.system.cfg.n_antennas)
        rx_filtered = tl.tenalg.mode_dot(rx, np.linalg.pinv(A), mode=2)
        B = self.delay_Basis.T
        Y = tl.unfold(rx_filtered, mode=1)
        beta, _, _ = MM_BSL(B, Y, a, b, c, d)
        return self._peak_tau(beta)


# ===========================================================================
# Fine delay dictionary + matched-filter delay objective (shared by the
# refinement, LSKRF and BO estimators below). Built ONCE per estimator in the
# compressed correlator subspace, so per-realization cost evaluations are cheap
# (no N-point FFT per call).
# ===========================================================================
class _DelayObjective:
    """Multipath-aware delay objective over a precomputed signature dictionary.

    For a delay VECTOR ``tau_vec`` (one entry per path, L paths), the paper's
    cost is the mode-2 least-squares residual

        f(tau) = || [Y]_(2) - D(tau) ( D(tau)^+ [Y]_(2) ) ||_F^2,

    with ``D(tau) = [ u(tau_0) ... u(tau_{L-1}) ]`` the compressed delay
    signatures. ``u(tau) = proj^H c(tau)`` is interpolated from a fine grid so
    each evaluation is cheap (no N-point FFT per call). Both the Bayesian
    optimisation and its L-BFGS-B refinement minimise this same ``residual``.
    """

    def __init__(self, system, n_grid=512, tau_max_frac=2.0, chunk=16):
        # tau_max_frac=2.0: delays span [0, 2 Tc] because the NLOS delay can be
        # up to tau_los (<=Tc) + Delta_tau (<=Tc) ~ 2 Tc. A [0,Tc] grid clipped
        # the second path and broke the 2-path fit.
        Tc = system.cfg.chip_period
        self.Tc = Tc
        self.tau_grid = np.linspace(0.0, tau_max_frac * Tc, n_grid)
        n_qw = system.proj.shape[1]
        U = np.empty((n_qw, n_grid), dtype=complex)
        for s in range(0, n_grid, chunk):
            taus = self.tau_grid[s:s + chunk]
            Cc = build_signal_C(system.cfg.bandwidth, Tc,
                                system.cfg.time_period, taus, system.CA_FFT)
            # Must match the forward signal signature exactly: simulate() uses
            # CQ = C^T @ proj, i.e. column signature = proj^T c(tau) (NO conjugate
            # on proj). Using proj^H here was a bug that made D(tau) the conjugate
            # of the signal subspace -> residual never vanished -> flat objective.
            U[:, s:s + taus.size] = system.proj.T @ Cc
        self.U = U                                   # raw signatures (LS-scaled)

    def signature(self, tau):
        x = np.interp(tau, self.tau_grid, np.arange(self.tau_grid.size))
        i0 = int(np.clip(np.floor(x), 0, self.tau_grid.size - 2))
        f = x - i0
        return (1 - f) * self.U[:, i0] + f * self.U[:, i0 + 1]

    def matrix_D(self, tau_vec):
        return np.column_stack([self.signature(t) for t in np.atleast_1d(tau_vec)])

    def residual(self, tau_vec, Yd):
        """Mode-2 LS residual f(tau) for a delay vector (lower is better)."""
        D = self.matrix_D(tau_vec)
        M, *_ = np.linalg.lstsq(D, Yd, rcond=None)
        R = Yd - D @ M
        return float(np.real(np.vdot(R, R)))

    # single-delay matched-filter energy (used by the LSKRF column matching)
    def energy(self, tau, Yd):
        u = self.signature(tau)
        u = u / (np.linalg.norm(u) or 1.0)
        return float(np.sum(np.abs(u.conj() @ Yd) ** 2))


def _delay_data(rx):
    """Mode-1 (delay/correlator subspace) unfolding used as the objective data."""
    return tl.unfold(rx, mode=1)


def refine_delays(objective, rx, tau_init_vec):
    """L-BFGS-B refinement of the FULL delay vector, started from ``tau_init_vec``
    (e.g. the BO result), minimising the same mode-2 residual f(tau)."""
    Yd = _delay_data(rx)
    Tc = objective.Tc
    Tmax = float(objective.tau_grid[-1])
    # Optimise in normalised units x = tau/Tc (O(1)): delays are ~1e-7 s, where
    # L-BFGS-B's default finite-difference step (~1e-8) would corrupt the
    # gradient and the refinement would never move (BO==BO+Ref).
    x0 = np.clip(np.atleast_1d(tau_init_vec) / Tc, 0, Tmax / Tc)
    res = minimize(lambda x: objective.residual(np.asarray(x) * Tc, Yd), x0,
                   method="L-BFGS-B", bounds=[(0.0, Tmax / Tc)] * x0.size)
    return np.sort(np.asarray(res.x) * Tc)


class LSKRFDelayEstimator(DelayEstimator):
    """Least-Squares Khatri-Rao Factorization delay estimator (paper baseline).

    Uses the IDEAL (assumed-calibrated) steering ``A_D`` built from the angles:
      1. spatially separate the L paths via the array mode:
         ``W = A_D^+ [Y]_(3)`` (shape L x K*n_qw);
      2. each row of W is a vectorized rank-1 ``gamma_l (x) cqw_l``; reshape to
         (K, n_qw) and take the rank-1 SVD -> delay factor ``cqw_l``;
      3. match ``cqw_l`` to the delay dictionary -> tau_l; LOS = earliest.

    Because the true array is ``A = A_D + eps*A_P``, ``A_D^+`` leaks energy
    across paths, biasing the recovered delays -> the error grows with eps
    (this is the LSKRF breakdown under calibration error reported in the paper).
    The estimator therefore needs the (assumed) angles: pass ``theta_deg_vec``.
    """

    def __init__(self, system, theta_deg_space=None, n_grid=512, n_paths=2):
        super().__init__(system, theta_deg_space)
        self._obj = _DelayObjective(system, n_grid=n_grid)
        self.n_paths = n_paths
        self.delays = None

    def _match(self, cqw):
        scores = np.abs(self._obj.U.conj().T @ cqw).ravel()
        return self._obj.tau_grid[int(np.argmax(scores))]

    def estimate(self, rx, theta_deg_vec=None):
        L = self.n_paths
        M = self.system.cfg.n_antennas
        K = self.system.cfg.n_epochs
        nqw = self._obj.U.shape[0]
        if theta_deg_vec is None:
            raise ValueError("LSKRF needs the (assumed-calibrated) angles "
                             "theta_deg_vec to build the ideal steering A_D.")
        A_D = array_lin(np.asarray(theta_deg_vec), M)        # (M, L) ideal
        Y3 = tl.unfold(rx, mode=2)                           # (M, K*n_qw)
        W = np.linalg.pinv(A_D) @ Y3                         # (L, K*n_qw)
        taus = []
        for ell in range(L):
            Wl = W[ell].reshape(K, nqw)                      # gamma_l (x) cqw_l
            _, _, Vh = np.linalg.svd(Wl, full_matrices=False)
            cqw_l = np.conj(Vh[0])                           # delay factor (n_qw)
            taus.append(self._match(cqw_l))
        self.delays = np.sort(np.abs(taus))
        self.tau_los_est = float(self.delays[0])             # LOS = earliest
        return self.tau_los_est


class BODelayEstimator(DelayEstimator):
    """Bayesian-optimisation delay estimator (paper method). A Gaussian-process
    surrogate (Matern 3/2) with an expected-improvement acquisition searches the
    FULL L-path delay vector over [0, Tc]^L, minimising the mode-2 residual
    f(tau), under a fixed budget ``i_max`` (+ ``n_init`` random starts). The LOS
    estimate is the earliest of the recovered delays; pair with
    :class:`RefinedEstimator` for the L-BFGS-B refinement.
    """

    def __init__(self, system, theta_deg_space=None, n_grid=512,
                 i_max=25, n_init=20, xi=0.1, n_paths=2, n_cand=800, seed=0):
        super().__init__(system, theta_deg_space)
        self._obj = _DelayObjective(system, n_grid=n_grid)
        self.i_max = i_max
        self.n_init = n_init
        self.xi = xi
        self.n_paths = n_paths
        self.n_cand = n_cand
        self.seed = seed
        self.delays = None

    def estimate(self, rx):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel
        from scipy.stats import norm

        Yd = _delay_data(rx)
        Tmax = float(self._obj.tau_grid[-1])
        L = self.n_paths
        rng = np.random.default_rng(self.seed)

        def g(tv):                       # maximise -residual
            return -self._obj.residual(tv, Yd)

        def rand_pts(n):
            return np.sort(rng.uniform(0, Tmax, size=(n, L)), axis=1)

        X = rand_pts(self.n_init)
        y = [g(x) for x in X]
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=self._obj.Tc / 10, nu=1.5)

        for _ in range(max(0, self.i_max - self.n_init)):
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                          alpha=1e-6)
            gp.fit(np.array(X), np.array(y))
            cand = rand_pts(self.n_cand)
            mu, sd = gp.predict(cand, return_std=True)
            best = np.max(y)
            sd = np.maximum(sd, 1e-12)
            z = (mu - best - self.xi) / sd
            ei = (mu - best - self.xi) * norm.cdf(z) + sd * norm.pdf(z)
            x_next = cand[int(np.argmax(ei))]
            X = np.vstack([X, x_next])
            y.append(g(x_next))

        self.delays = np.sort(X[int(np.argmax(y))])
        self.tau_los_est = float(self.delays[0])
        return self.tau_los_est


class RefinedEstimator(DelayEstimator):
    """Wraps a base estimator with the L-BFGS-B refinement stage ("X + Ref.").

    Refines the FULL delay vector returned by the base estimator (started from
    the base result) on the same mode-2 objective; LOS = earliest refined delay.
    Subclass and set ``base_cls`` (or use :func:`make_refined`).
    """
    base_cls = BSLDelayEstimator
    base_kwargs = {}

    def __init__(self, system, theta_deg_space=None, **kwargs):
        super().__init__(system, theta_deg_space)
        merged = {**self.base_kwargs, **kwargs}
        self.base = self.base_cls(system, theta_deg_space, **merged)
        self._obj = getattr(self.base, "_obj", None)
        if self._obj is None:
            self._obj = _DelayObjective(system, n_grid=merged.get("n_grid", 512))

    def estimate(self, rx):
        self.base.estimate(rx)
        tau0 = getattr(self.base, "delays", None)
        if tau0 is None:
            tau0 = np.array([self.base.tau_los_est])
        self.delays = refine_delays(self._obj, rx, tau0)
        self.tau_los_est = float(self.delays[0])
        return self.tau_los_est


def make_refined(base_cls, name=None, **base_kwargs):
    """Factory: build an 'X + Ref.' estimator class from a base estimator."""
    cls = type(name or (base_cls.__name__ + "Refined"),
               (RefinedEstimator,),
               {"base_cls": base_cls, "base_kwargs": base_kwargs})
    return cls


# Ready-to-use refined variants (swap these into ScenarioRunner)
BSLRefined = make_refined(BSLDelayEstimator, "BSLRefined")
LSKRFRefined = make_refined(LSKRFDelayEstimator, "LSKRFRefined")
BORefined = make_refined(BODelayEstimator, "BORefined")


# Name -> class registry, so scenarios / SLURM jobs can select the estimator
# layer by string (fully auditable in the YAML/CSV).
ESTIMATORS = {
    "BSL": BSLDelayEstimator,
    "BSL+Ref": BSLRefined,
    "BSLangle": BSLDelayAngleEstimator,
    "LSKRF": LSKRFDelayEstimator,
    "LSKRF+Ref": LSKRFRefined,
    "BO": BODelayEstimator,
    "BO+Ref": BORefined,
}
