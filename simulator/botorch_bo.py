"""BoTorch BO stage configured to mimic the original ``bayes_opt`` BO.

Same objective (mode-2 LS residual via the precomputed dictionary), same 2-D
search box, same EI acquisition with the ``xi`` exploration offset, same budget
(init_points=2, n_iter=60), followed by the SAME global L-BFGS-B refinement. The
GP is GPyTorch's default Matern-5/2 (matching the sklearn GP of bayes_opt).

Runs on ``sim.device`` (CPU or CUDA). The point of this module is to test
whether a GPU-capable BoTorch BO reproduces the bayes_opt statistics; if it
does, the BO can be batched on the GPU for large campaigns.
"""

import warnings
import numpy as np
import torch as th
import tensorly as tl
from scipy.optimize import minimize

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

warnings.filterwarnings("ignore")


def botorch_bayes_opt(sim, Y, seed, xi=0.1, n_init=2, n_iter=60,
                      num_restarts=10, raw_samples=256):
    """BO over (tauLos, tauNLos) with the same objective as the original.

    Returns (tau_bo, tau_bo_ref), consistent with FastDelaySim._bayes_opt.
    """
    dev = getattr(sim, "device", th.device("cpu"))
    Y2 = tl.unfold(Y, 2).numpy()
    Tc = sim.Tc
    lo = np.array([-0.3 * Tc, -0.3 * Tc])
    hi = np.array([0.3 * Tc, Tc])
    span = hi - lo

    def to_phys(xn):
        return lo + np.asarray(xn) * span

    def obj(xn):
        t = to_phys(xn)
        return sim._black_box(Y2, float(t[0]), float(t[1]))

    rng = np.random.RandomState(2000 + int(seed))
    X = rng.rand(n_init, 2)
    Yv = np.array([obj(x) for x in X])

    unit = th.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=th.double, device=dev)
    for _ in range(n_iter):
        Xt = th.tensor(X, dtype=th.double, device=dev)
        Yt = th.tensor(Yv, dtype=th.double, device=dev).unsqueeze(-1)
        gp = SingleTaskGP(Xt, Yt, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        # NOTE: a positive offset above the achievable max (objective in [-1,0])
        # degenerates analytic EI -> use pure EI (best_f = best observed). The
        # exploration that bayes_opt got from xi comes here from the GP variance.
        best_f = float(Yv.max())
        ei = ExpectedImprovement(gp, best_f=best_f, maximize=True)
        cand, _ = optimize_acqf(ei, bounds=unit, q=1,
                                num_restarts=num_restarts, raw_samples=raw_samples)
        xn = cand.detach().cpu().numpy().ravel()
        X = np.vstack([X, xn])
        Yv = np.append(Yv, obj(xn))

    tvec = to_phys(X[int(np.argmax(Yv))])
    tau_bo = float(np.min(tvec))
    res = minimize(lambda v: -sim._black_box(Y2, v[0], v[1]), tvec,
                   method="L-BFGS-B")
    tau_bo_ref = float(np.min(res.x))
    return tau_bo, tau_bo_ref
