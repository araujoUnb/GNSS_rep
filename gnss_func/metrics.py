"""Aggregation metrics for Monte-Carlo delay-estimation results.

Used to turn a column of per-realization ranging errors (meters) into the
figures of merit reported in the paper / response letter: RMSE, mean/median,
percentiles, and the outlier rate (Reviewer #2, Comment 4).
"""

import numpy as np


def rmse_m(errors):
    e = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(e ** 2)))


def outlier_rate(errors, threshold_m=5.0):
    """Fraction of realizations whose absolute error exceeds ``threshold_m``."""
    e = np.abs(np.asarray(errors, dtype=float))
    return float(np.mean(e > threshold_m))


def summary(errors, threshold_m=5.0):
    e = np.abs(np.asarray(errors, dtype=float))
    return {
        "n": int(e.size),
        "mean_m": float(np.mean(e)),
        "median_m": float(np.median(e)),
        "rmse_m": rmse_m(e),
        "p90_m": float(np.percentile(e, 90)),
        "p95_m": float(np.percentile(e, 95)),
        "outlier_rate": outlier_rate(e, threshold_m),
        "threshold_m": float(threshold_m),
    }
