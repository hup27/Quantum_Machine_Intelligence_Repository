"""Forecast evaluation metrics on log variance and positive variance scales."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import spearmanr


def log_to_variance(log_variance: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(np.asarray(log_variance, dtype=float), -30.0, 5.0))


def qlike_loss(actual_variance: np.ndarray, predicted_variance: np.ndarray) -> np.ndarray:
    actual = np.maximum(np.asarray(actual_variance, dtype=float), 1e-14)
    predicted = np.maximum(np.asarray(predicted_variance, dtype=float), 1e-14)
    ratio = actual / predicted
    return ratio - np.log(ratio) - 1.0


def squared_log_error(actual_log: np.ndarray, predicted_log: np.ndarray) -> np.ndarray:
    return (np.asarray(actual_log) - np.asarray(predicted_log)) ** 2


def asymmetric_quadratic_loss(actual_log: np.ndarray, predicted_log: np.ndarray, tau: float = 0.75) -> np.ndarray:
    error = np.asarray(actual_log) - np.asarray(predicted_log)
    weights = np.where(error >= 0.0, tau, 1.0 - tau)
    return weights * error**2


def metric_summary(actual_log: np.ndarray, predicted_log: np.ndarray) -> Dict[str, float]:
    actual_log = np.asarray(actual_log, dtype=float)
    predicted_log = np.asarray(predicted_log, dtype=float)
    actual_var = log_to_variance(actual_log)
    predicted_var = log_to_variance(predicted_log)
    error_log = predicted_log - actual_log
    error_var = predicted_var - actual_var
    # Avoid SciPy's ConstantInputWarning for deterministic constant baselines.
    if np.ptp(actual_log) == 0.0 or np.ptp(predicted_log) == 0.0:
        corr = np.nan
    else:
        corr = spearmanr(actual_log, predicted_log, nan_policy="omit").statistic
        if not np.isfinite(corr):
            corr = np.nan
    ss_res = float(np.sum(error_log**2))
    ss_tot = float(np.sum((actual_log - actual_log.mean()) ** 2))
    return {
        "n": int(len(actual_log)),
        "rmse_log": float(np.sqrt(np.mean(error_log**2))),
        "mae_log": float(np.mean(np.abs(error_log))),
        "bias_log": float(np.mean(error_log)),
        "r2_log": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "spearman_log": float(corr),
        "rmse_variance": float(np.sqrt(np.mean(error_var**2))),
        "mae_variance": float(np.mean(np.abs(error_var))),
        "qlike": float(np.mean(qlike_loss(actual_var, predicted_var))),
        "asym_q75": float(np.mean(asymmetric_quadratic_loss(actual_log, predicted_log, tau=0.75))),
    }
