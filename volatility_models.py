"""Causal GARCH/EGARCH filters implemented with Gaussian quasi-likelihood."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

E_ABS_STANDARD_NORMAL = np.sqrt(2.0 / np.pi)
PERCENT_SCALE = 100.0
LOG_PERCENT_VAR_ADJUSTMENT = 2.0 * np.log(PERCENT_SCALE)


@dataclass
class VolatilityFit:
    model: str
    params: Dict[str, float]
    return_mean: float
    success: bool
    objective: float
    message: str
    n_observations: int

    def to_dict(self) -> Dict[str, object]:
        out = asdict(self)
        out.update({f"param_{k}": v for k, v in self.params.items()})
        out.pop("params")
        return out


def _training_return_episodes(panel: pd.DataFrame) -> Tuple[List[np.ndarray], float, int]:
    arrays: List[np.ndarray] = []
    train_rows = panel.loc[panel["target_split"] == "train", ["episode", "return"]]
    values = train_rows["return"].dropna().to_numpy(dtype=float)
    if len(values) < 50:
        raise ValueError("Not enough training returns for volatility model")
    mean = float(np.mean(values))
    for _, group in train_rows.groupby("episode", sort=False):
        r = group["return"].dropna().to_numpy(dtype=float)
        if len(r) >= 5:
            arrays.append((r - mean) * PERCENT_SCALE)
    return arrays, mean, len(values)


def _egarch_nll(theta: np.ndarray, episodes: Sequence[np.ndarray], init_logvar: float) -> float:
    omega, alpha, gamma, beta = theta
    if not (-30.0 < omega < 10.0 and -2.0 < alpha < 2.0 and -2.0 < gamma < 2.0 and 0.0 <= beta < 0.9995):
        return 1e20
    total = 0.0
    for returns in episodes:
        logh = float(init_logvar)
        for r in returns:
            logh = float(np.clip(logh, -20.0, 20.0))
            h = np.exp(logh)
            total += 0.5 * (np.log(2.0 * np.pi) + logh + (r * r) / h)
            z = r / np.sqrt(h)
            logh = omega + beta * logh + alpha * (abs(z) - E_ABS_STANDARD_NORMAL) + gamma * z
            if not np.isfinite(logh) or total > 1e19:
                return 1e20
    return float(total)


def fit_egarch(panel: pd.DataFrame) -> VolatilityFit:
    episodes, mean, nobs = _training_return_episodes(panel)
    all_r = np.concatenate(episodes)
    init_logvar = float(np.log(max(np.var(all_r), 1e-6)))
    starts = [
        np.array([-0.15, 0.15, -0.08, 0.94]),
        np.array([-0.30, 0.10, -0.05, 0.90]),
        np.array([-0.05, 0.20, -0.12, 0.97]),
        np.array([init_logvar * 0.05, 0.08, 0.00, 0.92]),
    ]
    bounds = [(-10.0, 3.0), (-1.5, 1.5), (-1.5, 1.5), (0.0, 0.998)]
    best = None
    for start in starts:
        result = minimize(
            _egarch_nll,
            start,
            args=(episodes, init_logvar),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 600, "ftol": 1e-10, "gtol": 1e-7},
        )
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None
    params = {k: float(v) for k, v in zip(("omega", "alpha", "gamma", "beta"), best.x)}
    return VolatilityFit(
        model="EGARCH(1,1)",
        params=params,
        return_mean=mean,
        success=bool(best.success),
        objective=float(best.fun),
        message=str(best.message),
        n_observations=nobs,
    )


def _garch_nll(theta: np.ndarray, episodes: Sequence[np.ndarray], init_var: float) -> float:
    omega, alpha, beta = theta
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 0.9995:
        return 1e20
    total = 0.0
    for returns in episodes:
        h = float(init_var)
        prev_r = 0.0
        for r in returns:
            h = float(np.clip(h, 1e-10, 1e8))
            total += 0.5 * (np.log(2.0 * np.pi) + np.log(h) + (r * r) / h)
            h = omega + alpha * (r * r) + beta * h
            prev_r = r
            if not np.isfinite(h) or total > 1e19:
                return 1e20
    return float(total)


def fit_garch(panel: pd.DataFrame) -> VolatilityFit:
    episodes, mean, nobs = _training_return_episodes(panel)
    all_r = np.concatenate(episodes)
    init_var = float(max(np.var(all_r), 1e-6))
    starts = [
        np.array([init_var * 0.03, 0.08, 0.90]),
        np.array([init_var * 0.05, 0.12, 0.82]),
        np.array([init_var * 0.01, 0.05, 0.94]),
    ]
    bounds = [(1e-9, max(10.0 * init_var, 1.0)), (0.0, 0.7), (0.0, 0.998)]
    best = None
    for start in starts:
        result = minimize(
            _garch_nll,
            start,
            args=(episodes, init_var),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 600, "ftol": 1e-10, "gtol": 1e-7},
        )
        # Explicitly reject non-stationary points that can appear at a box corner.
        if result.x[1] + result.x[2] >= 0.9995:
            continue
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        raise RuntimeError("GARCH optimization failed for every initialization")
    params = {k: float(v) for k, v in zip(("omega", "alpha", "beta"), best.x)}
    return VolatilityFit(
        model="GARCH(1,1)",
        params=params,
        return_mean=mean,
        success=bool(best.success),
        objective=float(best.fun),
        message=str(best.message),
        n_observations=nobs,
    )


def apply_egarch(panel: pd.DataFrame, fit: VolatilityFit) -> pd.DataFrame:
    p = fit.params
    out = panel.copy()
    logh_current = np.full(len(out), np.nan, dtype=float)
    logh_next = np.full(len(out), np.nan, dtype=float)

    train_r = out.loc[out["target_split"] == "train", "return"].dropna().to_numpy(dtype=float)
    init_logh = float(np.log(max(np.var((train_r - fit.return_mean) * PERCENT_SCALE), 1e-6)))

    for _, idx in out.groupby("episode", sort=False).groups.items():
        indices = np.asarray(list(idx), dtype=int)
        indices = indices[np.argsort(out.loc[indices, "date"].to_numpy())]
        logh = init_logh
        for j in indices:
            logh = float(np.clip(logh, -20.0, 20.0))
            logh_current[j] = logh - LOG_PERCENT_VAR_ADJUSTMENT
            r = out.at[j, "return"]
            if np.isfinite(r):
                r_pct = (float(r) - fit.return_mean) * PERCENT_SCALE
                z = r_pct / np.sqrt(np.exp(logh))
            else:
                z = 0.0
            next_logh = (
                p["omega"]
                + p["beta"] * logh
                + p["alpha"] * (abs(z) - E_ABS_STANDARD_NORMAL)
                + p["gamma"] * z
            )
            next_logh = float(np.clip(next_logh, -20.0, 20.0))
            logh_next[j] = next_logh - LOG_PERCENT_VAR_ADJUSTMENT
            logh = next_logh

    out["egarch_log_var"] = logh_current
    out["egarch_next_log_var"] = logh_next
    return out


def apply_garch(panel: pd.DataFrame, fit: VolatilityFit) -> pd.DataFrame:
    p = fit.params
    out = panel.copy()
    logh_current = np.full(len(out), np.nan, dtype=float)
    logh_next = np.full(len(out), np.nan, dtype=float)

    train_r = out.loc[out["target_split"] == "train", "return"].dropna().to_numpy(dtype=float)
    init_h = float(max(np.var((train_r - fit.return_mean) * PERCENT_SCALE), 1e-6))

    for _, idx in out.groupby("episode", sort=False).groups.items():
        indices = np.asarray(list(idx), dtype=int)
        indices = indices[np.argsort(out.loc[indices, "date"].to_numpy())]
        h = init_h
        for j in indices:
            h = float(np.clip(h, 1e-10, 1e8))
            logh_current[j] = np.log(h) - LOG_PERCENT_VAR_ADJUSTMENT
            r = out.at[j, "return"]
            r_pct = 0.0 if not np.isfinite(r) else (float(r) - fit.return_mean) * PERCENT_SCALE
            next_h = p["omega"] + p["alpha"] * (r_pct * r_pct) + p["beta"] * h
            next_h = float(np.clip(next_h, 1e-10, 1e8))
            logh_next[j] = np.log(next_h) - LOG_PERCENT_VAR_ADJUSTMENT
            h = next_h

    out["garch_log_var"] = logh_current
    out["garch_next_log_var"] = logh_next
    return out


def add_ewma(panel: pd.DataFrame, fold: str, decay: float = 0.94) -> pd.DataFrame:
    out = panel.copy()
    next_logvar = np.full(len(out), np.nan, dtype=float)
    train_r = out.loc[out["target_split"] == "train", "return"].dropna().to_numpy(dtype=float)
    init_var = float(max(np.var(train_r), 1e-12))
    for _, idx in out.groupby("episode", sort=False).groups.items():
        indices = np.asarray(list(idx), dtype=int)
        indices = indices[np.argsort(out.loc[indices, "date"].to_numpy())]
        h = init_var
        for j in indices:
            r = out.at[j, "return"]
            if np.isfinite(r):
                h = decay * h + (1.0 - decay) * float(r) ** 2
            next_logvar[j] = np.log(max(h, 1e-12))
    out["ewma_next_log_var"] = next_logvar
    return out
