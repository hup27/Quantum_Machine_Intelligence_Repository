"""Econometric and simple forecasting baselines aligned to neural windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data import WindowBundle
from .metrics import metric_summary


@dataclass
class BaselineResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    coefficients: pd.DataFrame


def _window_predictors(bundle: WindowBundle, panel: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    episodes = {
        name: group.sort_values("date").reset_index(drop=True)
        for name, group in panel.groupby("episode", sort=False)
    }
    for _, meta in bundle.metadata.iterrows():
        ep = episodes[meta["episode"]]
        i = int(meta["row_index_in_episode"])
        history = ep.loc[:i, "log_parkinson_var"].to_numpy(dtype=float)
        if len(history) < 22:
            raise AssertionError("HAR predictor requested without 22 observations")
        rows.append(
            {
                "fold": meta["fold"],
                "split": meta["split"],
                "episode": meta["episode"],
                "feature_start_date": meta["feature_start_date"],
                "feature_end_date": meta["feature_end_date"],
                "target_date": meta["target_date"],
                "current_log_parkinson_var": float(history[-1]),
                "har_daily": float(history[-1]),
                "har_weekly": float(np.mean(history[-5:])),
                "har_monthly": float(np.mean(history[-22:])),
                "ewma_next_log_var": float(ep.at[i, "ewma_next_log_var"]),
                "garch_next_log_var": float(ep.at[i, "garch_next_log_var"]),
                "egarch_next_log_var": float(ep.at[i, "egarch_next_log_var"]),
                "actual_log_variance": float(bundle.y_raw[len(rows)]),
            }
        )
    return pd.DataFrame(rows)


def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def run_baselines(bundle: WindowBundle, panel: pd.DataFrame, fold: str) -> BaselineResult:
    design = _window_predictors(bundle, panel)
    train = design["split"] == "train"
    y_train = design.loc[train, "actual_log_variance"].to_numpy(dtype=float)

    predictions: Dict[str, np.ndarray] = {}
    coefficient_rows: List[dict] = []

    predictions["historical_mean"] = np.full(len(design), y_train.mean())
    coefficient_rows.append({"fold": fold, "model": "historical_mean", "term": "intercept", "value": y_train.mean()})

    predictions["persistence"] = design["current_log_parkinson_var"].to_numpy(dtype=float)

    for model, column in (
        ("ewma", "ewma_next_log_var"),
        ("garch", "garch_next_log_var"),
        ("egarch", "egarch_next_log_var"),
    ):
        x = design[column].to_numpy(dtype=float)
        X_train = np.column_stack([np.ones(train.sum()), x[train]])
        beta = _ols_fit(X_train, y_train)
        predictions[model] = beta[0] + beta[1] * x
        coefficient_rows.extend(
            [
                {"fold": fold, "model": model, "term": "intercept", "value": beta[0]},
                {"fold": fold, "model": model, "term": column, "value": beta[1]},
            ]
        )

    har_columns = ["har_daily", "har_weekly", "har_monthly"]
    har_x = design[har_columns].to_numpy(dtype=float)
    har_beta = _ols_fit(np.column_stack([np.ones(train.sum()), har_x[train]]), y_train)
    predictions["har"] = np.column_stack([np.ones(len(design)), har_x]) @ har_beta
    for term, value in zip(["intercept"] + har_columns, har_beta):
        coefficient_rows.append({"fold": fold, "model": "har", "term": term, "value": value})

    metric_rows: List[dict] = []
    prediction_frames: List[pd.DataFrame] = []
    for model, pred in predictions.items():
        for split in ("train", "validation", "test"):
            mask = design["split"].to_numpy() == split
            actual = design.loc[mask, "actual_log_variance"].to_numpy(dtype=float)
            predicted = pred[mask]
            metric_rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "seed": -1,
                    "split": split,
                    "best_epoch": np.nan,
                    "best_validation_mse_standardized": np.nan,
                    "runtime_seconds": 0.0,
                    "parameter_count": len([r for r in coefficient_rows if r["model"] == model]),
                    **metric_summary(actual, predicted),
                }
            )
            frame = design.loc[mask, [
                "fold", "split", "episode", "feature_start_date", "feature_end_date", "target_date"
            ]].copy()
            frame["model"] = model
            frame["seed"] = -1
            frame["actual_log_variance"] = actual
            frame["predicted_log_variance"] = predicted
            frame["actual_variance"] = np.exp(np.clip(actual, -30, 5))
            frame["predicted_variance"] = np.exp(np.clip(predicted, -30, 5))
            prediction_frames.append(frame)

    return BaselineResult(
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
        coefficients=pd.DataFrame(coefficient_rows),
    )
