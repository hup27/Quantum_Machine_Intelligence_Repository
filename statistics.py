"""Paired forecast-comparison procedures for serially dependent loss series."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import zlib

try:
    from arch.bootstrap import MCS
except ImportError:  # pragma: no cover - validated by the reproducible environment
    MCS = None

from .metrics import qlike_loss, squared_log_error


def newey_west_long_run_variance(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    gamma0 = np.dot(x, x) / n
    lrv = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - k / (lag + 1.0)
        gamma = np.dot(x[k:], x[:-k]) / n
        lrv += 2.0 * weight * gamma
    return float(max(lrv, 1e-16))


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 5, horizon: int = 1) -> Dict[str, float]:
    """HLN-corrected DM test for H0: equal expected loss.

    The differential is loss_a - loss_b, so a negative statistic favors model A.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Loss arrays must be aligned")
    d = a - b
    n = len(d)
    lrv = newey_west_long_run_variance(d, lag)
    stat = d.mean() / np.sqrt(lrv / n)
    correction_sq = (n + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n) / n
    stat_hln = stat * np.sqrt(max(correction_sq, 0.0))
    p = 2.0 * stats.t.sf(abs(stat_hln), df=max(n - 1, 1))
    return {
        "mean_loss_difference": float(d.mean()),
        "dm_statistic": float(stat_hln),
        "dm_pvalue": float(p),
        "n_dates": int(n),
        "hac_lag": int(lag),
    }


def moving_block_bootstrap_mean_ci(
    differential: np.ndarray,
    block_length: int = 10,
    repetitions: int = 4000,
    seed: int = 8243,
) -> Dict[str, float]:
    x = np.asarray(differential, dtype=float)
    n = len(x)
    if n < 3:
        return {"bootstrap_mean": float(x.mean()), "ci_low": np.nan, "ci_high": np.nan, "prob_less_zero": np.nan}
    block = min(block_length, n)
    rng = np.random.default_rng(seed)
    circular = np.concatenate([x, x[: block - 1]])
    means = np.empty(repetitions, dtype=float)
    blocks_needed = int(np.ceil(n / block))
    for r in range(repetitions):
        starts = rng.integers(0, n, size=blocks_needed)
        sample = np.concatenate([circular[s : s + block] for s in starts])[:n]
        means[r] = sample.mean()
    return {
        "bootstrap_mean": float(x.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "prob_less_zero": float(np.mean(means < 0.0)),
    }


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average neural forecasts across seeds on the variance scale."""
    test = predictions[predictions["split"] == "test"].copy()
    keys = ["fold", "model", "episode", "target_date"]
    grouped = (
        test.groupby(keys, as_index=False)
        .agg(
            actual_variance=("actual_variance", "first"),
            actual_log_variance=("actual_log_variance", "first"),
            predicted_variance=("predicted_variance", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    grouped["predicted_log_variance"] = np.log(np.maximum(grouped["predicted_variance"], 1e-14))
    return grouped


def forecast_comparison_table(
    predictions: pd.DataFrame,
    comparisons: Sequence[Tuple[str, str]],
    lag: int = 5,
    block_length: int = 10,
    repetitions: int = 4000,
) -> pd.DataFrame:
    ens = ensemble_predictions(predictions)
    rows: List[dict] = []
    for fold in sorted(ens["fold"].unique()):
        fold_data = ens[ens["fold"] == fold]
        for model_a, model_b in comparisons:
            a = fold_data[fold_data["model"] == model_a]
            b = fold_data[fold_data["model"] == model_b]
            merged = a.merge(
                b,
                on=["fold", "episode", "target_date"],
                suffixes=("_a", "_b"),
                validate="one_to_one",
            )
            if merged.empty:
                continue
            for metric in ("qlike", "squared_log_error"):
                if metric == "qlike":
                    loss_a = qlike_loss(merged["actual_variance_a"], merged["predicted_variance_a"])
                    loss_b = qlike_loss(merged["actual_variance_b"], merged["predicted_variance_b"])
                else:
                    loss_a = squared_log_error(merged["actual_log_variance_a"], merged["predicted_log_variance_a"])
                    loss_b = squared_log_error(merged["actual_log_variance_b"], merged["predicted_log_variance_b"])
                dm = dm_test(loss_a, loss_b, lag=lag)
                boot = moving_block_bootstrap_mean_ci(
                    loss_a - loss_b,
                    block_length=block_length,
                    repetitions=repetitions,
                    seed=zlib.crc32(f"{fold}|{model_a}|{model_b}|{metric}".encode("utf-8")),
                )
                rows.append(
                    {
                        "fold": fold,
                        "metric": metric,
                        "model_a": model_a,
                        "model_b": model_b,
                        **dm,
                        **boot,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pvalue_holm"] = np.nan
        for metric, idx in out.groupby("metric").groups.items():
            out.loc[idx, "pvalue_holm"] = holm_adjust(out.loc[idx, "dm_pvalue"].to_numpy())
    return out


def _paired_test_row(
    merged: pd.DataFrame,
    metric: str,
    model_a: str,
    model_b: str,
    scope: str,
    fold: str,
) -> dict:
    d = merged[f"{metric}_a"] - merged[f"{metric}_b"]
    t_stat, t_p = stats.ttest_rel(merged[f"{metric}_a"], merged[f"{metric}_b"])
    try:
        w_stat, w_p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    return {
        "scope": scope,
        "fold": fold,
        "metric": metric,
        "model_a": model_a,
        "model_b": model_b,
        "n_pairs": int(len(merged)),
        "mean_difference": float(d.mean()),
        "median_difference": float(d.median()),
        "standard_deviation_difference": float(d.std(ddof=1)),
        "standard_error_difference": float(d.std(ddof=1) / np.sqrt(len(d))),
        "ci_low_t": float(d.mean() - stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))),
        "ci_high_t": float(d.mean() + stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))),
        "win_rate_model_a": float(np.mean(d < 0.0)),
        "paired_t_statistic": float(t_stat),
        "paired_t_pvalue": float(t_p),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
    }


def paired_seed_table(metrics: pd.DataFrame, comparisons: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    """Paired initialization tests without treating fold-seed rows as independent.

    Tests are reported separately within each fold. A supplementary pooled row first
    averages each model's metric across folds within seed and then pairs the resulting
    seed-level averages, so the pooled sample size remains the number of seeds rather
    than the number of seed-fold combinations.
    """
    test = metrics[(metrics["split"] == "test") & (metrics["seed"] >= 0)].copy()
    rows: List[dict] = []
    for metric in ("qlike", "rmse_log"):
        for model_a, model_b in comparisons:
            a = test[test["model"] == model_a][["fold", "seed", metric]]
            b = test[test["model"] == model_b][["fold", "seed", metric]]
            merged = a.merge(b, on=["fold", "seed"], suffixes=("_a", "_b"), validate="one_to_one")
            for fold, group in merged.groupby("fold", sort=True):
                if len(group) >= 3:
                    rows.append(_paired_test_row(group, metric, model_a, model_b, "within_fold", str(fold)))
            pooled = merged.groupby("seed", as_index=False)[[f"{metric}_a", f"{metric}_b"]].mean()
            if len(pooled) >= 3:
                rows.append(_paired_test_row(pooled, metric, model_a, model_b, "seed_average_across_folds", "all"))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["paired_t_pvalue_holm"] = np.nan
        out["wilcoxon_pvalue_holm"] = np.nan
        for (metric, scope, fold), idx in out.groupby(["metric", "scope", "fold"], dropna=False).groups.items():
            out.loc[idx, "paired_t_pvalue_holm"] = holm_adjust(out.loc[idx, "paired_t_pvalue"].to_numpy())
            valid = out.loc[idx, "wilcoxon_pvalue"].notna()
            valid_idx = out.loc[idx].index[valid]
            if len(valid_idx):
                out.loc[valid_idx, "wilcoxon_pvalue_holm"] = holm_adjust(out.loc[valid_idx, "wilcoxon_pvalue"].to_numpy())
    return out


def model_confidence_set_table(
    predictions: pd.DataFrame,
    *,
    size: float = 0.10,
    block_length: int = 10,
    repetitions: int = 4000,
) -> pd.DataFrame:
    """Construct fold-specific Model Confidence Sets from seed-ensemble losses.

    The implementation delegates the Hansen--Lunde--Nason range statistic to
    ``arch.bootstrap.MCS`` and uses a circular block bootstrap. Smaller losses are
    better. Each returned row contains a model's inclusion status and MCS p-value.
    """
    if MCS is None:
        raise ImportError("The 'arch' package is required for Model Confidence Sets")
    ens = ensemble_predictions(predictions).sort_values(["fold", "target_date", "model"])
    rows: List[dict] = []
    for fold, fold_data in ens.groupby("fold", sort=True):
        actual_var = fold_data.pivot(index="target_date", columns="model", values="actual_variance")
        actual_log = fold_data.pivot(index="target_date", columns="model", values="actual_log_variance")
        pred_var = fold_data.pivot(index="target_date", columns="model", values="predicted_variance")
        pred_log = fold_data.pivot(index="target_date", columns="model", values="predicted_log_variance")
        common_models = sorted(set(pred_var.columns) & set(pred_log.columns))
        for metric in ("qlike", "squared_log_error"):
            if metric == "qlike":
                losses = pd.DataFrame(
                    {m: qlike_loss(actual_var[m].to_numpy(), pred_var[m].to_numpy()) for m in common_models},
                    index=pred_var.index,
                )
            else:
                losses = pd.DataFrame(
                    {m: squared_log_error(actual_log[m].to_numpy(), pred_log[m].to_numpy()) for m in common_models},
                    index=pred_log.index,
                )
            losses = losses.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
            seed = zlib.crc32(f"mcs|{fold}|{metric}".encode("utf-8"))
            procedure = MCS(
                losses,
                size=float(size),
                reps=int(repetitions),
                block_size=min(int(block_length), len(losses)),
                method="R",
                bootstrap="circular",
                seed=seed,
            )
            procedure.compute()
            pvalues = procedure.pvalues["Pvalue"]
            included = set(map(str, procedure.included))
            mean_losses = losses.mean().sort_values()
            ranks = mean_losses.rank(method="min")
            for model in common_models:
                rows.append(
                    {
                        "fold": str(fold),
                        "metric": metric,
                        "model": model,
                        "mean_loss": float(mean_losses[model]),
                        "loss_rank": int(ranks[model]),
                        "included": bool(model in included),
                        "mcs_pvalue": float(pvalues.loc[model]),
                        "confidence_level": float(1.0 - size),
                        "method": "R",
                        "bootstrap": "circular",
                        "block_length": int(min(block_length, len(losses))),
                        "repetitions": int(repetitions),
                        "n_dates": int(len(losses)),
                    }
                )
    return pd.DataFrame(rows)

