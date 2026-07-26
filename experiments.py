"""End-to-end experiment orchestration."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import json
import logging
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import yaml

from .baselines import run_baselines
from .data import FOLDS, build_windows, load_market_panel, mark_target_splits, save_clean_panel, scale_windows
from .metrics import metric_summary
from .models import (
    PRIMARY_MODEL_SPECS,
    ModelSpec,
    build_model,
    configured_primary_model_spec,
    count_parameters,
    quantum_transform_count_per_timestep,
)
from .statistics import (
    ensemble_predictions,
    forecast_comparison_table,
    model_confidence_set_table,
    paired_seed_table,
)
from .training import TrainResult, TrainingConfig, predict_with_noise, train_one
from .volatility_models import add_ewma, apply_egarch, apply_garch, fit_egarch, fit_garch

LOGGER = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_dirs(root: Path) -> None:
    for rel in (
        "data/processed",
        "results/predictions",
        "results/metrics",
        "results/statistics",
        "results/ablations",
        "results/ablations/runs",
        "results/ablations/chunks",
        "results/histories",
        "models",
        "logs",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)


def _atomic_to_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
    date_format: str | None = None,
) -> None:
    """Write a CSV atomically so interrupted runs never leave partial artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=index, date_format=date_format)
    os.replace(temporary, path)


def _run_paths(
    root: Path,
    fold: str,
    model_name: str,
    seed: int,
    *,
    ablation: bool = False,
) -> Dict[str, Path]:
    """Return the canonical checkpoint/CSV paths for one trained run."""
    stem = f"{fold}__{model_name}__seed{seed}"
    if ablation:
        run_dir = root / "results/ablations/runs"
        return {
            "state": root / f"models/ablation__{stem}.pt",
            "metrics": run_dir / f"{stem}__metrics.csv",
            "predictions": run_dir / f"{stem}__predictions.csv",
            "history": run_dir / f"{stem}__history.csv",
        }
    return {
        "state": root / f"models/{stem}.pt",
        "metrics": root / f"results/metrics/{stem}.csv",
        "predictions": root / f"results/predictions/{stem}.csv",
        "history": root / f"results/histories/{stem}.csv",
    }


def _state_matches_run(
    payload: object,
    spec: ModelSpec,
    scaled: object,
    training: TrainingConfig,
    fold: str,
    seed: int,
) -> bool:
    """Check that a checkpoint belongs to the exact requested configuration."""
    if not isinstance(payload, dict) or "state_dict" not in payload:
        return False
    expected_spec = asdict(spec)
    observed_spec = payload.get("spec")
    if not isinstance(observed_spec, dict):
        return False
    # Older exploratory checkpoints did not record the rotation sequence and
    # must never be mixed with the all-active primary circuit.
    if observed_spec != expected_spec:
        return False
    expected = {
        "input_dim": int(scaled.X["train"].shape[-1]),
        "hidden_dim": int(training.hidden_dim),
        "recurrent_layers": int(training.recurrent_layers),
        "dropout": float(training.dropout),
        "seed": int(seed),
        "fold": str(fold),
    }
    for key, value in expected.items():
        observed = payload.get(key)
        if isinstance(value, float):
            if observed is None or not np.isclose(float(observed), value, rtol=0.0, atol=1e-12):
                return False
        elif observed != value:
            return False
    best_epoch = payload.get("best_epoch")
    if best_epoch is None or int(best_epoch) <= 0:
        return False
    try:
        model = build_model(
            spec,
            input_dim=expected["input_dim"],
            hidden_dim=expected["hidden_dim"],
            recurrent_layers=expected["recurrent_layers"],
            dropout=expected["dropout"],
        )
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception:
        return False
    return True


def _load_completed_run(
    paths: Mapping[str, Path],
    spec: ModelSpec,
    scaled: object,
    training: TrainingConfig,
    fold: str,
    seed: int,
) -> TrainResult | None:
    """Load a valid completed run, or return ``None`` so it can be retrained."""
    if any(not path.exists() or path.stat().st_size == 0 for path in paths.values()):
        return None
    try:
        payload = torch.load(paths["state"], map_location="cpu", weights_only=False)
        if not _state_matches_run(payload, spec, scaled, training, fold, seed):
            return None
        model = build_model(
            spec,
            input_dim=int(scaled.X["train"].shape[-1]),
            hidden_dim=int(training.hidden_dim),
            recurrent_layers=int(training.recurrent_layers),
            dropout=float(training.dropout),
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        metrics = pd.read_csv(paths["metrics"])
        predictions = pd.read_csv(
            paths["predictions"],
            parse_dates=["feature_start_date", "feature_end_date", "target_date"],
        )
        history = pd.read_csv(paths["history"])
        if metrics.empty or predictions.empty or history.empty:
            return None
        if set(metrics["split"].astype(str)) != {"train", "validation", "test"}:
            return None
        if set(predictions["split"].astype(str)) != {"train", "validation", "test"}:
            return None
        if not (predictions["feature_end_date"] < predictions["target_date"]).all():
            return None
        best_epoch = int(payload["best_epoch"])
        if best_epoch not in set(pd.to_numeric(history["epoch"], errors="raise").astype(int)):
            return None
        first_metric = metrics.iloc[0]
        return TrainResult(
            model=model,
            history=history,
            metrics=metrics,
            predictions=predictions,
            best_epoch=best_epoch,
            best_validation_mse=float(
                first_metric.get("best_validation_mse_standardized", np.nan)
            ),
            runtime_seconds=float(first_metric.get("runtime_seconds", np.nan)),
            parameter_count=int(first_metric.get("parameter_count", count_parameters(model))),
        )
    except Exception as exc:
        LOGGER.warning(
            "Ignoring invalid completed run fold=%s model=%s seed=%s: %s",
            fold,
            spec.name,
            seed,
            exc,
        )
        return None


def _write_run_artifacts(result: TrainResult, paths: Mapping[str, Path]) -> None:
    """Persist the three tabular artifacts for a run whose checkpoint is saved."""
    if not paths["state"].exists():
        raise FileNotFoundError(f"Checkpoint was not written: {paths['state']}")
    _atomic_to_csv(result.metrics, paths["metrics"], index=False)
    _atomic_to_csv(
        result.predictions,
        paths["predictions"],
        index=False,
        date_format="%Y-%m-%d",
    )
    _atomic_to_csv(result.history, paths["history"], index=False)


def _training_config(config: dict) -> TrainingConfig:
    t = config["training"]
    return TrainingConfig(
        hidden_dim=int(t["hidden_dim"]),
        recurrent_layers=int(t["recurrent_layers"]),
        dropout=float(t["dropout"]),
        batch_size=int(t["batch_size"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        max_epochs=int(t["max_epochs"]),
        patience=int(t["patience"]),
        min_delta=float(t["min_delta"]),
        gradient_clip=float(t["gradient_clip"]),
    )


def _save_environment(root: Path) -> None:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
    }
    (root / "results/environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")


def _ensemble_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    ens = ensemble_predictions(predictions)
    rows: List[dict] = []
    for (fold, model), group in ens.groupby(["fold", "model"]):
        rows.append(
            {
                "fold": fold,
                "model": model,
                "split": "test",
                "aggregation": "variance_scale_seed_ensemble",
                "n_seeds": int(group["n_seeds"].max()),
                **metric_summary(
                    group["actual_log_variance"].to_numpy(),
                    group["predicted_log_variance"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(rows)


def _metric_summary_across_seeds(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "rmse_log",
        "mae_log",
        "bias_log",
        "r2_log",
        "spearman_log",
        "rmse_variance",
        "mae_variance",
        "qlike",
        "asym_q75",
        "best_epoch",
        "runtime_seconds",
        "parameter_count",
    ]
    rows: List[dict] = []
    for keys, group in metrics.groupby(["fold", "model", "split"], dropna=False):
        row = {"fold": keys[0], "model": keys[1], "split": keys[2], "n_runs": len(group)}
        for col in numeric:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _noise_sensitivity_rows(
    result,
    scaled,
    spec: ModelSpec,
    fold: str,
    seed: int,
    config: TrainingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = [
        ("exact", None, 0.0),
        ("shots_128", 128, 0.0),
        ("shots_512", 512, 0.0),
        ("readout_1pct", None, 0.01),
        ("readout_3pct", None, 0.03),
        ("shots_256_readout_1pct", 256, 0.01),
    ]
    metric_rows = []
    pred_frames = []
    for name, shots, readout_error in scenarios:
        pred_log = predict_with_noise(
            result.model,
            scaled.X["test"],
            scaled.target_scaler,
            config.batch_size,
            shots=shots,
            readout_error=readout_error,
            noise_seed=seed * 1009 + (shots or 0) + int(readout_error * 10000),
        )
        metric_rows.append(
            {
                "fold": fold,
                "model": spec.name,
                "seed": seed,
                "scenario": name,
                "shots": shots if shots is not None else -1,
                "readout_error": readout_error,
                **metric_summary(scaled.raw_y["test"], pred_log),
            }
        )
        frame = scaled.metadata["test"].copy()
        frame["model"] = spec.name
        frame["seed"] = seed
        frame["scenario"] = name
        frame["actual_log_variance"] = scaled.raw_y["test"]
        frame["predicted_log_variance"] = pred_log
        frame["actual_variance"] = np.exp(np.clip(scaled.raw_y["test"], -30, 5))
        frame["predicted_variance"] = np.exp(np.clip(pred_log, -30, 5))
        pred_frames.append(frame)
    return pd.DataFrame(metric_rows), pd.concat(pred_frames, ignore_index=True)


def run_all(root: Path, config_path: Path, quick: bool = False) -> None:
    _ensure_dirs(root)
    config = load_config(config_path)
    _save_environment(root)
    training_config = _training_config(config)
    seeds = [int(s) for s in config["project"]["random_seeds"]]
    model_names = list(config["models"]["primary"])
    if quick:
        seeds = seeds[:1]
        model_names = ["ligru", "fourier_ligru", "quantum_ligru", "egarch_quantum_ligru"]
        training_config.max_epochs = min(training_config.max_epochs, 8)
        training_config.patience = min(training_config.patience, 3)

    raw_panel = load_market_panel(
        root / "data/raw",
        epsilon=float(config["data"]["target_epsilon"]),
    )
    save_clean_panel(raw_panel, root / "data/processed/market_panel_clean.csv")

    all_metrics: List[pd.DataFrame] = []
    all_predictions: List[pd.DataFrame] = []
    all_histories: List[pd.DataFrame] = []
    all_coefficients: List[pd.DataFrame] = []
    fit_rows: List[dict] = []
    resource_rows: List[dict] = []
    noise_metrics: List[pd.DataFrame] = []
    noise_predictions: List[pd.DataFrame] = []
    fold_cache: Dict[str, Tuple[pd.DataFrame, object, object]] = {}

    for fold_definition in FOLDS:
        fold = fold_definition.name
        LOGGER.info("Preparing fold %s: %s", fold, fold_definition.description)
        panel = mark_target_splits(raw_panel, fold)
        egarch_fit = fit_egarch(panel)
        garch_fit = fit_garch(panel)
        fit_rows.extend([
            {"fold": fold, **egarch_fit.to_dict()},
            {"fold": fold, **garch_fit.to_dict()},
        ])
        panel = apply_egarch(panel, egarch_fit)
        panel = apply_garch(panel, garch_fit)
        panel = add_ewma(panel, fold)
        panel.to_csv(root / f"data/processed/panel_{fold}.csv", index=False, date_format="%Y-%m-%d")

        base_features = list(config["data"]["feature_columns"])
        hybrid_features = base_features + [config["data"]["hybrid_feature"]]
        base_bundle = build_windows(
            panel,
            fold,
            base_features,
            int(config["data"]["sequence_length"]),
        )
        hybrid_bundle = build_windows(
            panel,
            fold,
            hybrid_features,
            int(config["data"]["sequence_length"]),
        )
        base_scaled = scale_windows(base_bundle)
        hybrid_scaled = scale_windows(hybrid_bundle)
        fold_cache[fold] = (panel, base_scaled, hybrid_scaled)
        base_bundle.metadata.to_csv(
            root / f"data/processed/window_index_{fold}.csv", index=False, date_format="%Y-%m-%d"
        )

        baseline = run_baselines(base_bundle, panel, fold)
        all_metrics.append(baseline.metrics)
        all_predictions.append(baseline.predictions)
        all_coefficients.append(baseline.coefficients)

        for model_name in model_names:
            spec = configured_primary_model_spec(model_name, config.get("quantum", {}))
            scaled = hybrid_scaled if spec.uses_egarch else base_scaled
            for seed in seeds:
                LOGGER.info("Training fold=%s model=%s seed=%s", fold, model_name, seed)
                state_path = root / f"models/{fold}__{model_name}__seed{seed}.pt"
                result = train_one(
                    spec,
                    scaled.X,
                    scaled.y,
                    scaled.raw_y,
                    scaled.metadata,
                    scaled.target_scaler,
                    training_config,
                    seed,
                    fold,
                    save_state_path=state_path,
                )
                all_metrics.append(result.metrics)
                all_predictions.append(result.predictions)
                all_histories.append(result.history)
                result.metrics.to_csv(
                    root / f"results/metrics/{fold}__{model_name}__seed{seed}.csv", index=False
                )
                result.predictions.to_csv(
                    root / f"results/predictions/{fold}__{model_name}__seed{seed}.csv",
                    index=False,
                    date_format="%Y-%m-%d",
                )
                result.history.to_csv(
                    root / f"results/histories/{fold}__{model_name}__seed{seed}.csv", index=False
                )

                if fold == config["ablations"]["fold"] and spec.transform == "quantum" and seed in config["ablations"]["seeds"]:
                    nm, npred = _noise_sensitivity_rows(
                        result, scaled, spec, fold, seed, training_config
                    )
                    noise_metrics.append(nm)
                    noise_predictions.append(npred)

            # Resources depend on input dimensionality, but not seed.
            probe = build_model(
                spec,
                input_dim=scaled.X["train"].shape[-1],
                hidden_dim=training_config.hidden_dim,
                recurrent_layers=training_config.recurrent_layers,
                dropout=training_config.dropout,
            )
            resource_rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "input_dim": scaled.X["train"].shape[-1],
                    "trainable_parameters": count_parameters(probe),
                    "quantum_transforms_per_timestep": quantum_transform_count_per_timestep(
                        spec, training_config.recurrent_layers
                    ),
                    "qubits": spec.n_qubits if spec.transform == "quantum" else 0,
                    "circuit_layers": spec.circuit_layers if spec.transform == "quantum" else 0,
                    "statevector_dimension": 2 ** spec.n_qubits if spec.transform == "quantum" else 0,
                }
            )

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    histories = pd.concat(all_histories, ignore_index=True) if all_histories else pd.DataFrame()
    coefficients = pd.concat(all_coefficients, ignore_index=True)

    metrics.to_csv(root / "results/metrics/all_seed_metrics.csv", index=False)
    predictions.to_csv(
        root / "results/predictions/all_predictions.csv", index=False, date_format="%Y-%m-%d"
    )
    histories.to_csv(root / "results/histories/all_histories.csv", index=False)
    coefficients.to_csv(root / "results/metrics/baseline_coefficients.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(root / "results/metrics/volatility_filter_fits.csv", index=False)
    pd.DataFrame(resource_rows).drop_duplicates().to_csv(
        root / "results/metrics/model_resources.csv", index=False
    )

    summary = _metric_summary_across_seeds(metrics)
    summary.to_csv(root / "results/metrics/summary_across_seeds.csv", index=False)
    ensemble_metrics = _ensemble_metric_table(predictions)
    ensemble_metrics.to_csv(root / "results/metrics/ensemble_test_metrics.csv", index=False)

    comparisons = [
        ("quantum_ligru", "ligru"),
        ("quantum_ligru", "fourier_ligru"),
        ("egarch_quantum_ligru", "egarch_ligru"),
        ("egarch_quantum_ligru", "egarch_fourier_ligru"),
    ]
    stats_cfg = config["statistics"]
    comparison = forecast_comparison_table(
        predictions,
        comparisons,
        lag=int(stats_cfg["dm_lag"]),
        block_length=int(stats_cfg["block_length"]),
        repetitions=int(stats_cfg["bootstrap_repetitions"]),
    )
    comparison.to_csv(root / "results/statistics/forecast_comparisons.csv", index=False)
    paired = paired_seed_table(metrics, comparisons)
    paired.to_csv(root / "results/statistics/paired_seed_comparisons.csv", index=False)
    mcs = model_confidence_set_table(
        predictions,
        size=float(stats_cfg.get("mcs_size", 0.10)),
        block_length=int(stats_cfg["block_length"]),
        repetitions=int(stats_cfg["bootstrap_repetitions"]),
    )
    mcs.to_csv(root / "results/statistics/model_confidence_sets.csv", index=False)

    if noise_metrics:
        pd.concat(noise_metrics, ignore_index=True).to_csv(
            root / "results/ablations/measurement_sensitivity_metrics.csv", index=False
        )
        pd.concat(noise_predictions, ignore_index=True).to_csv(
            root / "results/ablations/measurement_sensitivity_predictions.csv",
            index=False,
            date_format="%Y-%m-%d",
        )

    if not quick:
        run_architecture_ablations(root, config, training_config, fold_cache, metrics, predictions)

    LOGGER.info("Experiment run completed")


def run_architecture_ablations(
    root: Path,
    config: dict,
    training_config: TrainingConfig,
    fold_cache: Mapping[str, Tuple[pd.DataFrame, object, object]],
    main_metrics: pd.DataFrame,
    main_predictions: pd.DataFrame,
) -> None:
    fold = str(config["ablations"]["fold"])
    seeds = [int(s) for s in config["ablations"]["seeds"]]
    _, base_scaled, hybrid_scaled = fold_cache[fold]
    ablation_specs = [
        ModelSpec("quantum_ligru_no_entanglement", False, "custom", "quantum", 2, 1, False),
        ModelSpec("quantum_ligru_depth2", False, "custom", "quantum", 2, 2, True),
        ModelSpec("quantum_ligru_4qubit", False, "custom", "quantum", 4, 1, True),
        ModelSpec("quantum_gru", False, "quantum_gru", "quantum", 2, 1, True),
        ModelSpec("egarch_quantum_ligru_no_entanglement", True, "custom", "quantum", 2, 1, False),
    ]
    metric_frames = []
    prediction_frames = []
    resource_rows = []
    for spec in ablation_specs:
        scaled = hybrid_scaled if spec.uses_egarch else base_scaled
        for seed in seeds:
            LOGGER.info("Ablation fold=%s model=%s seed=%s", fold, spec.name, seed)
            result = train_one(
                spec,
                scaled.X,
                scaled.y,
                scaled.raw_y,
                scaled.metadata,
                scaled.target_scaler,
                training_config,
                seed,
                fold,
                save_state_path=root / f"models/ablation__{fold}__{spec.name}__seed{seed}.pt",
            )
            metric_frames.append(result.metrics)
            prediction_frames.append(result.predictions)
        probe = build_model(
            spec,
            input_dim=scaled.X["train"].shape[-1],
            hidden_dim=training_config.hidden_dim,
            recurrent_layers=training_config.recurrent_layers,
            dropout=training_config.dropout,
        )
        resource_rows.append(
            {
                "model": spec.name,
                "trainable_parameters": count_parameters(probe),
                "quantum_transforms_per_timestep": quantum_transform_count_per_timestep(
                    spec, training_config.recurrent_layers
                ),
                "qubits": spec.n_qubits,
                "circuit_layers": spec.circuit_layers,
                "entangle": spec.entangle,
                "statevector_dimension": 2 ** spec.n_qubits,
            }
        )

    ablation_metrics = pd.concat(metric_frames, ignore_index=True)
    ablation_predictions = pd.concat(prediction_frames, ignore_index=True)
    # Add matched main variants for direct tables.
    main_keep = main_metrics[
        (main_metrics["fold"] == fold)
        & (main_metrics["seed"].isin(seeds))
        & (main_metrics["model"].isin(["quantum_ligru", "egarch_quantum_ligru"]))
    ]
    ablation_metrics = pd.concat([main_keep, ablation_metrics], ignore_index=True)
    main_pred_keep = main_predictions[
        (main_predictions["fold"] == fold)
        & (main_predictions["seed"].isin(seeds))
        & (main_predictions["model"].isin(["quantum_ligru", "egarch_quantum_ligru"]))
    ]
    ablation_predictions = pd.concat([main_pred_keep, ablation_predictions], ignore_index=True)

    ablation_metrics.to_csv(root / "results/ablations/architecture_metrics.csv", index=False)
    ablation_predictions.to_csv(
        root / "results/ablations/architecture_predictions.csv", index=False, date_format="%Y-%m-%d"
    )
    pd.DataFrame(resource_rows).to_csv(root / "results/ablations/architecture_resources.csv", index=False)


def configure_logging(root: Path) -> None:
    _ensure_dirs(root)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(root / "logs/run.log", mode="w", encoding="utf-8"),
        ],
    )
