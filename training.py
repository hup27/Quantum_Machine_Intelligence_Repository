"""Deterministic neural-network training and prediction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import copy
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import metric_summary
from .models import ModelSpec, build_model, count_parameters


@dataclass
class TrainingConfig:
    hidden_dim: int
    recurrent_layers: int
    dropout: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    min_delta: float
    gradient_clip: float


@dataclass
class TrainResult:
    model: nn.Module
    history: pd.DataFrame
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    best_epoch: int
    best_validation_mse: float
    runtime_seconds: float
    parameter_count: int


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def _predict_standardized(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    *,
    shots: Optional[int] = None,
    readout_error: float = 0.0,
    noise_seed: int = 0,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    generator = torch.Generator().manual_seed(noise_seed)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).float()
            pred = model(
                xb,
                shots=shots,
                readout_error=readout_error,
                generator=generator,
            )
            outputs.append(pred.detach().cpu().numpy())
    return np.concatenate(outputs).astype(float)


def train_one(
    spec: ModelSpec,
    X: Dict[str, np.ndarray],
    y: Dict[str, np.ndarray],
    raw_y: Dict[str, np.ndarray],
    metadata: Dict[str, pd.DataFrame],
    target_scaler: StandardScaler,
    config: TrainingConfig,
    seed: int,
    fold: str,
    save_state_path: Optional[Path] = None,
) -> TrainResult:
    set_seed(seed)
    model = build_model(
        spec,
        input_dim=X["train"].shape[-1],
        hidden_dim=config.hidden_dim,
        recurrent_layers=config.recurrent_layers,
        dropout=config.dropout,
    )
    parameter_count = count_parameters(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()
    train_loader = _loader(X["train"], y["train"], config.batch_size, True, seed)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    records: List[Dict[str, float]] = []
    started = perf_counter()

    X_val_t = torch.from_numpy(X["validation"]).float()
    y_val_t = torch.from_numpy(y["validation"]).float()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_sse = 0.0
        train_n = 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            train_sse += float(loss.detach()) * len(xb)
            train_n += len(xb)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_mse = float(criterion(val_pred, y_val_t))
        train_mse = train_sse / max(train_n, 1)
        records.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": val_mse})

        if val_mse < best_val - config.min_delta:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    runtime = perf_counter() - started
    model.load_state_dict(best_state)
    if save_state_path is not None:
        save_state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "spec": spec.__dict__,
            "input_dim": X["train"].shape[-1],
            "hidden_dim": config.hidden_dim,
            "recurrent_layers": config.recurrent_layers,
            "dropout": config.dropout,
            "seed": seed,
            "fold": fold,
            "best_epoch": best_epoch,
        }
        temporary_state = save_state_path.with_name(save_state_path.name + ".tmp")
        torch.save(payload, temporary_state)
        os.replace(temporary_state, save_state_path)

    metric_rows: List[Dict[str, object]] = []
    pred_rows: List[pd.DataFrame] = []
    for split in ("train", "validation", "test"):
        standardized = _predict_standardized(model, X[split], config.batch_size)
        pred_log = target_scaler.inverse_transform(standardized.reshape(-1, 1)).reshape(-1)
        summary = metric_summary(raw_y[split], pred_log)
        metric_rows.append(
            {
                "fold": fold,
                "model": spec.name,
                "seed": seed,
                "split": split,
                "best_epoch": best_epoch,
                "best_validation_mse_standardized": best_val,
                "runtime_seconds": runtime,
                "parameter_count": parameter_count,
                **summary,
            }
        )
        frame = metadata[split].copy()
        frame["model"] = spec.name
        frame["seed"] = seed
        frame["actual_log_variance"] = raw_y[split]
        frame["predicted_log_variance"] = pred_log
        frame["actual_variance"] = np.exp(np.clip(raw_y[split], -30, 5))
        frame["predicted_variance"] = np.exp(np.clip(pred_log, -30, 5))
        pred_rows.append(frame)

    return TrainResult(
        model=model,
        history=pd.DataFrame(records).assign(fold=fold, model=spec.name, seed=seed),
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(pred_rows, ignore_index=True),
        best_epoch=best_epoch,
        best_validation_mse=best_val,
        runtime_seconds=runtime,
        parameter_count=parameter_count,
    )


def predict_with_noise(
    model: nn.Module,
    X: np.ndarray,
    target_scaler: StandardScaler,
    batch_size: int,
    *,
    shots: Optional[int],
    readout_error: float,
    noise_seed: int,
) -> np.ndarray:
    standardized = _predict_standardized(
        model,
        X,
        batch_size,
        shots=shots,
        readout_error=readout_error,
        noise_seed=noise_seed,
    )
    return target_scaler.inverse_transform(standardized.reshape(-1, 1)).reshape(-1)
