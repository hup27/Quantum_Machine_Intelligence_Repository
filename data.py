"""Chronology-safe data preparation for the QMI volatility experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


EPISODE_FILES: Mapping[str, str] = {
    "black_monday_1987": "S&P 500 Historical Data 1987.csv",
    "dotcom_2000_2002": "S&P 500 Historical Data 2000-2002.csv",
    "gfc_2008": "S&P 500 Historical Data 2008.csv",
    "covid_2020_2022": "S&P 500 Historical Data 2020-2022.csv",
}


@dataclass(frozen=True)
class FoldDefinition:
    name: str
    description: str


FOLDS: Tuple[FoldDefinition, ...] = (
    FoldDefinition(
        "dotcom",
        "Train on 1987; validate on 2000H1; test on 2000H2-2002.",
    ),
    FoldDefinition(
        "gfc",
        "Train on 1987 and 2000-2001; validate on 2002; test on 2008.",
    ),
    FoldDefinition(
        "covid",
        "Train on 1987 and 2000-2002; validate on 2008; test on 2020-2022.",
    ),
)


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def load_market_panel(raw_dir: Path, epsilon: float = 1e-12) -> pd.DataFrame:
    """Load all supplied episodes and compute causal daily features.

    The source CSVs are reverse chronological. This function explicitly sorts each
    episode in ascending order and never constructs a sequence across an episode gap.
    The volatility target is the next day's log Parkinson range variance, not a
    claim of intraday realized variance.
    """
    frames: List[pd.DataFrame] = []
    for episode, filename in EPISODE_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing source data: {path}")
        df = pd.read_csv(path, encoding="utf-8-sig")
        required = {"Date", "Price", "Open", "High", "Low"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{filename} is missing columns: {sorted(missing)}")

        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="raise"),
                "close": _to_float(df["Price"]),
                "open": _to_float(df["Open"]),
                "high": _to_float(df["High"]),
                "low": _to_float(df["Low"]),
            }
        )
        out["episode"] = episode
        out = out.sort_values("date", kind="mergesort").reset_index(drop=True)
        if not out["date"].is_monotonic_increasing:
            raise AssertionError(f"Chronology repair failed for {episode}")
        if out["date"].duplicated().any():
            raise ValueError(f"Duplicate dates detected in {episode}")
        if out[["close", "open", "high", "low"]].isna().any().any():
            raise ValueError(f"Missing OHLC values detected in {episode}")
        if (out[["close", "open", "high", "low"]] <= 0).any().any():
            raise ValueError(f"Non-positive prices detected in {episode}")
        if (out["high"] < out["low"]).any():
            raise ValueError(f"High < low detected in {episode}")

        log_close = np.log(out["close"].to_numpy(dtype=float))
        returns = np.empty_like(log_close)
        returns[:] = np.nan
        returns[1:] = np.diff(log_close)
        log_hl = np.log(out["high"].to_numpy(dtype=float) / out["low"].to_numpy(dtype=float))
        park_var = np.maximum((log_hl**2) / (4.0 * np.log(2.0)), epsilon)

        out["return"] = returns
        out["abs_return"] = np.abs(returns)
        out["parkinson_var"] = park_var
        out["log_parkinson_var"] = np.log(park_var)
        out["intraday_return"] = np.log(out["close"] / out["open"])
        out["year"] = out["date"].dt.year
        frames.append(out)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "episode"], kind="mergesort").reset_index(drop=True)
    panel["row_id"] = np.arange(len(panel), dtype=int)
    return panel


def target_split(fold: str, episode: str, target_date: pd.Timestamp) -> str:
    """Assign a target date to train/validation/test for a predeclared fold."""
    d = pd.Timestamp(target_date)
    if fold == "dotcom":
        if episode == "black_monday_1987":
            return "train"
        if episode == "dotcom_2000_2002":
            if d <= pd.Timestamp("2000-06-30"):
                return "validation"
            return "test"
        return "unused"

    if fold == "gfc":
        if episode == "black_monday_1987":
            return "train"
        if episode == "dotcom_2000_2002":
            if d <= pd.Timestamp("2001-12-31"):
                return "train"
            return "validation"
        if episode == "gfc_2008":
            return "test"
        return "unused"

    if fold == "covid":
        if episode in {"black_monday_1987", "dotcom_2000_2002"}:
            return "train"
        if episode == "gfc_2008":
            return "validation"
        if episode == "covid_2020_2022":
            return "test"
        return "unused"

    raise KeyError(f"Unknown fold: {fold}")


def mark_target_splits(panel: pd.DataFrame, fold: str) -> pd.DataFrame:
    marked = panel.copy()
    marked["target_split"] = [
        target_split(fold, episode, date)
        for episode, date in zip(marked["episode"], marked["date"])
    ]
    return marked


@dataclass
class WindowBundle:
    X_raw: np.ndarray
    y_raw: np.ndarray
    metadata: pd.DataFrame
    feature_columns: Tuple[str, ...]

    def subset(self, split: str) -> "WindowBundle":
        mask = self.metadata["split"].to_numpy() == split
        return WindowBundle(
            X_raw=self.X_raw[mask],
            y_raw=self.y_raw[mask],
            metadata=self.metadata.loc[mask].reset_index(drop=True),
            feature_columns=self.feature_columns,
        )


def build_windows(
    panel: pd.DataFrame,
    fold: str,
    feature_columns: Sequence[str],
    sequence_length: int,
) -> WindowBundle:
    """Build next-day windows, with no window allowed to cross an episode boundary."""
    feature_columns = tuple(feature_columns)
    missing = set(feature_columns).difference(panel.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    X: List[np.ndarray] = []
    y: List[float] = []
    meta: List[dict] = []

    for episode, ep in panel.groupby("episode", sort=False):
        ep = ep.sort_values("date").reset_index(drop=True)
        features = ep.loc[:, feature_columns].to_numpy(dtype=float)
        target = ep["log_parkinson_var"].to_numpy(dtype=float)
        dates = ep["date"].to_numpy()
        returns = ep["return"].to_numpy(dtype=float)
        park_var = ep["parkinson_var"].to_numpy(dtype=float)

        # i is the final observed day; target is i+1.
        for i in range(sequence_length - 1, len(ep) - 1):
            target_date = pd.Timestamp(dates[i + 1])
            split = target_split(fold, episode, target_date)
            if split == "unused":
                continue
            x = features[i - sequence_length + 1 : i + 1]
            if not np.isfinite(x).all():
                # The first return of each episode is undefined; initial windows may contain it.
                continue
            if not np.isfinite(target[i + 1]):
                continue
            X.append(x)
            y.append(float(target[i + 1]))
            meta.append(
                {
                    "fold": fold,
                    "split": split,
                    "episode": episode,
                    "feature_start_date": pd.Timestamp(dates[i - sequence_length + 1]),
                    "feature_end_date": pd.Timestamp(dates[i]),
                    "target_date": target_date,
                    "current_log_parkinson_var": float(target[i]),
                    "target_parkinson_var": float(park_var[i + 1]),
                    "current_return": float(returns[i]),
                    "row_index_in_episode": int(i),
                }
            )

    if not X:
        raise RuntimeError(f"No windows constructed for fold={fold}")
    metadata = pd.DataFrame(meta).sort_values("target_date", kind="mergesort").reset_index(drop=True)
    order = metadata.index.to_numpy()
    # meta was appended episode-by-episode; restore the same sorted order explicitly.
    original_meta = pd.DataFrame(meta)
    sorted_idx = original_meta.sort_values("target_date", kind="mergesort").index.to_numpy()
    X_arr = np.asarray(X, dtype=np.float32)[sorted_idx]
    y_arr = np.asarray(y, dtype=np.float32)[sorted_idx]
    metadata = original_meta.iloc[sorted_idx].reset_index(drop=True)

    # Strong chronology invariant: every feature end precedes its target.
    if not (metadata["feature_end_date"] < metadata["target_date"]).all():
        raise AssertionError("At least one window uses contemporaneous/future target information")
    # Every window was created inside one episode.  Verify that target and
    # feature dates resolve to that episode in the cleaned panel.
    episode_intervals = (
        panel.groupby("episode")["date"].agg(["min", "max"]).to_dict("index")
    )
    for row in metadata.itertuples(index=False):
        bounds = episode_intervals[row.episode]
        if not (bounds["min"] <= row.feature_start_date <= row.feature_end_date < row.target_date <= bounds["max"]):
            raise AssertionError("Window crosses an episode boundary or has invalid chronology")

    return WindowBundle(X_arr, y_arr, metadata, feature_columns)


@dataclass
class ScaledWindows:
    X: Dict[str, np.ndarray]
    y: Dict[str, np.ndarray]
    raw_y: Dict[str, np.ndarray]
    metadata: Dict[str, pd.DataFrame]
    feature_scaler: RobustScaler
    target_scaler: StandardScaler
    feature_columns: Tuple[str, ...]


def scale_windows(bundle: WindowBundle) -> ScaledWindows:
    train = bundle.subset("train")
    if len(train.y_raw) < 50:
        raise ValueError("Training fold has too few windows")

    feature_scaler = RobustScaler(quantile_range=(25.0, 75.0))
    feature_scaler.fit(train.X_raw.reshape(-1, train.X_raw.shape[-1]))
    target_scaler = StandardScaler()
    target_scaler.fit(train.y_raw.reshape(-1, 1))

    X_out: Dict[str, np.ndarray] = {}
    y_out: Dict[str, np.ndarray] = {}
    raw_y: Dict[str, np.ndarray] = {}
    metadata: Dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        part = bundle.subset(split)
        if len(part.y_raw) == 0:
            raise ValueError(f"Fold is missing split: {split}")
        shape = part.X_raw.shape
        X_out[split] = feature_scaler.transform(
            part.X_raw.reshape(-1, shape[-1])
        ).reshape(shape).astype(np.float32)
        y_out[split] = target_scaler.transform(part.y_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)
        raw_y[split] = part.y_raw.astype(np.float64)
        metadata[split] = part.metadata.copy()

    return ScaledWindows(
        X=X_out,
        y=y_out,
        raw_y=raw_y,
        metadata=metadata,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=bundle.feature_columns,
    )


def save_clean_panel(panel: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, index=False, date_format="%Y-%m-%d")
