from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def split_frame(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, stratify_col: str | None = None, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("train/val/test ratios must sum to 1")
    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
    train, temp = train_test_split(df, train_size=train_ratio, random_state=seed, stratify=stratify)
    relative_val = val_ratio / (val_ratio + test_ratio)
    stratify_temp = temp[stratify_col] if stratify_col and stratify_col in temp.columns else None
    val, test = train_test_split(temp, train_size=relative_val, random_state=seed, stratify=stratify_temp)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def ensure_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"review_text", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Text review data missing columns: {sorted(missing)}")
    out = df.copy()
    out["credibility"] = out.get("credibility", 1.0)
    out["recency_weight"] = out.get("recency_weight", 1.0)
    out["user_id"] = out.get("user_id", range(len(out)))
    out["entity_id"] = out.get("entity_id", "unknown")
    return out


def ensure_image_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Image manifest missing columns: {sorted(missing)}")
    out = df.copy()
    out["label"] = out.get("label", 1)
    out["provenance"] = out.get("provenance", 1.0)
    out["text_polarity"] = out.get("text_polarity", 0.5)
    out["user_credibility"] = out.get("user_credibility", 1.0)
    out["user_id"] = out.get("user_id", range(len(out)))
    out["entity_id"] = out.get("entity_id", "unknown")
    return out


def ensure_behavior_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "step", "visit_frequency", "duration", "loyalty_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Behavior log data missing columns: {sorted(missing)}")
    out = df.copy()
    out["spending"] = out.get("spending", 0.0)
    out["engagement"] = out.get("engagement", 0.0)
    return out
