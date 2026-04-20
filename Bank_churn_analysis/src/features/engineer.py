"""
src/features/engineer.py
------------------------
Build the ML feature matrix from the raw/cleaned DataFrame.
Produces X (features) and y (target). Saves feature names for inference.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.config_loader import settings
from src.utils.logger import get_logger

log = get_logger(__name__)
S = settings


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = _interaction_features(df)
    df = _binned_features(df)
    df = _encode(df)
    X, y = _split(df)
    _save_feature_names(X)
    log.info(f"Feature matrix: {X.shape} | positive rate: {y.mean():.2%}")
    return X, y


def _interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio / interaction features that improve tree-model performance."""
    pairs = [
        ("Balance_Salary_Ratio",  "Balance",       "EstimatedSalary"),
        ("Products_per_Tenure",   "NumOfProducts", "Tenure"),
        ("Balance_per_Product",   "Balance",       "NumOfProducts"),
        ("CreditScore_Age_Ratio", "CreditScore",   "Age"),
    ]
    for name, num, den in pairs:
        df[name] = (df[num] / df[den].replace(0, np.nan)).fillna(0)
    return df


def _binned_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bin Age and Balance to capture non-linear effects."""
    fc = S.features
    balance_bins = [float(x) for x in fc.balance_bins]
    df["Age_Band"] = pd.cut(
        df["Age"], bins=fc.age_bins, labels=fc.age_labels, right=True
    ).astype(str)
    df["Balance_Band"] = pd.cut(
        df["Balance"], bins=balance_bins, labels=fc.balance_labels, right=True
    ).astype(str)
    return df


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Geography, Gender, Age_Band, Balance_Band.
    RFM_Segment is deliberately excluded — it lives in SQL/Power BI only.
    """
    cat_cols = list(S.data.categorical) + ["Age_Band", "Balance_Band"]
    return pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)


def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features from target; drop identifier columns."""
    y = df[S.data.target_col].copy()
    X = df.drop(columns=[S.data.target_col] + S.data.drop_cols, errors="ignore")
    return X, y


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    out = Path(S.paths.scaler)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out)
    log.info(f"Scaler saved -> {out}")
    return scaler


def _save_feature_names(X: pd.DataFrame) -> None:
    out = Path(S.paths.feature_names)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(X.columns.tolist(), f, indent=2)


def run(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return build_features(df)
