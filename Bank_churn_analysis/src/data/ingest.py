"""
src/data/ingest.py
------------------
Load raw CSV, enforce schema, run quality checks, persist parquet.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple

import mlflow
import pandas as pd

from src.utils.config_loader import settings
from src.utils.logger import get_logger

log = get_logger(__name__)
S = settings


def load_raw(path: str = None) -> pd.DataFrame:
    filepath = path or S.paths.raw_data
    log.info(f"Loading raw data: {filepath}")
    df = pd.read_csv(filepath)
    df = _cast_dtypes(df)
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} cols")
    return df


def validate(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    report = {
        "n_rows":        int(len(df)),
        "n_cols":        int(df.shape[1]),
        "duplicate_ids": int(df[S.data.customer_id].duplicated().sum()),
        "null_counts":   {k: int(v) for k, v in df.isnull().sum().items() if v > 0},
        "churn_rate":    round(float(df[S.data.target_col].mean()), 4),
        "geographies":   df["Geography"].unique().tolist(),
        "genders":       df["Gender"].unique().tolist(),
    }

    if report["duplicate_ids"] > 0:
        log.warning(f"Dropping {report['duplicate_ids']} duplicate rows")
        df = df.drop_duplicates(subset=[S.data.customer_id])

    if report["null_counts"]:
        log.warning(f"Imputing nulls: {report['null_counts']}")
        df = _impute(df)

    # Assertions
    assert df["CreditScore"].between(300, 900).all(), "CreditScore out of [300,900]"
    assert df["Age"].between(18, 100).all(),           "Age out of [18,100]"
    assert df["NumOfProducts"].between(1, 4).all(),    "NumOfProducts out of [1,4]"

    log.info(f"Validation passed | churn_rate={report['churn_rate']:.1%}")
    return df, report


def save_processed(df: pd.DataFrame) -> Path:
    out = Path(S.paths.processed_data)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info(f"Saved -> {out}")
    return out


def run(path: str = None) -> pd.DataFrame:
    df = load_raw(path)
    df, report = validate(df)
    save_processed(df)

    # MLflow
    mlflow.log_params({
        "ingest_n_rows":    report["n_rows"],
        "ingest_churn_rate": report["churn_rate"],
    })

    # Save quality report
    rp = Path("artifacts/data_quality_report.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(str(rp))
    return df


def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["CreditScore", "Age", "Tenure", "NumOfProducts",
              "HasCrCard", "IsActiveMember", "Exited"]:
        if c in df.columns:
            df[c] = df[c].astype(int)
    for c in ["Balance", "EstimatedSalary"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
    for c in ["Geography", "Gender"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    for col in S.data.numeric:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in S.data.categorical:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df
