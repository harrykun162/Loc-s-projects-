"""
src/features/rfm.py
-------------------
Compute RFM proxy scores, quintile rank (1–5), composite score,
and assign segment labels. Mirrors the SQL logic exactly.
"""
from __future__ import annotations
from functools import lru_cache
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config_loader import settings
from src.utils.logger import get_logger

log = get_logger(__name__)
S = settings


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    df = _raw_proxies(df)
    df = _quintile_scores(df)
    df = _composite(df)
    df = _segments(df)
    _log_stats(df)
    return df


def _raw_proxies(df: pd.DataFrame) -> pd.DataFrame:
    rc = S.rfm
    df["R_raw"] = (
        df["IsActiveMember"].astype(float) * float(rc.r_activity_weight)
        + df["Tenure"].astype(float) / float(rc.r_tenure_scale)
    ).round(4)

    df["F_raw"] = (
    df["NumOfProducts"].astype(float) * float(rc.f_products_weight)
    + df["IsActiveMember"].astype(float) * float(rc.f_active_weight)
    + df["HasCrCard"].astype(float) * float(rc.f_card_weight)
).round(2)

    df["M_raw"] = (
        df["Balance"].astype(float)        * float(rc.m_balance_weight)
        + df["EstimatedSalary"].astype(float) * float(rc.m_salary_weight)
    ).round(2)
    return df


def _quintile_scores(df: pd.DataFrame) -> pd.DataFrame:
    n = int(S.rfm.n_quintiles)
    ref = _reference_rfm_frame()
    for dim, col in [("R", "R_raw"), ("F", "F_raw"), ("M", "M_raw")]:
        reference = ref[col] if ref is not None and col in ref.columns else df[col]
        df[f"{dim}_score"] = _score_against_reference(df[col], reference, n)
    return df


@lru_cache(maxsize=1)
def _reference_rfm_frame() -> pd.DataFrame | None:
    path = Path(S.paths.processed_data)
    if not path.exists():
        return None
    ref = pd.read_parquet(path)
    return _raw_proxies(ref.copy())


def _score_against_reference(
    values: pd.Series,
    reference: pd.Series,
    n_quintiles: int,
) -> pd.Series:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    val = pd.to_numeric(values, errors="coerce")

    if ref.empty:
        fallback = max(1, (n_quintiles + 1) // 2)
        return pd.Series(fallback, index=values.index, dtype=int)

    quantiles = np.linspace(0, 1, n_quintiles + 1)[1:-1]
    edges = np.unique(ref.quantile(quantiles).to_numpy(dtype=float))

    scores = np.searchsorted(edges, val.to_numpy(dtype=float), side="right") + 1
    return pd.Series(scores, index=values.index).clip(1, n_quintiles).astype(int)


def _composite(df: pd.DataFrame) -> pd.DataFrame:
    df["RFM_Score"] = df["R_score"] + df["F_score"] + df["M_score"]
    df["RFM_Cell"]  = (
        df["R_score"].astype(str) + "-"
        + df["F_score"].astype(str) + "-"
        + df["M_score"].astype(str)
    )
    return df


def _segments(df: pd.DataFrame) -> pd.DataFrame:
    seg_map = {int(k): v for k, v in S.rfm.segments.items()}
    thresholds = sorted(seg_map.keys(), reverse=True)

    priority_map = {
        "Champion": 5, "Loyal Customer": 4,
        "Potential Loyalist": 3, "At Risk": 2, "Lost / Hibernating": 1,
    }

    def label(score):
        for t in thresholds:
            if score >= t:
                return seg_map[t]
        return seg_map[0]

    df["RFM_Segment"]        = df["RFM_Score"].apply(label)
    df["Retention_Priority"] = df["RFM_Segment"].map(priority_map).fillna(0).astype(int)
    return df


def _log_stats(df: pd.DataFrame) -> None:
    if S.data.target_col not in df.columns:
        return
    stats = (
        df.groupby("RFM_Segment")[S.data.target_col]
        .agg(["count", "mean"])
        .rename(columns={"count": "n", "mean": "churn_rate"})
    )
    log.info(f"\n{stats.to_string()}")
    for seg, row in stats.iterrows():
        key = seg.lower().replace(" ", "_").replace("/", "").replace("__", "_")
        mlflow.log_metrics({
            f"rfm_{key}_n":          int(row["n"]),
            f"rfm_{key}_churn_rate": round(float(row["churn_rate"]), 4),
        })


def run(df: pd.DataFrame) -> pd.DataFrame:
    df = build_rfm(df)
    out = Path(S.paths.rfm_data)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info(f"RFM master saved -> {out}")
    return df
