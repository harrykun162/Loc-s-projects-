"""
app/predictor.py
----------------
Stateless inference engine.  Loaded once at app startup via
FastAPI lifespan context.  Handles feature engineering and
model scoring for both single and batch requests.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from src.features.rfm import build_rfm
from src.features.engineer import (
    _interaction_features,
    _binned_features,
    _encode,
)
from src.utils.config_loader import settings
from src.utils.logger import get_logger

log = get_logger(__name__)
S = settings

# ── Risk tier labels ──────────────────────────────────────────────────────
RISK_TIERS = [
    (0.75, "Critical"),
    (0.50, "High"),
    (0.25, "Medium"),
    (0.00, "Low"),
]

# ── Retention recommendations per segment ────────────────────────────────
RECOMMENDATIONS = {
    "Champion"          : "Protect & reward — offer VIP tier or early access to products.",
    "Loyal Customer"    : "Upsell opportunity — cross-sell one additional product.",
    "Potential Loyalist": "Deepen engagement — push to second product; activate if inactive.",
    "At Risk"           : "Urgent re-engagement — outbound call, retention offer.",
    "Lost / Hibernating": "Selective win-back — contact only if balance > €50k.",
}


class ChurnPredictor:
    def __init__(self):
        self.pipeline      = None
        self.feature_names = []
        self.model_name    = "unknown"
        self.threshold     = float(S.model.decision_threshold)
        self._loaded       = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        model_path   = Path(S.paths.best_model)
        feature_path = Path(S.paths.feature_names)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run: python scripts/train_pipeline.py first."
            )

        self.pipeline   = joblib.load(model_path)
        self.model_name = model_path.stem

        if feature_path.exists():
            with open(feature_path) as f:
                self.feature_names = json.load(f)

        self._loaded = True
        log.info(
            f"Model loaded: {self.model_name} | "
            f"features: {len(self.feature_names)} | "
            f"threshold: {self.threshold}"
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_one(self, customer_dict: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame([customer_dict])
        return self._score(df)[0]

    def predict_batch(self, customers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        df = pd.DataFrame(customers)
        return self._score(df)

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _score(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        # 1. RFM engineering
        df = build_rfm(df)

        # 2. Capture RFM outputs before they get encoded/dropped
        rfm_cols = df[["R_score", "F_score", "M_score",
                        "RFM_Score", "RFM_Segment",
                        "Retention_Priority"]].copy()

        # 3. Feature engineering
        df = _interaction_features(df)
        df = _binned_features(df)
        df = _encode(df)

        # 4. Align to training feature schema
        X = self._align(df)

        # 5. Score
        proba = self.pipeline.predict_proba(X)[:, 1]
        pred  = (proba >= self.threshold).astype(int)

        # 6. Build output dicts
        results = []
        for i in range(len(proba)):
            seg = rfm_cols["RFM_Segment"].iloc[i]
            results.append({
                "churn_probability" : round(float(proba[i]), 4),
                "churn_predicted"   : int(pred[i]),
                "risk_segment"      : _risk_tier(proba[i]),
                "rfm_score"         : int(rfm_cols["RFM_Score"].iloc[i]),
                "rfm_segment"       : seg,
                "retention_priority": int(rfm_cols["Retention_Priority"].iloc[i]),
                "r_score"           : int(rfm_cols["R_score"].iloc[i]),
                "f_score"           : int(rfm_cols["F_score"].iloc[i]),
                "m_score"           : int(rfm_cols["M_score"].iloc[i]),
                "recommendation"    : RECOMMENDATIONS.get(seg, "Monitor closely."),
            })
        return results

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Match columns to training feature schema exactly."""
        if not self.feature_names:
            return df
        # Drop target + identifier + RFM raw columns if present
        drop = (S.data.drop_cols
                + [S.data.target_col, "RFM_Cell", "R_raw", "F_raw", "M_raw"]
                + ["R_score", "F_score", "M_score", "RFM_Score",
                   "RFM_Segment", "Retention_Priority"])
        X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        return X[self.feature_names]


def _risk_tier(prob: float) -> str:
    for threshold, label in RISK_TIERS:
        if prob >= threshold:
            return label
    return "Low"


# Singleton — imported by main.py
predictor = ChurnPredictor()
