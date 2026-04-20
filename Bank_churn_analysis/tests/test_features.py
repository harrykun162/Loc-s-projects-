"""
tests/test_features.py
-----------------------
Unit tests for RFM engineering and feature engineering modules.
These run fully offline — no model or CSV required.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.rfm      import build_rfm
from src.features.engineer import (
    _interaction_features,
    _binned_features,
    _encode,
)


# ── Sample data fixture ───────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal 5-row customer DataFrame."""
    return pd.DataFrame({
        "CustomerId"     : [1, 2, 3, 4, 5],
        "Surname"        : ["A", "B", "C", "D", "E"],
        "CreditScore"    : [650, 720, 580, 490, 810],
        "Geography"      : ["France", "Germany", "Spain", "France", "Germany"],
        "Gender"         : ["Male", "Female", "Male", "Female", "Male"],
        "Age"            : [42, 55, 33, 61, 28],
        "Tenure"         : [5, 2, 8, 0, 10],
        "Balance"        : [75000.0, 0.0, 125000.0, 55000.0, 200000.0],
        "NumOfProducts"  : [2, 1, 1, 3, 2],
        "HasCrCard"      : [1, 1, 0, 1, 0],
        "IsActiveMember" : [1, 0, 1, 0, 1],
        "EstimatedSalary": [98000.0, 55000.0, 72000.0, 110000.0, 145000.0],
        "Exited"         : [0, 1, 0, 1, 0],
    })


# ── RFM engineering ───────────────────────────────────────────────────────

class TestRFMEngineering:
    def test_rfm_columns_added(self, sample_df):
        df = build_rfm(sample_df.copy())
        for col in ["R_raw", "F_raw", "M_raw",
                    "R_score", "F_score", "M_score",
                    "RFM_Score", "RFM_Cell", "RFM_Segment",
                    "Retention_Priority"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_rfm_scores_in_range(self, sample_df):
        df = build_rfm(sample_df.copy())
        assert df["R_score"].between(1, 5).all()
        assert df["F_score"].between(1, 5).all()
        assert df["M_score"].between(1, 5).all()

    def test_composite_score_range(self, sample_df):
        df = build_rfm(sample_df.copy())
        assert df["RFM_Score"].between(3, 15).all()

    def test_rfm_cell_format(self, sample_df):
        df = build_rfm(sample_df.copy())
        for cell in df["RFM_Cell"]:
            parts = cell.split("-")
            assert len(parts) == 3
            assert all(p.isdigit() for p in parts)

    def test_valid_segments(self, sample_df):
        valid = {"Champion", "Loyal Customer", "Potential Loyalist",
                 "At Risk", "Lost / Hibernating"}
        df = build_rfm(sample_df.copy())
        assert set(df["RFM_Segment"].unique()).issubset(valid)

    def test_retention_priority_range(self, sample_df):
        df = build_rfm(sample_df.copy())
        assert df["Retention_Priority"].between(1, 5).all()

    def test_f_raw_equals_num_products(self, sample_df):
        df = build_rfm(sample_df.copy())
        assert (df["F_raw"] == df["NumOfProducts"].astype(float)).all()

    def test_zero_balance_customer(self, sample_df):
        """Customer with zero balance should not error."""
        df = build_rfm(sample_df.copy())
        zero_bal = df[df["Balance"] == 0.0]
        assert len(zero_bal) == 1

    def test_inactive_member_lower_r(self, sample_df):
        """Inactive members should have lower R_raw than active with same tenure."""
        df = build_rfm(sample_df.copy())
        active   = df[df["IsActiveMember"] == 1]["R_raw"]
        inactive = df[df["IsActiveMember"] == 0]["R_raw"]
        # The average of active should be higher due to the +1 bonus
        assert active.mean() > inactive.mean()


# ── Feature engineering ───────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_interaction_features_created(self, sample_df):
        df = _interaction_features(sample_df.copy())
        for feat in ["Balance_Salary_Ratio", "Products_per_Tenure",
                     "Balance_per_Product", "CreditScore_Age_Ratio"]:
            assert feat in df.columns

    def test_no_nan_in_interactions(self, sample_df):
        """Division by zero (Tenure=0) should be filled with 0."""
        df = _interaction_features(sample_df.copy())
        assert not df["Products_per_Tenure"].isna().any()

    def test_binned_features_created(self, sample_df):
        df = _binned_features(sample_df.copy())
        assert "Age_Band"     in df.columns
        assert "Balance_Band" in df.columns

    def test_age_bands_are_strings(self, sample_df):
        df = _binned_features(sample_df.copy())
        assert df["Age_Band"].dtype == object

    def test_encode_creates_dummies(self, sample_df):
        df = build_rfm(sample_df.copy())
        df = _interaction_features(df)
        df = _binned_features(df)
        df = _encode(df)
        # Geography and Gender should be one-hot encoded
        assert "Geography" not in df.columns
        assert "Gender"    not in df.columns
        # At least one dummy column per category
        geo_cols = [c for c in df.columns if c.startswith("Geography_")]
        assert len(geo_cols) >= 2

    def test_row_count_preserved(self, sample_df):
        df = build_rfm(sample_df.copy())
        df = _interaction_features(df)
        df = _binned_features(df)
        df = _encode(df)
        assert len(df) == len(sample_df)


# ── Edge cases ────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_row(self):
        single = pd.DataFrame([{
            "CustomerId": 999, "Surname": "X",
            "CreditScore": 700, "Geography": "Spain", "Gender": "Female",
            "Age": 35, "Tenure": 4, "Balance": 60000.0,
            "NumOfProducts": 2, "HasCrCard": 1,
            "IsActiveMember": 1, "EstimatedSalary": 85000.0,
            "Exited": 0,
        }])
        df = build_rfm(single)
        assert len(df) == 1
        assert df["RFM_Score"].iloc[0] >= 3

    def test_all_germany(self, sample_df):
        df_de = sample_df.copy()
        df_de["Geography"] = "Germany"
        df = build_rfm(df_de)
        assert len(df) == len(sample_df)
