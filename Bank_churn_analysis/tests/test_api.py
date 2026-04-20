"""
tests/test_api.py
-----------------
Pytest test suite for the FastAPI application.
Uses TestClient (no live server needed).

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Fixture: mock predictor so tests don't need a trained model ──────────
MOCK_PREDICTION = {
    "churn_probability" : 0.72,
    "churn_predicted"   : 1,
    "risk_segment"      : "High",
    "rfm_score"         : 6,
    "rfm_segment"       : "At Risk",
    "retention_priority": 2,
    "r_score"           : 2,
    "f_score"           : 2,
    "m_score"           : 2,
    "recommendation"    : "Urgent re-engagement — outbound call, retention offer.",
}

VALID_CUSTOMER = {
    "CreditScore"    : 650,
    "Geography"      : "France",
    "Gender"         : "Male",
    "Age"            : 42,
    "Tenure"         : 5,
    "Balance"        : 75000.0,
    "NumOfProducts"  : 2,
    "HasCrCard"      : 1,
    "IsActiveMember" : 1,
    "EstimatedSalary": 98000.0,
}


@pytest.fixture(scope="module")
def client():
    """Create a test client with the predictor mocked."""
    with patch("app.predictor.predictor") as mock_pred:
        mock_pred.is_loaded         = True
        mock_pred.model_name        = "xgboost"
        mock_pred.threshold         = 0.40
        mock_pred.feature_names     = ["CreditScore", "Age"]
        mock_pred.predict_one.return_value   = MOCK_PREDICTION
        mock_pred.predict_batch.return_value = [MOCK_PREDICTION, MOCK_PREDICTION]

        from app.main import app
        with TestClient(app) as c:
            yield c


# ── Health & system ───────────────────────────────────────────────────────

class TestSystem:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body

    def test_root_redirects(self, client):
        r = client.get("/", follow_redirects=False)
        assert r.status_code in (301, 302, 307, 308)

    def test_model_info(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        body = r.json()
        assert "model_name" in body
        assert "threshold"  in body

    def test_ui_returns_html(self, client):
        r = client.get("/ui")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Bank Churn Predictor" in r.text


# ── Single prediction ─────────────────────────────────────────────────────

class TestSinglePrediction:
    def test_valid_customer(self, client):
        r = client.post("/predict", json=VALID_CUSTOMER)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["churn_predicted"] in (0, 1)
        assert body["risk_segment"] in ("Low", "Medium", "High", "Critical")
        assert "rfm_segment" in body
        assert "recommendation" in body

    def test_missing_field_returns_422(self, client):
        bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "Age"}
        r   = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_invalid_geography_returns_422(self, client):
        bad = {**VALID_CUSTOMER, "Geography": "Australia"}
        r   = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_invalid_gender_returns_422(self, client):
        bad = {**VALID_CUSTOMER, "Gender": "Other"}
        r   = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_credit_score_out_of_range(self, client):
        bad = {**VALID_CUSTOMER, "CreditScore": 200}
        r   = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_age_boundary_values(self, client):
        for age in [18, 100]:
            r = client.post("/predict", json={**VALID_CUSTOMER, "Age": age})
            assert r.status_code == 200

    def test_zero_balance(self, client):
        r = client.post("/predict", json={**VALID_CUSTOMER, "Balance": 0.0})
        assert r.status_code == 200

    def test_all_geographies(self, client):
        for geo in ["France", "Germany", "Spain"]:
            r = client.post("/predict", json={**VALID_CUSTOMER, "Geography": geo})
            assert r.status_code == 200

    def test_inactive_member(self, client):
        r = client.post("/predict", json={**VALID_CUSTOMER, "IsActiveMember": 0})
        assert r.status_code == 200

    def test_rfm_score_range(self, client):
        r    = client.post("/predict", json=VALID_CUSTOMER)
        body = r.json()
        assert 3  <= body["rfm_score"]         <= 15
        assert 1  <= body["r_score"]           <= 5
        assert 1  <= body["f_score"]           <= 5
        assert 1  <= body["m_score"]           <= 5
        assert 1  <= body["retention_priority"] <= 5


# ── Batch prediction ──────────────────────────────────────────────────────

class TestBatchPrediction:
    def test_valid_batch(self, client):
        payload = {"customers": [VALID_CUSTOMER, VALID_CUSTOMER]}
        r       = client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 2
        assert len(body["results"]) == 2
        assert 0.0 <= body["predicted_churn_rate"] <= 1.0

    def test_single_item_batch(self, client):
        payload = {"customers": [VALID_CUSTOMER]}
        r       = client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        assert r.json()["total"] == 1

    def test_empty_batch_returns_422(self, client):
        r = client.post("/predict/batch", json={"customers": []})
        assert r.status_code == 422

    def test_batch_has_all_fields(self, client):
        payload = {"customers": [VALID_CUSTOMER]}
        r       = client.post("/predict/batch", json=payload)
        body    = r.json()
        result  = body["results"][0]
        for field in ["churn_probability", "churn_predicted", "risk_segment",
                      "rfm_segment", "rfm_score", "recommendation"]:
            assert field in result, f"Missing field: {field}"


# ── Schema validation ─────────────────────────────────────────────────────

class TestSchemaValidation:
    @pytest.mark.parametrize("field,bad_value", [
        ("NumOfProducts",  5),
        ("NumOfProducts",  0),
        ("HasCrCard",      2),
        ("IsActiveMember", -1),
        ("Tenure",         11),
        ("Tenure",         -1),
    ])
    def test_out_of_range_fields(self, client, field, bad_value):
        bad = {**VALID_CUSTOMER, field: bad_value}
        r   = client.post("/predict", json=bad)
        assert r.status_code == 422, (
            f"Expected 422 for {field}={bad_value}, got {r.status_code}"
        )
