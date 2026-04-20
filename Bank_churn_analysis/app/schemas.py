"""
app/schemas.py
--------------
Pydantic v2 request and response schemas for the FastAPI app.
These enforce types, provide validation, and auto-generate
the OpenAPI docs at /docs.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request — single customer
# ---------------------------------------------------------------------------

class CustomerInput(BaseModel):
    """
    All fields a relationship manager would know about a customer.
    Matches the raw bank_customers.csv schema exactly.
    """
    CreditScore     : int   = Field(..., ge=300, le=900,  example=650,
                                    description="Customer credit score [300–900]")
    Geography       : str   = Field(...,                  example="France",
                                    description="Country: France | Germany | Spain")
    Gender          : str   = Field(...,                  example="Male",
                                    description="Male | Female")
    Age             : int   = Field(..., ge=18, le=100,   example=42)
    Tenure          : int   = Field(..., ge=0,  le=10,    example=5,
                                    description="Years with bank [0–10]")
    Balance         : float = Field(..., ge=0,            example=75000.0,
                                    description="Account balance in €")
    NumOfProducts   : int   = Field(..., ge=1, le=4,      example=2,
                                    description="Number of bank products [1–4]")
    HasCrCard       : int   = Field(..., ge=0, le=1,      example=1,
                                    description="Has credit card: 1=yes, 0=no")
    IsActiveMember  : int   = Field(..., ge=0, le=1,      example=1,
                                    description="Active member: 1=yes, 0=no")
    EstimatedSalary : float = Field(..., ge=0,            example=98000.0,
                                    description="Estimated annual salary in €")

    @field_validator("Geography")
    @classmethod
    def validate_geography(cls, v):
        allowed = {"France", "Germany", "Spain"}
        if v not in allowed:
            raise ValueError(f"Geography must be one of {allowed}")
        return v

    @field_validator("Gender")
    @classmethod
    def validate_gender(cls, v):
        allowed = {"Male", "Female"}
        if v not in allowed:
            raise ValueError(f"Gender must be one of {allowed}")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "CreditScore": 650, "Geography": "France", "Gender": "Male",
            "Age": 42, "Tenure": 5, "Balance": 75000.0,
            "NumOfProducts": 2, "HasCrCard": 1,
            "IsActiveMember": 1, "EstimatedSalary": 98000.0,
        }
    }}


# ---------------------------------------------------------------------------
# Request — batch
# ---------------------------------------------------------------------------

class BatchInput(BaseModel):
    customers: List[CustomerInput] = Field(..., min_length=1, max_length=5000)


# ---------------------------------------------------------------------------
# Response — single prediction
# ---------------------------------------------------------------------------

class PredictionOutput(BaseModel):
    churn_probability : float  = Field(..., description="Predicted churn probability [0–1]")
    churn_predicted   : int    = Field(..., description="Binary prediction (0=retained, 1=churned)")
    risk_segment      : str    = Field(..., description="Low | Medium | High | Critical")
    rfm_score         : int    = Field(..., description="Composite RFM score [3–15]")
    rfm_segment       : str    = Field(..., description="Champion | Loyal Customer | ...")
    retention_priority: int    = Field(..., description="Retention priority [1–5]")
    r_score           : int    = Field(..., description="Recency score [1–5]")
    f_score           : int    = Field(..., description="Frequency score [1–5]")
    m_score           : int    = Field(..., description="Monetary score [1–5]")
    recommendation    : str    = Field(..., description="Action recommendation")


# ---------------------------------------------------------------------------
# Response — batch
# ---------------------------------------------------------------------------

class BatchOutput(BaseModel):
    total             : int
    predicted_churners: int
    predicted_churn_rate: float
    results           : List[PredictionOutput]


# ---------------------------------------------------------------------------
# Response — model info
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    model_name   : str
    model_version: str
    threshold    : float
    features     : int
    status       : str


# ---------------------------------------------------------------------------
# Response — health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status      : str
    model_loaded: bool
    version     : str
