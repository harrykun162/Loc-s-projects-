"""
Streamlit frontend for the bank churn prediction model.

Run locally with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.predictor import predictor
from app.schemas import CustomerInput
from src.utils.config_loader import settings

S = settings

st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_model():
    predictor.load()
    return predictor


def _risk_color(segment: str) -> str:
    return {
        "Critical": "#A32D2D",
        "High": "#D85A30",
        "Medium": "#C98612",
        "Low": "#3B6D11",
    }.get(segment, "#185FA5")


def main() -> None:
    st.title("Bank Churn Predictor")
    st.caption("RFM-enhanced churn prediction for European bank customers")

    try:
        model = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Train the model first with `python scripts/train_pipeline.py`.")
        st.stop()

    with st.sidebar:
        st.subheader("Model")
        st.write(f"Name: `{model.model_name}`")
        st.write(f"Version: `{S.project.version}`")
        st.write(f"Threshold: `{model.threshold:.2f}`")
        st.write(f"Features: `{len(model.feature_names)}`")

    left, right = st.columns([1.1, 0.9])

    with left:
        st.subheader("Customer Profile")
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
                geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
                tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
                products = st.selectbox("Products Held", [1, 2, 3, 4], index=1)
                has_card = st.selectbox(
                    "Has Credit Card",
                    [1, 0],
                    index=0,
                    format_func=lambda x: "Yes" if x == 1 else "No",
                )
            with col2:
                age = st.number_input("Age", min_value=18, max_value=100, value=42)
                gender = st.selectbox("Gender", ["Male", "Female"])
                balance = st.number_input("Balance (EUR)", min_value=0.0, value=75000.0, step=1000.0)
                salary = st.number_input("Estimated Salary (EUR)", min_value=0.0, value=98000.0, step=1000.0)
                active_member = st.selectbox(
                    "Active Member",
                    [1, 0],
                    index=0,
                    format_func=lambda x: "Yes" if x == 1 else "No",
                )

            submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    with right:
        st.subheader("Prediction")
        if submitted:
            payload = {
                "CreditScore": int(credit_score),
                "Geography": geography,
                "Gender": gender,
                "Age": int(age),
                "Tenure": int(tenure),
                "Balance": float(balance),
                "NumOfProducts": int(products),
                "HasCrCard": int(has_card),
                "IsActiveMember": int(active_member),
                "EstimatedSalary": float(salary),
            }

            try:
                validated = CustomerInput(**payload)
                result = model.predict_one(validated.model_dump())
            except ValidationError as exc:
                st.error("Input validation failed.")
                st.code(str(exc))
                st.stop()
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

            risk_color = _risk_color(result["risk_segment"])
            st.metric("Churn Probability", f'{result["churn_probability"]:.1%}')
            st.markdown(
                (
                    f"<div style='padding:0.6rem 0.9rem;border-radius:0.75rem;"
                    f"background:{risk_color}14;color:{risk_color};font-weight:600;'>"
                    f"{result['risk_segment']} Risk</div>"
                ),
                unsafe_allow_html=True,
            )

            a, b, c = st.columns(3)
            a.metric("Recency", result["r_score"])
            b.metric("Frequency", result["f_score"])
            c.metric("Monetary", result["m_score"])

            st.write(f"RFM Segment: `{result['rfm_segment']}`")
            st.write(f"RFM Score: `{result['rfm_score']} / 15`")
            st.write(f"Retention Priority: `{result['retention_priority']} / 5`")
            st.info(result["recommendation"])
        else:
            st.caption("Submit the form to score a customer.")


if __name__ == "__main__":
    main()
