"""Interactive Streamlit frontend for the bank churn prediction model."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.predictor import predictor
from app.schemas import CustomerInput
from src.utils.config_loader import settings

S = settings
INPUT_FIELDS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
DEFAULT_PROFILE = {
    "CreditScore": 650, "Geography": "France", "Gender": "Male", "Age": 42,
    "Tenure": 5, "Balance": 75000.0, "NumOfProducts": 2, "HasCrCard": 1,
    "IsActiveMember": 1, "EstimatedSalary": 98000.0,
}
PROFILE_PRESETS = {
    "Typical customer": DEFAULT_PROFILE,
    "Active, multi-product": {
        **DEFAULT_PROFILE, "CreditScore": 720, "Age": 35, "Tenure": 8,
        "Balance": 120000.0, "NumOfProducts": 3, "IsActiveMember": 1,
    },
    "Low engagement": {
        **DEFAULT_PROFILE, "CreditScore": 590, "Geography": "Germany", "Age": 58,
        "Tenure": 2, "Balance": 0.0, "NumOfProducts": 1, "IsActiveMember": 0,
    },
}

st.set_page_config(page_title="Bank Churn Predictor", page_icon=":bar_chart:", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model():
    predictor.load()
    return predictor


def _risk_color(segment: str) -> str:
    return {
        "Critical": "#A32D2D", "High": "#D85A30", "Medium": "#C98612",
        "Low": "#3B6D11",
    }.get(segment, "#185FA5")


def _apply_preset(name: str) -> None:
    for field, value in PROFILE_PRESETS[name].items():
        st.session_state[f"profile_{field}"] = value


def _profile_from_inputs() -> dict:
    return {
        "CreditScore": int(st.session_state.profile_CreditScore),
        "Geography": st.session_state.profile_Geography,
        "Gender": st.session_state.profile_Gender,
        "Age": int(st.session_state.profile_Age),
        "Tenure": int(st.session_state.profile_Tenure),
        "Balance": float(st.session_state.profile_Balance),
        "NumOfProducts": int(st.session_state.profile_NumOfProducts),
        "HasCrCard": int(st.session_state.profile_HasCrCard),
        "IsActiveMember": int(st.session_state.profile_IsActiveMember),
        "EstimatedSalary": float(st.session_state.profile_EstimatedSalary),
    }


def _customer_form() -> dict | None:
    preset_col, _ = st.columns([1, 2])
    preset_col.selectbox(
        "Start from a sample profile",
        list(PROFILE_PRESETS),
        key="profile_preset",
        on_change=lambda: _apply_preset(st.session_state.profile_preset),
    )

    for field, value in DEFAULT_PROFILE.items():
        st.session_state.setdefault(f"profile_{field}", value)

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            st.number_input("Credit Score", 300, 900, key="profile_CreditScore")
            st.selectbox("Geography", ["France", "Germany", "Spain"], key="profile_Geography")
            st.number_input("Tenure (years)", 0, 10, key="profile_Tenure")
            st.selectbox("Products Held", [1, 2, 3, 4], key="profile_NumOfProducts")
            st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x else "No", key="profile_HasCrCard")
        with right:
            st.number_input("Age", 18, 100, key="profile_Age")
            st.selectbox("Gender", ["Male", "Female"], key="profile_Gender")
            st.number_input("Balance (EUR)", min_value=0.0, step=1000.0, key="profile_Balance")
            st.number_input("Estimated Salary (EUR)", min_value=0.0, step=1000.0, key="profile_EstimatedSalary")
            st.selectbox("Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No", key="profile_IsActiveMember")
        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True, type="primary")
    return _profile_from_inputs() if submitted else None


def _show_single_result(model, profile: dict, threshold: float) -> None:
    try:
        validated = CustomerInput(**profile)
        result = model.predict_one(validated.model_dump())
    except ValidationError as exc:
        st.error("Input validation failed.")
        st.code(str(exc))
        return
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    probability = result["churn_probability"]
    flagged = probability >= threshold
    label = "Flagged for retention review" if flagged else "Below review threshold"
    risk_color = _risk_color(result["risk_segment"])

    st.metric("Churn probability", f"{probability:.1%}", label)
    st.progress(min(max(float(probability), 0.0), 1.0), text="Probability scale · 0% to 100%")
    st.caption(f"Review threshold: {threshold:.0%}. The threshold changes the flag; risk bands remain the model's standard bands.")
    st.markdown(
        f"<div style='padding:0.6rem 0.9rem;border-radius:0.75rem;"
        f"background:{risk_color}14;color:{risk_color};font-weight:600;'>"
        f"{result['risk_segment']} risk</div>", unsafe_allow_html=True,
    )

    metrics = st.columns(3)
    for col, label, value in zip(metrics, ["Recency", "Frequency", "Monetary"], [result["r_score"], result["f_score"], result["m_score"]]):
        col.metric(label, f"{value} / 5")

    st.write(f"**RFM segment:** {result['rfm_segment']}  ·  **RFM score:** {result['rfm_score']} / 15")
    st.write(f"**Retention priority:** {result['retention_priority']} / 5")
    st.info(result["recommendation"])

    with st.expander("Explore what-if scenarios"):
        st.caption("These are model comparisons for changed inputs, not causal estimates of what an intervention will achieve.")
        scenarios = [{"Scenario": "Current profile", **profile}]
        if profile["IsActiveMember"] == 1:
            active_scenario = {**profile, "IsActiveMember": 0}
            scenarios.append({"Scenario": "If inactive", **active_scenario})
        else:
            active_scenario = {**profile, "IsActiveMember": 1}
            scenarios.append({"Scenario": "If active", **active_scenario})
        if profile["NumOfProducts"] < 4:
            scenarios.append({"Scenario": "+1 product", **{**profile, "NumOfProducts": profile["NumOfProducts"] + 1}})
        else:
            scenarios.append({"Scenario": "1 fewer product", **{**profile, "NumOfProducts": 3}})
        scenario_results = model.predict_batch(scenarios)
        chart = pd.DataFrame({
            "Churn probability": [item["churn_probability"] for item in scenario_results]
        }, index=[item["Scenario"] for item in scenarios])
        st.bar_chart(chart, y="Churn probability", y_label="Churn probability", color="#4C9BE8")
        table = pd.DataFrame({
            "Scenario": [item["Scenario"] for item in scenarios],
            "Churn probability": [f"{item['churn_probability']:.1%}" for item in scenario_results],
            "Risk band": [item["risk_segment"] for item in scenario_results],
            "RFM segment": [item["rfm_segment"] for item in scenario_results],
        })
        st.dataframe(table, hide_index=True, use_container_width=True)


def _show_batch_scoring(model, threshold: float) -> None:
    st.markdown("Upload a CSV containing these columns: " + ", ".join(f"`{field}`" for field in INPUT_FIELDS))
    uploaded = st.file_uploader("Customer profiles CSV", type=["csv"], key="batch_upload")
    if uploaded is None:
        st.caption("Extra columns are ignored. Invalid rows are skipped and reported.")
        return
    try:
        source = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return
    missing = [field for field in INPUT_FIELDS if field not in source.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return
    st.caption(f"Loaded {len(source):,} rows. Previewing the first 5.")
    st.dataframe(source.head(), use_container_width=True)
    score_clicked = st.button("Score customers", type="primary", key="score_batch")
    if not score_clicked:
        prior = st.session_state.get("batch_results")
        if prior is not None:
            prior["retention_review"] = prior["churn_probability"] >= threshold
            _render_batch_results(prior, threshold)
        return

    valid_profiles, valid_indices, row_errors = [], [], []
    for idx, row in source.iterrows():
        try:
            profile = CustomerInput(**row[INPUT_FIELDS].to_dict()).model_dump()
            valid_profiles.append(profile)
            valid_indices.append(idx)
        except ValidationError as exc:
            row_errors.append({"row": int(idx) + 2, "error": str(exc.errors()[0]["msg"])})
    if not valid_profiles:
        st.error("No valid customer rows to score.")
        return

    try:
        predictions = model.predict_batch(valid_profiles)
    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")
        return

    scored = source.loc[valid_indices].reset_index(drop=True).copy()
    scored["churn_probability"] = [item["churn_probability"] for item in predictions]
    scored["risk_segment"] = [item["risk_segment"] for item in predictions]
    scored["rfm_segment"] = [item["rfm_segment"] for item in predictions]
    scored["rfm_score"] = [item["rfm_score"] for item in predictions]
    scored["retention_priority"] = [item["retention_priority"] for item in predictions]
    scored["retention_review"] = scored["churn_probability"] >= threshold

    st.session_state.batch_results = scored
    st.session_state.batch_errors = row_errors
    st.session_state.batch_threshold = threshold

    if row_errors:
        st.warning(f"Skipped {len(row_errors)} invalid row(s). Row numbers include the CSV header.")
        st.dataframe(pd.DataFrame(row_errors), hide_index=True, use_container_width=True)
    _render_batch_results(scored, threshold)


def _render_batch_results(scored: pd.DataFrame, threshold: float) -> None:
    if scored.empty:
        return
    flagged = int(scored["retention_review"].sum())
    top = scored.loc[scored["churn_probability"].idxmax()]
    kpis = st.columns(4)
    kpis[0].metric("Customers scored", f"{len(scored):,}")
    kpis[1].metric("Flagged for review", f"{flagged:,}", f"{flagged / len(scored):.1%}")
    kpis[2].metric("Average churn probability", f"{scored['churn_probability'].mean():.1%}")
    kpis[3].metric("Highest score", f"{top['churn_probability']:.1%}")

    chart_col, geo_col = st.columns(2)
    with chart_col:
        st.subheader("Customers by risk band")
        counts = scored["risk_segment"].value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
        st.bar_chart(counts.rename("Customers"), color="#4C9BE8")
    with geo_col:
        st.subheader("Average probability by country")
        if "Geography" in scored.columns:
            geo = scored.groupby("Geography")["churn_probability"].mean().sort_values(ascending=False)
            st.bar_chart(geo.rename("Average churn probability"), color="#D85A30")

    st.subheader("Prioritised customer list")
    ordered = scored.sort_values(["retention_review", "churn_probability"], ascending=[False, False])
    st.dataframe(ordered, hide_index=True, use_container_width=True)
    st.download_button(
        "Download scored CSV",
        data=ordered.to_csv(index=False).encode("utf-8"),
        file_name="bank_churn_scored_customers.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(f"Review flags use the current {threshold:.0%} threshold.")


def main() -> None:
    st.title("Bank Churn Predictor")
    st.caption("Interactive churn scoring, RFM context, and retention prioritisation")

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
        threshold = st.slider(
            "Retention review threshold",
            min_value=0.05, max_value=0.95, value=float(model.threshold), step=0.01,
            help="Customers at or above this predicted probability are flagged for retention review.",
        )
        st.write(f"Features: `{len(model.feature_names)}`")
        st.caption("Threshold affects the review flag, not the model probability.")

    single_tab, batch_tab = st.tabs(["Single customer", "Batch scoring"])
    with single_tab:
        form_col, result_col = st.columns([1.1, 0.9], gap="large")
        with form_col:
            st.subheader("Customer Profile")
            profile = _customer_form()
            if profile is not None:
                st.session_state.last_profile = profile
        with result_col:
            st.subheader("Prediction")
            displayed_profile = profile or st.session_state.get("last_profile")
            if displayed_profile is not None:
                _show_single_result(model, displayed_profile, threshold)
            else:
                st.caption("Choose a sample profile or enter customer details, then select Predict Churn Risk.")
    with batch_tab:
        st.subheader("Score a customer portfolio")
        _show_batch_scoring(model, threshold)


if __name__ == "__main__":
    main()
