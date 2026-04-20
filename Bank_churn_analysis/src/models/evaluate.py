"""
src/models/evaluate.py
----------------------
Full evaluation suite: classification metrics, ROC/PR curves,
confusion matrix, score distribution, SHAP explanations.
All plots saved to artifacts/plots/ and logged to MLflow.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    RocCurveDisplay, PrecisionRecallDisplay,
    average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from src.utils.config_loader import settings
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
log = get_logger(__name__)
S = settings

PLOT_DIR = Path("artifacts/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
PALETTE = {"retained": "#3B6D11", "churned": "#A32D2D",
           "blue": "#185FA5", "amber": "#854F0B"}


def evaluate(pipe, X_test, y_test, X_train,
             name: str = "best_model") -> Dict[str, float]:
    threshold = float(S.model.decision_threshold)
    y_proba   = pipe.predict_proba(X_test)[:, 1]
    y_pred    = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc":       round(roc_auc_score(y_test, y_proba), 4),
        "avg_precision": round(average_precision_score(y_test, y_proba), 4),
        "f1":            round(f1_score(y_test, y_pred), 4),
        "precision":     round(precision_score(y_test, y_pred), 4),
        "recall":        round(recall_score(y_test, y_pred), 4),
        "threshold":     threshold,
    }
    log.info(f"Evaluation metrics: {metrics}")
    mlflow.log_metrics(metrics)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    rp = Path("artifacts/classification_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(str(rp))

    # Plots
    _roc_plot(pipe, X_test, y_test, name)
    _pr_plot(pipe, X_test, y_test, name)
    _cm_plot(y_test, y_pred, name)
    _score_dist(y_proba, y_test, name, threshold)
    _shap_analysis(pipe, X_train, X_test, name)

    return metrics


def _roc_plot(pipe, X_test, y_test, name):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, color=PALETTE["blue"])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(f"ROC Curve — {name}", fontsize=12)
    _save(fig, f"{name}_roc.png")


def _pr_plot(pipe, X_test, y_test, name):
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=ax, color=PALETTE["amber"])
    ax.set_title(f"Precision-Recall — {name}", fontsize=12)
    _save(fig, f"{name}_pr.png")


def _cm_plot(y_test, y_pred, name):
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn",
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"], ax=ax)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12)
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    _save(fig, f"{name}_cm.png")


def _score_dist(y_proba, y_test, name, threshold):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color, lbl in [(0, PALETTE["retained"], "Retained"),
                               (1, PALETTE["churned"],  "Churned")]:
        ax.hist(y_proba[y_test == label], bins=40, alpha=0.6, color=color,
                label=f"{lbl} (n={int((y_test==label).sum()):,})",
                density=True, edgecolor="white")
    ax.axvline(threshold, color="black", lw=1.5, linestyle=":",
               label=f"Threshold={threshold}")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — {name}", fontsize=12)
    ax.legend(fontsize=9)
    _save(fig, f"{name}_score_dist.png")


def _shap_analysis(pipe, X_train, X_test, name, n=500):
    try:
        Xt = _pre_clf(pipe, X_test[:n])
        Xb = _pre_clf(pipe, X_train)
        clf = pipe.named_steps["clf"]
        try:
            explainer   = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(Xt)
        except Exception:
            bg          = shap.kmeans(Xb, 50)
            explainer   = shap.KernelExplainer(clf.predict_proba, bg)
            shap_values = explainer.shap_values(Xt)

        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        feat_names = X_test.columns.tolist()

        fig, _ = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, Xt, feature_names=feat_names, show=False)
        plt.title(f"SHAP Summary — {name}", fontsize=12, pad=12)
        _save(fig, f"{name}_shap_summary.png")

        fig2, _ = plt.subplots(figsize=(8, 6))
        shap.summary_plot(sv, Xt, feature_names=feat_names,
                          plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance — {name}", fontsize=12, pad=12)
        _save(fig2, f"{name}_shap_bar.png")

        log.info("SHAP analysis complete")
    except Exception as e:
        log.warning(f"SHAP failed: {e}")


def _pre_clf(pipe, X):
    Xt = X.copy().values if hasattr(X, "values") else X.copy()
    for step_name, step in pipe.steps:
        if step_name == "clf":
            break
        if hasattr(step, "transform"):
            Xt = step.transform(Xt)
    return Xt


def _save(fig, filename: str):
    path = PLOT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path))


def run(pipe, X_test, y_test, X_train, name: str = "best_model") -> dict:
    return evaluate(pipe, X_test, y_test, X_train, name)
