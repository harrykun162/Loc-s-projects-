"""
src/models/train.py
-------------------
Train Logistic Regression (baseline), Random Forest, and XGBoost
with SMOTE, StratifiedKFold CV, and optional Optuna tuning.
Saves the best model to artifacts/best_model.pkl.
"""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.config_loader import settings
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
log = get_logger(__name__)
S = settings

SCORING = ["roc_auc", "average_precision", "f1", "precision", "recall"]


def split(X: pd.DataFrame, y: pd.Series) -> Tuple:
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(S.model.test_size),
        random_state=int(S.model.random_state),
    )
    train_idx, test_idx = next(sss.split(X, y))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def _pipeline(estimator) -> ImbPipeline:
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(k_neighbors=int(S.model.smote_k),
                         random_state=int(S.model.random_state))),
        ("clf",    estimator),
    ])


def _cv(pipe, X_train, y_train, name: str) -> Dict[str, float]:
    cv = StratifiedKFold(
        n_splits=int(S.model.cv_folds), shuffle=True,
        random_state=int(S.model.random_state),
    )
    res = cross_validate(pipe, X_train, y_train, cv=cv,
                         scoring=SCORING, n_jobs=-1, return_train_score=False)
    scores = {}
    for m in SCORING:
        scores[f"cv_{m}_mean"] = round(float(res[f"test_{m}"].mean()), 4)
        scores[f"cv_{m}_std"]  = round(float(res[f"test_{m}"].std()),  4)
    mlflow.log_metrics({f"{name}_{k}": v for k, v in scores.items()})
    log.info(f"{name} CV ROC-AUC: {scores['cv_roc_auc_mean']:.4f} "
             f"± {scores['cv_roc_auc_std']:.4f}")
    return scores


def train_logistic_regression(X_train, y_train) -> ImbPipeline:
    log.info("Training Logistic Regression baseline")
    lr   = LogisticRegression(C=1.0, max_iter=1000,
                               class_weight="balanced",
                               random_state=int(S.model.random_state))
    pipe = _pipeline(lr)
    _cv(pipe, X_train, y_train, "lr")
    pipe.fit(X_train, y_train)
    _save(pipe, "logistic_regression")
    return pipe


def train_random_forest(X_train, y_train) -> ImbPipeline:
    log.info("Training Random Forest")
    rf   = RandomForestClassifier(n_estimators=300, max_depth=8,
                                   class_weight="balanced",
                                   random_state=int(S.model.random_state),
                                   n_jobs=-1)
    pipe = _pipeline(rf)
    _cv(pipe, X_train, y_train, "rf")
    pipe.fit(X_train, y_train)
    _save(pipe, "random_forest")
    return pipe


def train_xgboost(X_train, y_train, tune: bool = True, n_trials: int = 30) -> ImbPipeline:
    log.info(f"Training XGBoost (tune={tune})")
    xgb_cfg = dict(S.model.xgb_params)

    if tune:
        best_p = _optuna(X_train, y_train, n_trials)
        xgb_cfg.update(best_p)

    xgb  = XGBClassifier(**xgb_cfg, eval_metric="auc",
                          use_label_encoder=False, verbosity=0)
    pipe = _pipeline(xgb)
    _cv(pipe, X_train, y_train, "xgb")
    pipe.fit(X_train, y_train)
    _save(pipe, "xgboost")
    return pipe


def _optuna(X_train, y_train, n_trials: int) -> dict:
    cv = StratifiedKFold(n_splits=3, shuffle=True,
                         random_state=int(S.model.random_state))

    def objective(trial):
        p = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 8.0),
        }
        pipe = _pipeline(XGBClassifier(**p, eval_metric="auc",
                                        use_label_encoder=False, verbosity=0,
                                        random_state=int(S.model.random_state)))
        return cross_validate(pipe, X_train, y_train, cv=cv,
                              scoring="roc_auc", n_jobs=-1)["test_score"].mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info(f"Optuna best AUC: {study.best_value:.4f}")
    mlflow.log_metric("optuna_best_auc", round(study.best_value, 4))
    return study.best_params


def pick_best(models: dict, X_test, y_test) -> Tuple[str, object]:
    """Return (name, pipeline) of model with highest test ROC-AUC."""
    from sklearn.metrics import roc_auc_score
    scores = {}
    for name, pipe in models.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        scores[name] = roc_auc_score(y_test, proba)
        mlflow.log_metric(f"{name}_test_roc_auc", round(scores[name], 4))
        log.info(f"{name} test ROC-AUC: {scores[name]:.4f}")
    best = max(scores, key=scores.get)
    log.info(f"Best model: {best} ({scores[best]:.4f})")
    return best, models[best]


def _save(pipe, name: str) -> Path:
    out = Path("artifacts") / f"{name}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)
    mlflow.log_artifact(str(out))
    return out


def run(X_train, y_train, X_test, y_test,
        tune: bool = True, n_trials: int = 30) -> Tuple[str, object, dict]:
    models = {
        "logistic_regression": train_logistic_regression(X_train, y_train),
        "random_forest":       train_random_forest(X_train, y_train),
        "xgboost":             train_xgboost(X_train, y_train, tune, n_trials),
    }
    best_name, best_pipe = pick_best(models, X_test, y_test)

    # Save champion model as best_model.pkl
    best_path = Path(S.paths.best_model)
    joblib.dump(best_pipe, best_path)
    mlflow.log_artifact(str(best_path))
    log.info(f"Champion model saved -> {best_path}")

    return best_name, best_pipe, models
