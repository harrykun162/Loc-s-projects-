"""
scripts/train_pipeline.py
-------------------------
End-to-end ML training pipeline.  Run this once before starting
the FastAPI app to produce artifacts/best_model.pkl.

Pipeline steps
--------------
  1. Ingest & validate      src/data/ingest.py
  2. Feature engineering    src/features/engineer.py
  3. Model training         src/models/train.py
  4. Evaluation & SHAP      src/models/evaluate.py

Usage
-----
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --no-tune
    python scripts/train_pipeline.py --data path/to/custom.csv
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from src.data        import ingest
from src.features    import engineer
from src.models      import train, evaluate as eval_module
from src.utils.config_loader import settings
from src.utils.logger import get_logger

log = get_logger("train_pipeline")
S = settings


def run(data_path: str = None, tune: bool = True, n_trials: int = 30):
    mlflow.set_tracking_uri(S.mlflow.tracking_uri)
    mlflow.set_experiment(S.mlflow.experiment_name)

    with mlflow.start_run(run_name="bank_churn_training_pipeline"):
        t0 = time.time()

        # ------------------------------------------------------------------
        # Step 1 — Ingest & validate
        # ------------------------------------------------------------------
        log.info("=" * 55)
        log.info("STEP 1 — Data Ingest & Validation")
        log.info("=" * 55)
        with mlflow.start_run(run_name="1_ingest", nested=True):
            df = ingest.run(data_path)

        # ------------------------------------------------------------------
        # Step 2 — Feature engineering
        #   Builds interaction features, binned features, OHE.
        #   RFM segment columns are NOT added here — see rfm_engineering.sql.
        # ------------------------------------------------------------------
        log.info("=" * 55)
        log.info("STEP 2 — Feature Engineering")
        log.info("=" * 55)
        with mlflow.start_run(run_name="2_features", nested=True):
            X, y = engineer.run(df)

        # ------------------------------------------------------------------
        # Step 3 — Train / test split + model training
        # ------------------------------------------------------------------
        log.info("=" * 55)
        log.info("STEP 3 — Model Training")
        log.info("=" * 55)
        with mlflow.start_run(run_name="3_train", nested=True):
            X_train, X_test, y_train, y_test = train.split(X, y)
            best_name, best_pipe, all_models = train.run(
                X_train, y_train, X_test, y_test,
                tune=tune, n_trials=n_trials,
            )

        # ------------------------------------------------------------------
        # Step 4 — Evaluate + SHAP
        # ------------------------------------------------------------------
        log.info("=" * 55)
        log.info("STEP 4 — Evaluation & SHAP")
        log.info("=" * 55)
        with mlflow.start_run(run_name="4_evaluate", nested=True):
            metrics = eval_module.run(best_pipe, X_test, y_test, X_train, best_name)

        elapsed = time.time() - t0
        mlflow.log_metric("total_pipeline_seconds", round(elapsed, 1))

        log.info("=" * 55)
        log.info(f"Pipeline complete in {elapsed:.1f}s")
        log.info(f"Best model : {best_name}")
        log.info(f"ROC-AUC    : {metrics.get('roc_auc', 'N/A')}")
        log.info(f"F1         : {metrics.get('f1', 'N/A')}")
        log.info(f"Model saved: artifacts/best_model.pkl")
        log.info(f"MLflow UI  : mlflow ui --port 5000")
        log.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default=None, help="Path to raw CSV")
    parser.add_argument("--no-tune",  dest="tune", action="store_false")
    parser.add_argument("--trials",   type=int, default=30)
    args = parser.parse_args()
    run(data_path=args.data, tune=args.tune, n_trials=args.trials)