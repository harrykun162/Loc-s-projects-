# Bank Churn Analysis — End-to-End ML Pipeline

RFM-enhanced churn prediction for European bank customers,
served via a FastAPI localhost website with a full web UI.
The project can also be deployed with Streamlit for a simpler hosted frontend.

---

## Project structure
```
Bank_churn_analysis/
│
├── data/
│   ├── raw/                    ← Drop bank_customers.csv here
│   ├── processed/              ← Auto-generated parquet files
│   └── external/               ← Any reference data
│
├── notebooks/                  ← EDA.ipynb (exploratory work)
│
├── src/
│   ├── data/ingest.py          ← Step 1: load, validate, persist
│   ├── features/rfm.py         ← Step 2: RFM proxy engineering
│   ├── features/engineer.py    ← Step 3: ML feature matrix
│   ├── models/train.py         ← Step 4: LR + RF + XGBoost + Optuna
│   ├── models/evaluate.py      ← Step 5: metrics, ROC, SHAP
│   └── utils/
│       ├── config_loader.py    ← YAML + .env settings singleton
│       └── logger.py           ← Structured rotating logger
│
├── app/
│   ├── main.py                 ← FastAPI app (all routes + web UI)
│   ├── predictor.py            ← Inference engine (loaded once)
│   └── schemas.py              ← Pydantic request/response models
│
├── configs/
│   └── config.yaml             ← All project settings
│
├── scripts/
│   ├── train_pipeline.py       ← Full training run
│   └── start_app.py            ← App launcher with env awareness
│
├── tests/
│   ├── test_api.py             ← FastAPI endpoint tests
│   └── test_features.py        ← Feature engineering unit tests
│
├── great_expectations/
│   └── validate.py             ← Data quality expectation suite
│
├── docker/
│   ├── Dockerfile              ← Multi-stage production image
│   └── docker-compose.yml      ← API + MLflow services
│
├── .github/workflows/
│   └── ci.yml                  ← GitHub Actions: lint, test, docker
│
├── .env                        ← Environment variables
├── Makefile                    ← One-command workflows
└── requirements.txt
```

---

## Quickstart (5 steps)

```bash
# 1. Install dependencies
make install

# 2. Place your dataset
cp bank_churn_RFM.csv data/raw/

# 3. Validate data quality
make validate

# 4. Train all models
make train-fast          # fast (no Optuna tuning)
make train               # full (with Optuna, ~5–10 min)

# 5. Start the web app
make serve
make serve-streamlit
```

Then open your browser at **http://localhost:8000/ui**
or **http://localhost:8501** for Streamlit.

---

## URLs

| URL | Description |
|-----|-------------|
| `http://localhost:8000/ui`    | Interactive web UI — predict churn for any customer |
| `http://localhost:8000/docs`  | Auto-generated Swagger API documentation |
| `http://localhost:8000/redoc` | ReDoc API documentation |
| `http://localhost:8000/health`| Health check (JSON) |
| `http://localhost:5000`       | MLflow experiment tracking UI |

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health`        | Liveness probe |
| `GET`  | `/model/info`    | Model metadata |
| `POST` | `/predict`       | Single customer prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 5,000) |
| `GET`  | `/ui`            | Web UI |

### Single prediction example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 42,
    "Tenure": 5,
    "Balance": 75000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 98000.0
  }'
```

Response:
```json
{
  "churn_probability":  0.2341,
  "churn_predicted":    0,
  "risk_segment":       "Low",
  "rfm_score":          11,
  "rfm_segment":        "Loyal Customer",
  "retention_priority": 4,
  "r_score":            4,
  "f_score":            3,
  "m_score":            4,
  "recommendation":     "Upsell opportunity — cross-sell one additional product."
}
```

---

## Running tests

```bash
make test           # all tests
make test-cov       # with HTML coverage report
pytest tests/test_features.py -v   # feature tests only
pytest tests/test_api.py -v        # API tests only
```

---

## Docker

```bash
make docker-build   # build image
make docker-up      # start API + MLflow
make docker-down    # stop services
```

---

## MLflow

```bash
make mlflow-ui      # open tracking UI at http://localhost:5000
```

MLflow tracks every training run with:
- All hyperparameters (XGBoost, RF, LR)
- CV metrics (ROC-AUC, F1, precision, recall)
- SHAP plots, confusion matrix, ROC/PR curves
- Model artifacts registered in Model Registry

---

## Make commands reference

```bash
make help           # show all commands
make install        # install dependencies
make validate       # run Great Expectations suite
make train          # full training with Optuna
make train-fast     # training without tuning
make serve          # start dev server (hot reload)
make serve-prod     # start production server
make test           # run all tests
make test-cov       # tests + coverage
make lint           # ruff linter
make mlflow-ui      # open MLflow dashboard
make docker-build   # build Docker image
make docker-up      # start all services
make serve-streamlit # start Streamlit on port 8501
make clean          # remove caches
```

---

## Streamlit deployment

Use `app/streamlit_app.py` as the Streamlit entrypoint.

### Local run

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Streamlit Community Cloud settings

- Main file path: `app/streamlit_app.py`
- Install command: `pip install -r requirements.txt`
- Python version: `3.11` is a safe target for this stack

### Required deployment artifacts

- `artifacts/best_model.pkl`
- `artifacts/feature_names.json`
- `configs/config.yaml`

### Recommended app structure for Streamlit

- Keep `app/predictor.py` as the inference layer.
- Use Streamlit as the frontend instead of the current `/ui` HTML route.
- Load the model with `st.cache_resource` so it initializes once per app process.
- Validate user inputs with `CustomerInput` before scoring.
