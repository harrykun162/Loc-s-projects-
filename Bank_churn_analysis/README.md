# Bank Churn Analysis

## Project Overview

This project is an end-to-end machine learning application for predicting customer churn in a retail banking context.

The main goal is to help a bank identify customers who are likely to leave, understand their customer value profile, and recommend a practical retention action. The project combines:

- Churn prediction using supervised machine learning
- RFM-style customer segmentation using banking behavior proxies
- A local Streamlit UI for interactive predictions
- A FastAPI service for API-based predictions
- MLflow tracking for experiment metrics and model artifacts
- Optional Docker serving for the trained model

The final application allows a user to enter a customer profile and receive:

- Predicted churn probability
- Churn risk tier
- RFM score breakdown
- RFM customer segment
- Retention priority
- Recommended retention action

## Business Aim

Customer churn is expensive for banks because acquiring a new customer usually costs more than retaining an existing one. This project aims to support proactive retention by forecasting whether a customer is likely to churn and by adding RFM-style context to explain the customer's relationship quality.

In this project, RFM is adapted for bank customer data:

- Recency proxy: customer activity and tenure
- Frequency proxy: number of products, active membership, and credit card ownership
- Monetary proxy: account balance and estimated salary

The model predicts churn probability, while the RFM logic helps translate the prediction into a more business-friendly customer segment and action recommendation.

## Project Structure

```text
Bank_churn_analysis/
|
|-- app/
|   |-- main.py              # FastAPI app and browser UI
|   |-- predictor.py         # Inference layer used by FastAPI and Streamlit
|   |-- schemas.py           # Request and response validation
|   |-- streamlit_app.py     # Streamlit UI
|
|-- configs/
|   |-- config.yaml          # Project paths, model settings, RFM weights
|
|-- data/
|   |-- raw/                 # Place the raw CSV dataset here
|   |-- processed/           # Generated processed data
|
|-- docker/
|   |-- Dockerfile
|   |-- docker-compose.yml
|
|-- scripts/
|   |-- train_pipeline.py    # Full training pipeline
|   |-- start_app.py         # FastAPI launcher
|
|-- src/
|   |-- data/ingest.py       # Load and validate data
|   |-- features/engineer.py # Feature engineering
|   |-- features/rfm.py      # RFM scoring and segmentation
|   |-- models/train.py      # Model training
|   |-- models/evaluate.py   # Evaluation and plots
|   |-- utils/               # Config and logging helpers
|
|-- tests/                   # Unit and API tests
|-- artifacts/               # Generated trained models and reports
|-- requirements.txt
|-- README.md
```

## Requirements

Recommended environment:

- Python 3.11
- Windows PowerShell, macOS terminal, or Linux shell
- Docker Desktop, only if you want to serve the app with Docker

Install Python dependencies from:

```text
requirements.txt
```

## Quickstart: Run Locally With Streamlit

The recommended beginner-friendly workflow is:

1. Install dependencies
2. Place the dataset in the expected folder
3. Train the model pipeline
4. Launch the Streamlit UI

Run all commands from the `Bank_churn_analysis` folder.

### 1. Move Into The Project Folder

If you are currently in the parent repository folder:

```powershell
cd Bank_churn_analysis
```

### 2. Create And Activate A Virtual Environment

On Windows PowerShell:

```powershell
python -m venv ..\venv
..\venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python -m venv ../venv
source ../venv/bin/activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Add The Dataset

Place the raw dataset here:

```text
data/raw/Bank_churn_RFM.csv
```

The default raw data path is configured in:

```text
configs/config.yaml
```

If your file has a different name, either rename it to `Bank_churn_RFM.csv` or update the `paths.raw_data` value in `configs/config.yaml`.

### 5. Train The Model Pipeline

For a faster training run without Optuna tuning:

```powershell
python scripts/train_pipeline.py --no-tune
```

For the full training run with Optuna tuning:

```powershell
python scripts/train_pipeline.py
```

The training pipeline will:

- Load and validate the raw dataset
- Create processed data
- Engineer model features
- Train Logistic Regression, Random Forest, and XGBoost models
- Select the best model by test ROC-AUC
- Save the trained pipeline and feature schema
- Generate evaluation metrics and plots
- Log metrics and artifacts to MLflow

After training, these files should exist:

```text
artifacts/best_model.pkl
artifacts/feature_names.json
data/processed/features.parquet
```

These files are required for local prediction.

### 6. Launch The Streamlit UI

```powershell
streamlit run app/streamlit_app.py
```

Use the form to enter a customer profile and click `Predict Churn Risk`.


## RFM Logic

The RFM component gives business context to the churn prediction.

Current RFM proxies:

- `R_raw`: based on active membership and tenure
- `F_raw`: based on number of products, active membership, and credit card ownership
- `M_raw`: based on balance and estimated salary

The app scores new customers against the processed training population in:

```text
data/processed/features.parquet
```

This is important because RFM scores are relative. If this file is missing, retrain the pipeline:

```powershell
python scripts/train_pipeline.py
```

## Running Tests

Run all tests:

```powershell
pytest tests/ -v --tb=short
```

Run feature tests only:

```powershell
pytest tests/test_features.py -v
```

Run API tests only:

```powershell
pytest tests/test_api.py -v
```


### Docker cannot connect to the Docker API

Open Docker Desktop and wait until it is running. Then test:

```powershell
docker version
docker info
```

After Docker is healthy, rerun:

```powershell
docker compose -f docker/docker-compose.yml up --build -d api
```

## Recommended Local Workflow

For most users, this is the simplest complete workflow:

```powershell
cd Bank_churn_analysis
python -m venv ..\venv
..\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/train_pipeline.py --no-tune
streamlit run app/streamlit_app.py
```
