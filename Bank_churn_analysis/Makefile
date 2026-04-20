# ============================================================
#  Bank Churn Analysis — Makefile
#  Usage: make <target>
#  All commands run from the project root.
# ============================================================

.PHONY: help install validate train serve serve-streamlit test lint clean docker-build docker-up

PYTHON  := python
UVICORN := uvicorn
PORT    := 8000

# ── Default target ────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Bank Churn Analysis — available commands"
	@echo "  ─────────────────────────────────────────"
	@echo "  make install        Install all Python dependencies"
	@echo "  make validate       Run Great Expectations data validation"
	@echo "  make train          Train all models (with Optuna tuning)"
	@echo "  make train-fast     Train all models (no tuning)"
	@echo "  make serve          Start the FastAPI app on localhost:$(PORT)"
	@echo "  make serve-streamlit Start the Streamlit app on localhost:8501"
	@echo "  make serve-prod     Start in production mode (4 workers)"
	@echo "  make test           Run the full test suite"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make lint           Run ruff linter"
	@echo "  make mlflow-ui      Open MLflow tracking UI on port 5000"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-up      Start API + MLflow via docker-compose"
	@echo "  make docker-down    Stop docker-compose services"
	@echo "  make clean          Remove generated artifacts and caches"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# ── Data validation ───────────────────────────────────────────────────────
validate:
	$(PYTHON) great_expectations/validate.py --strict
	@echo "✓ Data validation passed"

# ── Training ──────────────────────────────────────────────────────────────
train: validate
	$(PYTHON) scripts/train_pipeline.py
	@echo "✓ Training complete — model saved to artifacts/best_model.pkl"

train-fast: validate
	$(PYTHON) scripts/train_pipeline.py --no-tune
	@echo "✓ Fast training complete"

# ── App ───────────────────────────────────────────────────────────────────
serve:
	@echo "Starting app on http://localhost:$(PORT)"
	@echo "  UI:   http://localhost:$(PORT)/ui"
	@echo "  Docs: http://localhost:$(PORT)/docs"
	$(UVICORN) app.main:app --host 0.0.0.0 --port $(PORT) --reload

serve-streamlit:
	streamlit run app/streamlit_app.py

serve-prod:
	$(UVICORN) app.main:app --host 0.0.0.0 --port $(PORT) --workers 4

# ── Testing ───────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov=app \
	       --cov-report=term-missing --cov-report=html:artifacts/coverage
	@echo "✓ Coverage report → artifacts/coverage/index.html"

# ── Linting ───────────────────────────────────────────────────────────────
lint:
	ruff check src/ app/ scripts/ tests/ --ignore E501
	@echo "✓ Lint passed"

# ── MLflow ────────────────────────────────────────────────────────────────
mlflow-ui:
	mlflow ui --port 5000 --backend-store-uri mlruns

# ── Docker ────────────────────────────────────────────────────────────────
docker-build:
	docker build -f docker/Dockerfile -t bank-churn-api:latest .
	@echo "✓ Docker image built: bank-churn-api:latest"

docker-up:
	docker compose -f docker/docker-compose.yml up -d
	@echo "✓ Services started"
	@echo "  API:    http://localhost:$(PORT)/ui"
	@echo "  MLflow: http://localhost:5000"

docker-down:
	docker compose -f docker/docker-compose.yml down

# ── Clean ─────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "*.egg-info"   -exec rm -rf {} + 2>/dev/null; true
	@echo "✓ Cleaned pycaches"
