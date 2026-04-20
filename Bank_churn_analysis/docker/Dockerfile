# ── Build stage ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Bank Churn Analysis"
LABEL description="Bank customer churn prediction API"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY src/           ./src/
COPY app/           ./app/
COPY configs/       ./configs/
COPY .env           ./.env

# Runtime directories — owned by appuser
RUN mkdir -p artifacts data/raw data/processed mlruns logs \
 && chown -R appuser:appuser /app

# Copy pre-trained model artifacts if they exist
COPY artifacts/     ./artifacts/

USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
