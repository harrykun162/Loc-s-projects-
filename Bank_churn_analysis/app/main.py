"""
app/main.py
-----------
FastAPI application exposing the bank churn prediction model.

Routes:
  GET  /               → Redirect to /docs
  GET  /health         → Liveness probe
  GET  /model/info     → Model metadata
  POST /predict        → Single customer prediction
  POST /predict/batch  → Batch prediction (up to 5,000 customers)

Start with:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.predictor import predictor
from app.schemas import (
    BatchInput, BatchOutput,
    CustomerInput, HealthResponse,
    ModelInfo, PredictionOutput,
)
from src.utils.config_loader import settings
from src.utils.logger import get_logger

log = get_logger(__name__)
S = settings


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — loading model...")
    try:
        predictor.load()
        log.info("Model loaded successfully")
    except FileNotFoundError as e:
        log.error(str(e))
        # App will start but /predict will return 503
    yield
    log.info("Shutting down")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = S.app.title,
    description = S.app.description,
    version     = S.project.version,
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the interactive UI."""
    return RedirectResponse(url="/ui")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness probe — returns 200 if app is running."""
    return HealthResponse(
        status       = "ok",
        model_loaded = predictor.is_loaded,
        version      = S.project.version,
    )


@app.get("/model/info", response_model=ModelInfo, tags=["System"])
async def model_info():
    """Return metadata about the loaded model."""
    _require_model()
    return ModelInfo(
        model_name    = predictor.model_name,
        model_version = S.project.version,
        threshold     = predictor.threshold,
        features      = len(predictor.feature_names),
        status        = "loaded",
    )


@app.post(
    "/predict",
    response_model = PredictionOutput,
    tags           = ["Prediction"],
    summary        = "Predict churn for a single customer",
)
async def predict(customer: CustomerInput) -> PredictionOutput:
    """
    Submit a single customer's profile and receive:
    - Churn probability (0–1)
    - Binary prediction
    - Risk segment (Low / Medium / High / Critical)
    - Full RFM breakdown (R, F, M scores + segment label)
    - Actionable retention recommendation
    """
    _require_model()
    try:
        result = predictor.predict_one(customer.model_dump())
        return PredictionOutput(**result)
    except Exception as e:
        log.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))


@app.post(
    "/predict/batch",
    response_model = BatchOutput,
    tags           = ["Prediction"],
    summary        = "Predict churn for a batch of customers",
)
async def predict_batch(payload: BatchInput) -> BatchOutput:
    """
    Submit up to 5,000 customers at once.
    Returns individual predictions plus aggregate statistics.
    """
    _require_model()
    try:
        customers = [c.model_dump() for c in payload.customers]
        results   = predictor.predict_batch(customers)
        churners  = sum(r["churn_predicted"] for r in results)
        return BatchOutput(
            total               = len(results),
            predicted_churners  = churners,
            predicted_churn_rate= round(churners / len(results), 4),
            results             = [PredictionOutput(**r) for r in results],
        )
    except Exception as e:
        log.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))


# ---------------------------------------------------------------------------
# Interactive Web UI (localhost website)
# ---------------------------------------------------------------------------

@app.get("/ui", response_class=HTMLResponse,
         include_in_schema=False, tags=["UI"])
async def ui():
    """
    Single-page web interface for the churn prediction model.
    No JavaScript framework required — pure HTML + Tailwind CDN.
    """
    return HTMLResponse(content=_render_ui())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_model():
    if not predictor.is_loaded:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Model not loaded. Run scripts/train_pipeline.py first.",
        )


def _render_ui() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Bank Churn Predictor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: { extend: { colors: {
        brand:  { DEFAULT: '#185FA5', light: '#2a7fd4', dark: '#0e3d6e' },
        danger: { DEFAULT: '#A32D2D', light: '#c94040' },
        safe:   { DEFAULT: '#3B6D11', light: '#4e8f17' },
        warn:   { DEFAULT: '#854F0B', light: '#a8650e' },
      }}}
    }
  </script>
</head>
<body class="bg-gray-50 min-h-screen font-sans">

<!-- Header -->
<header class="bg-brand text-white shadow-md">
  <div class="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
    <div>
      <h1 class="text-xl font-bold tracking-tight">Bank Churn Predictor</h1>
      <p class="text-blue-200 text-sm">RFM-enhanced ML model · European bank customers</p>
    </div>
    <div class="flex gap-3">
      <a href="/docs"   target="_blank"
         class="text-xs bg-white/20 hover:bg-white/30 px-3 py-1.5 rounded-full transition">
        API Docs
      </a>
      <a href="/health" target="_blank"
         class="text-xs bg-white/20 hover:bg-white/30 px-3 py-1.5 rounded-full transition">
        Health
      </a>
    </div>
  </div>
</header>

<main class="max-w-5xl mx-auto px-6 py-8">
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">

    <!-- Input Form -->
    <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
      <h2 class="text-base font-semibold text-gray-800 mb-5">Customer Profile</h2>

      <div class="grid grid-cols-2 gap-4">
        <!-- CreditScore -->
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">
            Credit Score <span class="text-gray-400">(300–900)</span>
          </label>
          <input id="CreditScore" type="number" value="650" min="300" max="900"
                 class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm
                        focus:outline-none focus:ring-2 focus:ring-brand/40"/>
        </div>

        <!-- Age -->
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Age</label>
          <input id="Age" type="number" value="42" min="18" max="100"
                 class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm
                        focus:outline-none focus:ring-2 focus:ring-brand/40"/>
        </div>

        <!-- Geography -->
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Geography</label>
          <select id="Geography"
                  class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm
                         focus:outline-none focus:ring-2 focus:ring-brand/40 bg-white">
            <option>France</option>
            <option>Germany</option>
            <option>Spain</option>
          </select>
        </div>

        <!-- Gender -->
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Gender</label>
          <select id="Gender"
                  class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm
                         focus:outline-none focus:ring-2 focus:ring-brand/40 bg-white">
            <option>Male</option>
            <option>Female</option>
          </select>
        </div>

        <!-- Tenure -->
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">
            Tenure <span class="text-gray-400">(years)</span>
          </label>
          <input id="Tenure" type="number" value="5" min="0" max="10"
                 class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm
                        focus:outline-none focus:ring-2 focus:ring-brand/40"/>
        </div>

        <!-- Balance -->
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">
            Balance <span class="text-gray-400">(€)</span>
          </label>
          <input id="Balance" type="number" value="75000" min="0"
                 class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm
                        focus:outline-none focus:ring-2 focus:ring-brand/40"/>
        </div>

        <!-- NumOfProducts -->
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Products held</label>
          <select id="NumOfProducts"
                  class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm
                         focus:outline-none focus:ring-2 focus:ring-brand/40 bg-white">
            <option value="1">1</option>
            <option value="2" selected>2</option>
            <option value="3">3</option>
            <option value="4">4</option>
          </select>
        </div>

        <!-- EstimatedSalary -->
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">
            Est. Salary <span class="text-gray-400">(€)</span>
          </label>
          <input id="EstimatedSalary" type="number" value="98000" min="0"
                 class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm
                        focus:outline-none focus:ring-2 focus:ring-brand/40"/>
        </div>

        <!-- HasCrCard -->
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Has Credit Card</label>
          <select id="HasCrCard"
                  class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm
                         focus:outline-none focus:ring-2 focus:ring-brand/40 bg-white">
            <option value="1">Yes</option>
            <option value="0">No</option>
          </select>
        </div>

        <!-- IsActiveMember -->
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Active Member</label>
          <select id="IsActiveMember"
                  class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm
                         focus:outline-none focus:ring-2 focus:ring-brand/40 bg-white">
            <option value="1">Yes</option>
            <option value="0">No</option>
          </select>
        </div>
      </div>

      <!-- Submit -->
      <button onclick="predict()"
              class="mt-6 w-full bg-brand hover:bg-brand-light text-white font-medium
                     py-2.5 rounded-xl transition-all duration-150 text-sm
                     active:scale-[0.98] shadow-sm">
        Predict Churn Risk
      </button>

      <!-- Error -->
      <div id="error" class="hidden mt-3 p-3 bg-red-50 border border-red-200
                             rounded-lg text-xs text-red-700"></div>
    </div>

    <!-- Results Panel -->
    <div id="results" class="hidden">
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">

        <!-- Churn probability gauge -->
        <div class="text-center mb-6">
          <div id="risk-badge"
               class="inline-block px-4 py-1.5 rounded-full text-sm font-semibold mb-3">
          </div>
          <div class="relative w-36 h-36 mx-auto">
            <svg class="w-full h-full -rotate-90" viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="50" fill="none"
                      stroke="#f3f4f6" stroke-width="12"/>
              <circle id="prob-ring" cx="60" cy="60" r="50" fill="none"
                      stroke="#185FA5" stroke-width="12"
                      stroke-linecap="round"
                      stroke-dasharray="314"
                      stroke-dashoffset="314"
                      class="transition-all duration-700"/>
            </svg>
            <div class="absolute inset-0 flex flex-col items-center justify-center">
              <span id="prob-pct" class="text-3xl font-bold text-gray-800">—</span>
              <span class="text-xs text-gray-500 mt-0.5">churn prob</span>
            </div>
          </div>
        </div>

        <!-- RFM scores -->
        <div class="grid grid-cols-3 gap-3 mb-5">
          <div class="bg-gray-50 rounded-xl p-3 text-center">
            <div id="r-score" class="text-2xl font-bold text-brand">—</div>
            <div class="text-xs text-gray-500 mt-0.5">Recency</div>
          </div>
          <div class="bg-gray-50 rounded-xl p-3 text-center">
            <div id="f-score" class="text-2xl font-bold text-brand">—</div>
            <div class="text-xs text-gray-500 mt-0.5">Frequency</div>
          </div>
          <div class="bg-gray-50 rounded-xl p-3 text-center">
            <div id="m-score" class="text-2xl font-bold text-brand">—</div>
            <div class="text-xs text-gray-500 mt-0.5">Monetary</div>
          </div>
        </div>

        <!-- Segment + priority -->
        <div class="flex items-center justify-between mb-4 px-1">
          <div>
            <p class="text-xs text-gray-500">RFM Segment</p>
            <p id="rfm-segment" class="text-sm font-semibold text-gray-800 mt-0.5">—</p>
          </div>
          <div class="text-right">
            <p class="text-xs text-gray-500">RFM Score</p>
            <p id="rfm-score" class="text-sm font-semibold text-gray-800 mt-0.5">—</p>
          </div>
          <div class="text-right">
            <p class="text-xs text-gray-500">Priority</p>
            <div id="priority-dots" class="flex gap-0.5 mt-1 justify-end"></div>
          </div>
        </div>

        <!-- Recommendation -->
        <div id="recommendation"
             class="bg-blue-50 border border-blue-100 rounded-xl p-3.5
                    text-xs text-blue-800 leading-relaxed">
        </div>
      </div>
    </div>

    <!-- Placeholder when no result yet -->
    <div id="placeholder"
         class="bg-white rounded-2xl shadow-sm border border-gray-100 p-6
                flex flex-col items-center justify-center text-center min-h-[300px]">
      <div class="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mb-3">
        <svg class="w-6 h-6 text-gray-400" fill="none" stroke="currentColor"
             stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round"
                d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
      </div>
      <p class="text-sm text-gray-500">Fill in the customer profile<br>and click Predict.</p>
    </div>

  </div>
</main>

<script>
  const RISK_STYLES = {
    "Critical": { ring: "#A32D2D", badge: "bg-red-100 text-red-800"   },
    "High"    : { ring: "#D85A30", badge: "bg-orange-100 text-orange-800" },
    "Medium"  : { ring: "#EF9F27", badge: "bg-yellow-100 text-yellow-800" },
    "Low"     : { ring: "#3B6D11", badge: "bg-green-100 text-green-800"  },
  };

  async function predict() {
    const btn   = document.querySelector("button");
    const errEl = document.getElementById("error");
    errEl.classList.add("hidden");
    btn.textContent = "Predicting…";
    btn.disabled    = true;

    const payload = {
      CreditScore    : +document.getElementById("CreditScore").value,
      Geography      : document.getElementById("Geography").value,
      Gender         : document.getElementById("Gender").value,
      Age            : +document.getElementById("Age").value,
      Tenure         : +document.getElementById("Tenure").value,
      Balance        : +document.getElementById("Balance").value,
      NumOfProducts  : +document.getElementById("NumOfProducts").value,
      HasCrCard      : +document.getElementById("HasCrCard").value,
      IsActiveMember : +document.getElementById("IsActiveMember").value,
      EstimatedSalary: +document.getElementById("EstimatedSalary").value,
    };

    try {
      const resp = await fetch("/predict", {
        method  : "POST",
        headers : { "Content-Type": "application/json" },
        body    : JSON.stringify(payload),
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      renderResult(data);

    } catch (e) {
      errEl.textContent = "Error: " + e.message;
      errEl.classList.remove("hidden");
    } finally {
      btn.textContent = "Predict Churn Risk";
      btn.disabled    = false;
    }
  }

  function renderResult(d) {
    document.getElementById("placeholder").style.display = "none";
    document.getElementById("results").classList.remove("hidden");

    const pct   = Math.round(d.churn_probability * 100);
    const style = RISK_STYLES[d.risk_segment] || RISK_STYLES["Low"];

    // Probability ring
    const circ  = 2 * Math.PI * 50;
    const dash  = circ * (1 - d.churn_probability);
    document.getElementById("prob-ring").style.strokeDashoffset = dash;
    document.getElementById("prob-ring").style.stroke = style.ring;
    document.getElementById("prob-pct").textContent   = pct + "%";

    // Risk badge
    const badge = document.getElementById("risk-badge");
    badge.className = "inline-block px-4 py-1.5 rounded-full text-sm font-semibold mb-3 " + style.badge;
    badge.textContent = d.risk_segment + " Risk";

    // RFM
    document.getElementById("r-score").textContent    = d.r_score;
    document.getElementById("f-score").textContent    = d.f_score;
    document.getElementById("m-score").textContent    = d.m_score;
    document.getElementById("rfm-segment").textContent= d.rfm_segment;
    document.getElementById("rfm-score").textContent  = d.rfm_score + " / 15";

    // Priority dots
    const dotsEl = document.getElementById("priority-dots");
    dotsEl.innerHTML = "";
    for (let i = 1; i <= 5; i++) {
      const dot = document.createElement("div");
      dot.className = "w-2.5 h-2.5 rounded-full " +
        (i <= d.retention_priority ? "bg-brand" : "bg-gray-200");
      dotsEl.appendChild(dot);
    }

    // Recommendation
    document.getElementById("recommendation").textContent = d.recommendation;
  }
</script>

</body>
</html>"""
