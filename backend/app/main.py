"""main.py - FastAPI application factory.

"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.database import create_tables
from app.routes import dashboard, prediction, upload
from app.schemas.schemas import HealthResponse
from app.utils.helpers import configure_logging

# ── Logging ───────────────────────────────────────────────────────────────────
configure_logging(logging.INFO)
logger = logging.getLogger(__name__)

# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Solar Flare Forecasting API",
    description=(
        "Production-ready backend for solar flare forecasting using "
        "Aditya-L1 SOLEXS (soft X-ray) and HEL1OS (hard X-ray) data. "
        "Exposes REST endpoints for CSV ingestion, real-time prediction, "
        "dashboard visualisation, and alert generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup() -> None:
    create_tables()
    logger.info("Database tables created / verified.")


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(upload.router)
app.include_router(prediction.router)
app.include_router(dashboard.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    """Liveness probe — returns 200 when the service is up."""
    return HealthResponse(status="healthy")
