"""
schemas.py — Pydantic request / response models.
"""

from pydantic import BaseModel, Field, field_validator


# ── Request schemas ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    soft_xray_flux: float = Field(..., gt=0, description="Soft X-ray flux (SOLEXS)")
    hard_xray_flux: float = Field(..., gt=0, description="Hard X-ray flux (HEL1OS)")

    @field_validator("soft_xray_flux", "hard_xray_flux")
    @classmethod
    def must_be_finite(cls, v: float) -> float:
        import math
        if not math.isfinite(v):
            raise ValueError("Flux values must be finite numbers.")
        return v


# ── Response schemas ──────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    flare_probability: float = Field(..., description="Probability 0–100 %")
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    prediction: str = Field(..., description="Human-readable verdict")


class UploadResponse(BaseModel):
    message: str
    rows_ingested: int


class DashboardResponse(BaseModel):
    latest_soft_flux: float
    latest_hard_flux: float
    risk_level: str
    flare_probability: float


class AlertResponse(BaseModel):
    alert: bool
    message: str


class HealthResponse(BaseModel):
    status: str
