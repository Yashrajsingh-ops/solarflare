"""Pydantic request / response models."""

from datetime import datetime
from typing import Any

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
    flare_class: str = Field(..., description="A | B | C | M | X flare class")
    reasons: list[str] = Field(default_factory=list)
    impact: dict[str, Any] = Field(default_factory=dict)
    model_name: str = Field(default="heuristic")
    trend: str = Field(default="STABLE")


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


class HistoryItem(BaseModel):
    id: int
    created_at: datetime
    soft_xray_flux: float
    hard_xray_flux: float
    flare_probability: float
    risk_level: str
    prediction: str
    flare_class: str


class HistoryResponse(BaseModel):
    items: list[HistoryItem]


class FlareEventItem(BaseModel):
    id: int
    event_time: datetime
    flare_class: str
    flare_probability: float
    risk_level: str
    summary: str


class FlareEventsResponse(BaseModel):
    items: list[FlareEventItem]


class AnalyticsResponse(BaseModel):
    total_observations: int
    total_predictions: int
    total_flare_events: int
    flare_class_counts: dict[str, int]
    risk_counts: dict[str, int]
    average_probability: float
    high_risk_alerts: int
    critical_risk_alerts: int


class ExplanationResponse(BaseModel):
    prediction: str
    reasons: list[str]
    flare_class: str
    impact: dict[str, Any]


class AlertHistoryItem(BaseModel):
    id: int
    created_at: datetime
    alert_level: str
    message: str
    acknowledged: bool


class AlertHistoryResponse(BaseModel):
    items: list[AlertHistoryItem]


class HealthResponse(BaseModel):
    status: str
