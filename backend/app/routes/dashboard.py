"""
dashboard.py — Dashboard summary and alert endpoints.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.database import PredictionLog, SolarObservation, get_db
from app.schemas.schemas import AlertResponse, DashboardResponse
from app.services.alert_service import get_current_alert
from app.services.prediction_service import predict

router = APIRouter(prefix="/api", tags=["Dashboard"])
logger = logging.getLogger(__name__)


@router.get("/dashboard", response_model=DashboardResponse, summary="Latest dashboard snapshot")
def dashboard(db: Session = Depends(get_db)) -> DashboardResponse:
    """
    Returns the most recent flux reading together with the associated
    flare probability and risk level.

    If no observations are in the database yet, returns zeroed values.
    """
    latest_obs: SolarObservation | None = (
        db.query(SolarObservation)
        .order_by(SolarObservation.timestamp.desc())
        .first()
    )

    if latest_obs is None:
        logger.info("Dashboard: no observations found, returning zeros")
        return DashboardResponse(
            latest_soft_flux=0.0,
            latest_hard_flux=0.0,
            risk_level="LOW",
            flare_probability=0.0,
        )

    try:
        result = predict(latest_obs.soft_xray_flux, latest_obs.hard_xray_flux)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return DashboardResponse(
        latest_soft_flux=latest_obs.soft_xray_flux,
        latest_hard_flux=latest_obs.hard_xray_flux,
        risk_level=result["risk_level"],
        flare_probability=result["flare_probability"],
    )


@router.get("/alerts", response_model=AlertResponse, summary="Current alert status")
def alerts(db: Session = Depends(get_db)) -> AlertResponse:
    """
    Checks the most recent prediction log and raises an alert if
    flare probability exceeds the configured threshold (default 50 %).
    """
    return AlertResponse(**get_current_alert(db))
