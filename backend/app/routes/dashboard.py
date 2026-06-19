"""Dashboard summary and alert endpoints."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.models.database import AlertHistory, FlareEvent, PredictionLog, SolarObservation, get_db
from app.schemas.schemas import (
    AlertHistoryResponse,
    AlertResponse,
    AnalyticsResponse,
    DashboardResponse,
    ExplanationResponse,
    FlareEventsResponse,
    HistoryResponse,
)
from app.services.alert_service import get_current_alert
from app.services.analytics_service import build_analytics
from app.services.history_service import flare_event_rows, history_rows
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


@router.get("/history", response_model=HistoryResponse, summary="Prediction history")
def history(db: Session = Depends(get_db), limit: int = Query(50, ge=1, le=500)) -> HistoryResponse:
    items = history_rows(db)[:limit]
    return HistoryResponse(items=items)


@router.get("/flare-events", response_model=FlareEventsResponse, summary="Historical flare events")
def flare_events(db: Session = Depends(get_db), limit: int = Query(50, ge=1, le=500)) -> FlareEventsResponse:
    return FlareEventsResponse(items=flare_event_rows(db)[:limit])


@router.get("/analytics", response_model=AnalyticsResponse, summary="Platform analytics")
def analytics(db: Session = Depends(get_db)) -> AnalyticsResponse:
    return AnalyticsResponse(**build_analytics(db))


@router.get("/alerts/history", response_model=AlertHistoryResponse, summary="Alert history")
def alert_history(db: Session = Depends(get_db), limit: int = Query(50, ge=1, le=500)) -> AlertHistoryResponse:
    rows = (
        db.query(AlertHistory)
        .order_by(AlertHistory.created_at.desc())
        .limit(limit)
        .all()
    )
    return AlertHistoryResponse(
        items=[
            {
                "id": row.id,
                "created_at": row.created_at,
                "alert_level": row.alert_level,
                "message": row.message,
                "acknowledged": row.acknowledged,
            }
            for row in rows
        ]
    )


@router.get("/explain", response_model=ExplanationResponse, summary="Explain latest prediction")
def explain(db: Session = Depends(get_db)) -> ExplanationResponse:
    latest: PredictionLog | None = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .first()
    )
    if latest is None:
        raise HTTPException(status_code=404, detail="No predictions found.")

    result = predict(latest.soft_xray_flux, latest.hard_xray_flux)
    return ExplanationResponse(
        prediction=result["prediction"],
        reasons=result["reasons"],
        flare_class=result["flare_class"],
        impact=result["impact"],
    )
