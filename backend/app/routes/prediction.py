"""
prediction.py — Real-time solar flare prediction endpoint.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.database import PredictionLog, get_db
from app.schemas.schemas import PredictRequest, PredictResponse
from app.services.prediction_service import predict

router = APIRouter(prefix="/api", tags=["Prediction"])
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=PredictResponse, summary="Predict solar flare probability")
def predict_flare(
    payload: PredictRequest,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """
    Given real-time soft and hard X-ray flux readings, return:
    - **flare_probability** — probability 0–100 %
    - **risk_level** — LOW | MEDIUM | HIGH | CRITICAL
    - **prediction** — human-readable verdict
    """
    try:
        result = predict(payload.soft_xray_flux, payload.hard_xray_flux)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    # Persist to audit log
    log = PredictionLog(
        soft_xray_flux=payload.soft_xray_flux,
        hard_xray_flux=payload.hard_xray_flux,
        flare_probability=result["flare_probability"],
        risk_level=result["risk_level"],
        prediction=result["prediction"],
    )
    db.add(log)
    db.commit()

    return PredictResponse(**result)
