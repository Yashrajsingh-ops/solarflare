"""
alert_service.py — Determine alert status from the latest prediction log.
"""

import logging

from sqlalchemy.orm import Session

from app.models.database import PredictionLog

logger = logging.getLogger(__name__)

ALERT_THRESHOLD = 50.0  # percent — trigger alert above this probability


def get_current_alert(db: Session) -> dict:
    """
    Query the most recent prediction and decide whether to raise an alert.

    Returns
    -------
    dict with keys: alert (bool), message (str)
    """
    latest: PredictionLog | None = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .first()
    )

    if latest is None:
        return {"alert": False, "message": "No predictions recorded yet."}

    if latest.flare_probability >= ALERT_THRESHOLD:
        msg = (
            f"High probability solar flare detected — "
            f"{latest.flare_probability:.1f}% [{latest.risk_level}]"
        )
        logger.warning("ALERT: %s", msg)
        return {"alert": True, "message": msg}

    return {
        "alert": False,
        "message": f"All clear — current probability {latest.flare_probability:.1f}% [{latest.risk_level}]",
    }
