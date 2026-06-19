"""Historical event helpers."""

import json

from sqlalchemy.orm import Session

from app.models.database import AlertHistory, FlareEvent, PredictionInsight, PredictionLog


def _flare_class_from_log(log: PredictionLog) -> str:
    if log.insight is not None:
        return log.insight.flare_class
    if log.risk_level == "CRITICAL":
        return "X"
    if log.risk_level == "HIGH":
        return "M"
    if log.risk_level == "MEDIUM":
        return "C"
    return "B"


def persist_prediction_artifacts(
    db: Session,
    prediction_log: PredictionLog,
    prediction_result: dict,
) -> None:
    insight = PredictionInsight(
        prediction_log_id=prediction_log.id,
        flare_class=prediction_result["flare_class"],
        model_name=prediction_result["model_name"],
        reasons_json=json.dumps(prediction_result["reasons"]),
        impact_json=json.dumps(prediction_result["impact"]),
    )
    event = FlareEvent(
        prediction_log_id=prediction_log.id,
        flare_class=prediction_result["flare_class"],
        flare_probability=prediction_result["flare_probability"],
        risk_level=prediction_result["risk_level"],
        summary=f"{prediction_result['prediction']} ({prediction_result['flare_class']}-class)",
    )
    db.add(insight)
    db.add(event)

    if prediction_result["risk_level"] in {"HIGH", "CRITICAL"}:
        alert = AlertHistory(
            prediction_log_id=prediction_log.id,
            alert_level=prediction_result["risk_level"],
            message=f"Solar Flare Warning: {prediction_result['risk_level']} Activity Detected",
        )
        db.add(alert)


def history_rows(db: Session) -> list[dict]:
    rows = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).all()
    items: list[dict] = []
    for row in rows:
        flare_class = _flare_class_from_log(row)
        items.append(
            {
                "id": row.id,
                "created_at": row.created_at,
                "soft_xray_flux": row.soft_xray_flux,
                "hard_xray_flux": row.hard_xray_flux,
                "flare_probability": row.flare_probability,
                "risk_level": row.risk_level,
                "prediction": row.prediction,
                "flare_class": flare_class,
            }
        )
    return items


def flare_event_rows(db: Session) -> list[dict]:
    events = db.query(FlareEvent).order_by(FlareEvent.event_time.desc()).all()
    return [
        {
            "id": event.id,
            "event_time": event.event_time,
            "flare_class": event.flare_class,
            "flare_probability": event.flare_probability,
            "risk_level": event.risk_level,
            "summary": event.summary,
        }
        for event in events
    ]
