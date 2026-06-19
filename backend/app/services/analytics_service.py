"""Analytics helpers for the solar flare platform."""

from collections import Counter

from sqlalchemy.orm import Session

from app.models.database import AlertHistory, FlareEvent, PredictionLog, SolarObservation


def build_analytics(db: Session) -> dict:
    observations = db.query(SolarObservation).count()
    predictions = db.query(PredictionLog).count()
    flare_events = db.query(FlareEvent).count()

    prediction_rows = db.query(PredictionLog).all()
    flare_class_counts = Counter()
    risk_counts = Counter()
    probabilities = []

    for row in prediction_rows:
        risk_counts[row.risk_level] += 1
        probabilities.append(row.flare_probability)

    for event in db.query(FlareEvent).all():
        flare_class_counts[event.flare_class] += 1

    alerts = db.query(AlertHistory).all()
    high_alerts = sum(1 for alert in alerts if alert.alert_level == "HIGH")
    critical_alerts = sum(1 for alert in alerts if alert.alert_level == "CRITICAL")

    return {
        "total_observations": observations,
        "total_predictions": predictions,
        "total_flare_events": flare_events,
        "flare_class_counts": dict(sorted(flare_class_counts.items())),
        "risk_counts": dict(sorted(risk_counts.items())),
        "average_probability": round(sum(probabilities) / len(probabilities), 1) if probabilities else 0.0,
        "high_risk_alerts": high_alerts,
        "critical_risk_alerts": critical_alerts,
    }
