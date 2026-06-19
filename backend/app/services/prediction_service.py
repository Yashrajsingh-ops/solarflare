"""Prediction, explainability, and flare-class utilities."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.feature_engineering import FEATURE_COLUMNS, build_single_row_features

logger = logging.getLogger(__name__)

MODEL_PATH = Path("ml/xgboost_model.pkl")
RF_MODEL_PATH = Path("ml/random_forest_model.pkl")
SCALER_PATH = Path("ml/scaler.pkl")

_model = None
_scaler = None
_model_name = None


def _load_artifacts():
    """Lazy-load model and scaler from disk."""
    global _model, _scaler, _model_name
    if _model is not None:
        return

    if MODEL_PATH.exists() and SCALER_PATH.exists():
        import joblib

        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _model_name = "xgboost"
        logger.info("XGBoost model loaded from %s", MODEL_PATH)
    elif RF_MODEL_PATH.exists() and SCALER_PATH.exists():
        import joblib

        _model = joblib.load(RF_MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _model_name = "random_forest"
        logger.info("RandomForest model loaded from %s", RF_MODEL_PATH)
    else:
        _model_name = "heuristic"
        logger.warning(
            "Model files not found — using heuristic fallback. "
            "Run ml/train_model.py to train the real model."
        )


def _heuristic_probability(soft: float, hard: float) -> float:
    """Simple physics-inspired fallback probability in [0, 1]."""
    combined = soft + hard * 2
    prob = 1 - np.exp(-combined / 120.0)
    return float(np.clip(prob, 0.0, 1.0))


def _categorise(prob_pct: float) -> tuple[str, str]:
    if prob_pct < 25:
        return "LOW", "Solar Flare Unlikely"
    if prob_pct < 50:
        return "MEDIUM", "Solar Flare Possible"
    if prob_pct < 75:
        return "HIGH", "Solar Flare Likely"
    return "CRITICAL", "Solar Flare Imminent"


def classify_flare(soft_xray_flux: float, hard_xray_flux: float, flare_probability: float) -> str:
    """Map the current state to a flare class label."""
    if flare_probability >= 90 or (soft_xray_flux >= 120 and hard_xray_flux >= 60):
        return "X"
    if flare_probability >= 75 or soft_xray_flux >= 90 or hard_xray_flux >= 45:
        return "M"
    if flare_probability >= 50 or soft_xray_flux >= 60 or hard_xray_flux >= 30:
        return "C"
    if flare_probability >= 25 or soft_xray_flux >= 30 or hard_xray_flux >= 12:
        return "B"
    return "A"


def explain_prediction(features: dict, flare_probability: float) -> list[str]:
    """Return short human-readable explanation strings."""
    reasons: list[str] = []
    soft = features["soft_xray_flux"]
    hard = features["hard_xray_flux"]
    ratio = features["soft_hard_ratio"]

    if soft >= 50:
        reasons.append("Rapid Soft X-ray Increase")
    if hard >= 20:
        reasons.append("Hard X-ray Spike Detected")
    if ratio >= 2.0:
        reasons.append("Soft/Hard Flux Ratio Elevated")
    if flare_probability >= 50:
        reasons.append("Similar Historical Pattern Found")

    if not reasons:
        reasons.append("Flux levels currently remain below flare thresholds")
    return reasons[:4]


def assess_impact(flare_class: str, risk_level: str, flare_probability: float) -> dict:
    """Rule-based impact engine for hackathon presentation value."""
    score = 0
    if flare_class in {"M", "X"}:
        score += 2
    if flare_class == "X":
        score += 2
    if risk_level in {"HIGH", "CRITICAL"}:
        score += 1
    if flare_probability >= 75:
        score += 1

    if score >= 5:
        severity = "EXTREME"
        impacts = ["GPS Disruption", "Satellite Communication Issues", "Radio Blackout Risk", "Space Weather Severity: Extreme"]
    elif score >= 3:
        severity = "HIGH"
        impacts = ["Satellite Communication Issues", "Radio Blackout Risk", "Space Weather Severity: High"]
    elif score >= 2:
        severity = "MODERATE"
        impacts = ["Potential GPS Drift", "Minor Communication Degradation", "Space Weather Severity: Moderate"]
    else:
        severity = "LOW"
        impacts = ["No significant operational impact expected", "Space Weather Severity: Low"]

    return {"severity": severity, "impacts": impacts}


def _trend_from_features(features: dict) -> str:
    if features["soft_change_rate"] > 0 and features["hard_change_rate"] > 0:
        return "RISING"
    if features["soft_change_rate"] < 0 and features["hard_change_rate"] < 0:
        return "FALLING"
    return "STABLE"


def predict(soft_xray_flux: float, hard_xray_flux: float) -> dict:
    _load_artifacts()

    features = build_single_row_features(soft_xray_flux, hard_xray_flux)
    feature_row = pd.DataFrame([features])[FEATURE_COLUMNS]

    if _model is not None:
        X = _scaler.transform(feature_row) if _scaler else feature_row.values
        prob = float(_model.predict_proba(X)[0][1])
        model_name = _model_name or "xgboost"
    else:
        prob = _heuristic_probability(soft_xray_flux, hard_xray_flux)
        model_name = "heuristic"

    prob_pct = round(prob * 100, 1)
    risk_level, prediction = _categorise(prob_pct)
    flare_class = classify_flare(soft_xray_flux, hard_xray_flux, prob_pct)
    reasons = explain_prediction(features, prob_pct)
    impact = assess_impact(flare_class, risk_level, prob_pct)

    logger.info(
        "Prediction — soft=%.2f hard=%.2f → prob=%.1f%% [%s] class=%s",
        soft_xray_flux,
        hard_xray_flux,
        prob_pct,
        risk_level,
        flare_class,
    )

    return {
        "flare_probability": prob_pct,
        "risk_level": risk_level,
        "prediction": prediction,
        "flare_class": flare_class,
        "reasons": reasons,
        "impact": impact,
        "model_name": model_name,
        "feature_snapshot": json.dumps(features, default=float),
        "trend": _trend_from_features(features),
    }
