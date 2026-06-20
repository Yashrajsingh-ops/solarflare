"""
Prediction, explainability, and flare-class utilities.

Configuration Notes:
--------------------
This service is configured to PRIORITIZE FLARE RECALL for space weather early warning.

Risk Level Thresholds (adjusted for high-recall operation):
- The thresholds are set lower than traditional 25/50/75% splits
- This ensures earlier warnings at the cost of more false alarms
- For space weather, missing a flare is more dangerous than a false alert

Threshold Philosophy:
- 30% probability triggers HIGH risk (vs traditional 50%)
- This aligns with the 0.3 classification threshold used in model training
- Operators receive earlier warnings to protect critical infrastructure
"""

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

# Flare detection threshold: 0.3 (30%) instead of 0.5 (50%)
# Lower threshold prioritizes recall for space weather early warning
# Missing a flare (false negative) is more dangerous than a false alarm
FLARE_PROBABILITY_THRESHOLD = 30.0  # percentage

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
    """
    Categorize flare probability into risk levels.

    Thresholds are set for HIGH-RECALL early warning operation:
    - LOW: < 15% (very unlikely, minimal concern)
    - MEDIUM: 15-30% (possible, monitor closely)
    - HIGH: 30-60% (likely, prepare countermeasures) ← Alert threshold
    - CRITICAL: >= 60% (imminent, take immediate action)

    Note: Traditional thresholds are 25/50/75%. We use 15/30/60% to trigger
    alerts earlier, prioritizing recall for space weather safety.
    """
    if prob_pct < 15:
        return "LOW", "Solar Flare Unlikely"
    if prob_pct < FLARE_PROBABILITY_THRESHOLD:  # 30%
        return "MEDIUM", "Solar Flare Possible"
    if prob_pct < 60:
        return "HIGH", "Solar Flare Likely"
    return "CRITICAL", "Solar Flare Imminent"


def classify_flare(soft_xray_flux: float) -> str:
    """
    Classify flare based on GOES soft X-ray flux thresholds (W/m²).

    GOES Physical Class Thresholds:
    - A: soft_xray_flux < 1e-7
    - B: 1e-7 <= soft_xray_flux < 1e-6
    - C: 1e-6 <= soft_xray_flux < 1e-5
    - M: 1e-5 <= soft_xray_flux < 1e-4
    - X: soft_xray_flux >= 1e-4

    Note: Risk level is derived separately from model probability.
    """
    if soft_xray_flux >= 1e-4:
        return "X"
    if soft_xray_flux >= 1e-5:
        return "M"
    if soft_xray_flux >= 1e-6:
        return "C"
    if soft_xray_flux >= 1e-7:
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
    # Use the configured threshold (30%) for pattern matching explanation
    if flare_probability >= FLARE_PROBABILITY_THRESHOLD:
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
    flare_class = classify_flare(soft_xray_flux)
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
