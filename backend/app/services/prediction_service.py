"""
prediction_service.py — Load the trained XGBoost model and run inference.
Falls back to a heuristic scorer when the model file is not yet present
(useful during development before ml/train_model.py has been run).
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.feature_engineering import FEATURE_COLUMNS, build_single_row_features

logger = logging.getLogger(__name__)

MODEL_PATH = Path("ml/xgboost_model.pkl")
SCALER_PATH = Path("ml/scaler.pkl")

_model = None
_scaler = None


def _load_artifacts():
    """Lazy-load model and scaler from disk."""
    global _model, _scaler
    if _model is not None:
        return

    if MODEL_PATH.exists() and SCALER_PATH.exists():
        import joblib

        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        logger.info("XGBoost model loaded from %s", MODEL_PATH)
    else:
        logger.warning(
            "Model files not found — using heuristic fallback. "
            "Run ml/train_model.py to train the real model."
        )


def _heuristic_probability(soft: float, hard: float) -> float:
    """
    Simple physics-inspired heuristic used as fallback when the model
    has not been trained yet.  Returns a probability in [0, 1].
    """
    # Higher combined flux → higher probability, saturating near 1
    combined = soft + hard * 2
    prob = 1 - np.exp(-combined / 120.0)
    return float(np.clip(prob, 0.0, 1.0))


def _categorise(prob_pct: float) -> tuple[str, str]:
    """Map probability percentage to (risk_level, prediction) strings."""
    if prob_pct < 25:
        return "LOW", "Solar Flare Unlikely"
    elif prob_pct < 50:
        return "MEDIUM", "Solar Flare Possible"
    elif prob_pct < 75:
        return "HIGH", "Solar Flare Likely"
    else:
        return "CRITICAL", "Solar Flare Imminent"


def predict(soft_xray_flux: float, hard_xray_flux: float) -> dict:
    """
    Run a prediction for a single (soft, hard) flux pair.

    Returns
    -------
    dict with keys: flare_probability, risk_level, prediction
    """
    _load_artifacts()

    features = build_single_row_features(soft_xray_flux, hard_xray_flux)
    feature_row = pd.DataFrame([features])[FEATURE_COLUMNS]

    if _model is not None:
        X = _scaler.transform(feature_row) if _scaler else feature_row.values
        prob = float(_model.predict_proba(X)[0][1])
    else:
        prob = _heuristic_probability(soft_xray_flux, hard_xray_flux)

    prob_pct = round(prob * 100, 1)
    risk_level, prediction = _categorise(prob_pct)

    logger.info(
        "Prediction — soft=%.2f hard=%.2f → prob=%.1f%% [%s]",
        soft_xray_flux,
        hard_xray_flux,
        prob_pct,
        risk_level,
    )

    return {
        "flare_probability": prob_pct,
        "risk_level": risk_level,
        "prediction": prediction,
    }
