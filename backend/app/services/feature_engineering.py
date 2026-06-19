"""
feature_engineering.py — Time-series feature generation from solar flux data.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML-ready features from a cleaned observation DataFrame.

    New columns
    -----------
    rolling_mean_5     : 5-row rolling mean of soft X-ray flux
    rolling_mean_15    : 15-row rolling mean of soft X-ray flux
    soft_change_rate   : first-order difference of soft flux
    hard_change_rate   : first-order difference of hard flux
    moving_avg         : 3-row centred moving average of soft flux
    soft_hard_ratio    : soft / (hard + ε)
    prev_soft_flux     : lag-1 of soft flux
    prev_hard_flux     : lag-1 of hard flux
    flux_delta         : absolute difference between soft and hard flux
    """
    df = df.copy()

    eps = 1e-9  # avoid division by zero

    df["rolling_mean_5"] = (
        df["soft_xray_flux"].rolling(window=5, min_periods=1).mean()
    )
    df["rolling_mean_15"] = (
        df["soft_xray_flux"].rolling(window=15, min_periods=1).mean()
    )
    df["soft_change_rate"] = df["soft_xray_flux"].diff().fillna(0)
    df["hard_change_rate"] = df["hard_xray_flux"].diff().fillna(0)
    df["moving_avg"] = (
        df["soft_xray_flux"].rolling(window=3, min_periods=1, center=True).mean()
    )
    df["soft_hard_ratio"] = df["soft_xray_flux"] / (df["hard_xray_flux"] + eps)
    df["prev_soft_flux"] = df["soft_xray_flux"].shift(1).fillna(df["soft_xray_flux"])
    df["prev_hard_flux"] = df["hard_xray_flux"].shift(1).fillna(df["hard_xray_flux"])
    df["flux_delta"] = np.abs(df["soft_xray_flux"] - df["hard_xray_flux"])

    logger.info("Feature engineering complete — shape: %s", df.shape)
    return df


FEATURE_COLUMNS = [
    "soft_xray_flux",
    "hard_xray_flux",
    "rolling_mean_5",
    "rolling_mean_15",
    "soft_change_rate",
    "hard_change_rate",
    "moving_avg",
    "soft_hard_ratio",
    "prev_soft_flux",
    "prev_hard_flux",
    "flux_delta",
]


def build_single_row_features(soft: float, hard: float) -> dict:
    """
    Construct a feature dict for a single real-time prediction request.
    Rolling/lag features default to the provided values when history is absent.
    """
    eps = 1e-9
    return {
        "soft_xray_flux": soft,
        "hard_xray_flux": hard,
        "rolling_mean_5": soft,
        "rolling_mean_15": soft,
        "soft_change_rate": 0.0,
        "hard_change_rate": 0.0,
        "moving_avg": soft,
        "soft_hard_ratio": soft / (hard + eps),
        "prev_soft_flux": soft,
        "prev_hard_flux": hard,
        "flux_delta": abs(soft - hard),
    }
