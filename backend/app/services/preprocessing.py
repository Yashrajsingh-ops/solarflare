"""
preprocessing.py — CSV validation and cleaning.
"""

import io
import logging

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"timestamp", "soft_xray_flux", "hard_xray_flux"}


def parse_and_validate_csv(content: bytes) -> pd.DataFrame:
    """
    Parse raw CSV bytes, validate schema, and return a clean DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing or all rows are invalid.
    """
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    # Column check
    missing = REQUIRED_COLUMNS - set(df.columns.str.strip().str.lower())
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df.columns = df.columns.str.strip().str.lower()

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Cast flux columns to float
    df["soft_xray_flux"] = pd.to_numeric(df["soft_xray_flux"], errors="coerce")
    df["hard_xray_flux"] = pd.to_numeric(df["hard_xray_flux"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["timestamp", "soft_xray_flux", "hard_xray_flux"])

    # Remove physically unreasonable negatives
    df = df[(df["soft_xray_flux"] >= 0) & (df["hard_xray_flux"] >= 0)]
    df = df.sort_values("timestamp").reset_index(drop=True)

    after = len(df)
    logger.info("Preprocessing: %d rows in → %d rows clean", before, after)

    if after == 0:
        raise ValueError("No valid rows remain after cleaning.")

    return df
