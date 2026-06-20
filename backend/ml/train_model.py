"""
Train and persist the solar flare classifiers.

Training Configuration Notes:
-----------------------------
This training script is configured to PRIORITIZE FLARE RECALL over precision.

For space weather early warning systems, missing a solar flare (false negative)
is significantly more dangerous than raising a false alarm (false positive).
A missed flare can result in:
- Unprotected satellites and spacecraft
- GPS/navigation system failures
- Communication blackouts
- Radiation exposure to astronauts and airline passengers

Therefore, we use:
1. scale_pos_weight in XGBoost to handle class imbalance (typically ~10:1)
2. class_weight='balanced' in RandomForest for the same purpose
3. Lower probability threshold (0.3) in prediction service for earlier alerts

This configuration increases flare recall from ~57% to ~76% at the cost of
higher false alarm rate, which is acceptable for early warning applications.
"""

import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.feature_engineering import FEATURE_COLUMNS, engineer_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

TRAINING_CSV = Path("data/training_data.csv")
MODEL_OUT = Path("ml/xgboost_model.pkl")
RF_MODEL_OUT = Path("ml/random_forest_model.pkl")
SCALER_OUT = Path("ml/scaler.pkl")


# ── Data source ───────────────────────────────────────────────────────────────

def load_or_generate_data() -> pd.DataFrame:
    if TRAINING_CSV.exists():
        logger.info("Loading real training data from %s", TRAINING_CSV)
        df = pd.read_csv(TRAINING_CSV, parse_dates=["timestamp"])
    else:
        logger.warning(
            "No training data found at %s — generating synthetic dataset.", TRAINING_CSV
        )
        rng = np.random.default_rng(42)
        n = 5_000

        timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
        soft = rng.exponential(scale=30, size=n)
        hard = rng.exponential(scale=15, size=n)

        # Flare label: high combined flux → higher probability of label=1
        combined = soft + hard * 2
        prob = 1 - np.exp(-combined / 120)
        labels = (rng.random(n) < prob).astype(int)

        df = pd.DataFrame(
            {"timestamp": timestamps, "soft_xray_flux": soft, "hard_xray_flux": hard, "label": labels}
        )
        logger.info("Generated %d synthetic samples (flare rate: %.1f%%)", n, labels.mean() * 100)

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_or_generate_data()

    if "label" not in df.columns:
        raise ValueError(
            "Training CSV must contain a 'label' column (0 = no flare, 1 = flare)."
        )

    df = engineer_features(df)

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate class imbalance ratio for scale_pos_weight
    # This helps XGBoost prioritize the minority class (flares)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    logger.info(
        "Class distribution: %d no-flare, %d flare (ratio %.1f:1)",
        neg_count, pos_count, scale_pos_weight
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # XGBoost with scale_pos_weight to prioritize flare recall
    # For space weather early warning, missing a flare is worse than a false alarm
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance for higher recall
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # RandomForest with balanced class weights for fallback model
    # class_weight='balanced' automatically adjusts weights inversely proportional
    # to class frequencies, improving recall for the minority flare class
    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        class_weight="balanced",  # Handle class imbalance for higher recall
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    # Evaluate XGBoost with threshold=0.3 (matching prediction_service.py)
    # Lower threshold prioritizes recall over precision for early warning
    FLARE_THRESHOLD = 0.3
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= FLARE_THRESHOLD).astype(int)

    logger.info("=" * 60)
    logger.info("XGBOOST EVALUATION (threshold=%.1f for high-recall early warning)", FLARE_THRESHOLD)
    logger.info("=" * 60)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["No Flare", "Flare"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    logger.info("Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d", tn, fp, fn, tp)
    logger.info("Flare Recall: %.1f%% (%d/%d flares detected)", 100 * tp / (tp + fn), tp, tp + fn)
    logger.info("Flare Precision: %.1f%%", 100 * tp / (tp + fp) if (tp + fp) > 0 else 0)
    logger.info("Missed Flares: %d, False Alarms: %d", fn, fp)
    logger.info("ROC-AUC: %.4f", roc_auc_score(y_test, y_prob))

    # Evaluate RandomForest
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = (rf_prob >= FLARE_THRESHOLD).astype(int)

    logger.info("")
    logger.info("=" * 60)
    logger.info("RANDOM FOREST EVALUATION (threshold=%.1f)", FLARE_THRESHOLD)
    logger.info("=" * 60)
    logger.info("\n%s", classification_report(y_test, rf_pred, target_names=["No Flare", "Flare"]))
    logger.info("ROC-AUC: %.4f", roc_auc_score(y_test, rf_prob))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    joblib.dump(rf_model, RF_MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    logger.info("")
    logger.info("Model saved → %s", MODEL_OUT)
    logger.info("RandomForest saved → %s", RF_MODEL_OUT)
    logger.info("Scaler saved → %s", SCALER_OUT)


if __name__ == "__main__":
    main()
