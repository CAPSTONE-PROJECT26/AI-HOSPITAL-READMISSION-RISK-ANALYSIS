from pathlib import Path

import joblib
import pandas as pd

from features.feature_engineering import add_engineered_features


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
    / "calibrated_model.joblib"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessor.joblib"
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Calibrated model not found: {MODEL_PATH}"
    )

if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessor not found: {PREPROCESSOR_PATH}"
    )


model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_THRESHOLD = 0.20


# ============================================================
# RISK BAND
# ============================================================

def get_risk_band(probability: float) -> str:
    """
    Convert predicted probability into a simple risk band.
    """

    if probability < 0.20:
        return "Low"

    if probability < 0.50:
        return "Medium"

    return "High"


# ============================================================
# PREDICTION
# ============================================================

def predict_patient(patient_data: dict) -> dict:
    """
    Run the complete patient prediction pipeline.

    Raw patient data
        ↓
    Feature engineering
        ↓
    Saved preprocessing pipeline
        ↓
    Calibrated model
        ↓
    Readmission probability
    """

    # --------------------------------------------------------
    # Convert raw dictionary to DataFrame
    # --------------------------------------------------------

    raw_df = pd.DataFrame([patient_data])

    # --------------------------------------------------------
    # Preserve patient ID
    # --------------------------------------------------------

    patient_id = raw_df["patient_id"].iloc[0]

    # --------------------------------------------------------
    # Remove patient ID from model features
    # --------------------------------------------------------

    model_input = raw_df.drop(
        columns=["patient_id"],
        errors="ignore",
    )

    # --------------------------------------------------------
    # Apply the SAME feature engineering used during training
    # --------------------------------------------------------

    engineered_df = add_engineered_features(
        model_input
    )

    # --------------------------------------------------------
    # Apply the fitted preprocessing pipeline
    # --------------------------------------------------------

    processed_data = preprocessor.transform(
        engineered_df
    )

    # --------------------------------------------------------
    # Generate calibrated probability
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(processed_data)[0][1]
    )

    # --------------------------------------------------------
    # Classification using selected project threshold
    # --------------------------------------------------------

    predicted_class = int(
        probability >= PREDICTION_THRESHOLD
    )

    # --------------------------------------------------------
    # Risk band
    # --------------------------------------------------------

    risk_band = get_risk_band(probability)

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "patient_id": str(patient_id),
        "readmission_probability": round(
            probability,
            4,
        ),
        "risk_percentage": round(
            probability * 100,
            2,
        ),
        "risk_band": risk_band,
        "predicted_class": predicted_class,
    }