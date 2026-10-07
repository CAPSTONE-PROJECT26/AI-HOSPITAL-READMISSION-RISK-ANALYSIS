from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# Columns that should never be used as predictive features
ID_COLUMNS = ["patient_id"]
TARGET_COLUMN = "readmitted_30d"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for the hospital readmission model.

    This function is designed specifically for the current demo_patients.csv
    schema. It does not modify the original columns.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # 1. AGE-BASED FEATURES
    # ---------------------------------------------------------
    if "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")

        # Age groups for nonlinear age effects
        df["age_group"] = pd.cut(
            age,
            bins=[0, 40, 60, 75, 120],
            labels=["young", "middle_aged", "older", "elderly"],
            right=False,
        )

        # Higher-risk age indicator
        df["age_75_plus"] = (age >= 75).astype(int)

    # ---------------------------------------------------------
    # 2. LENGTH-OF-STAY FEATURES
    # ---------------------------------------------------------
    if "length_of_stay" in df.columns:
        los = pd.to_numeric(df["length_of_stay"], errors="coerce")

        # Log transform reduces the effect of very long stays
        df["length_of_stay_log"] = np.log1p(los.clip(lower=0))

        # Long-stay indicator
        df["long_stay"] = (los >= 7).astype(int)

    # ---------------------------------------------------------
    # 3. PRIOR ADMISSION FEATURES
    # ---------------------------------------------------------
    if "prior_admissions" in df.columns:
        prior = pd.to_numeric(
            df["prior_admissions"],
            errors="coerce",
        ).fillna(0)

        # Any previous admission
        df["has_prior_admission"] = (prior > 0).astype(int)

        # Recurrent admission indicator
        df["frequent_prior_admission"] = (prior >= 2).astype(int)

        # Log transform for skewed admission counts
        df["prior_admissions_log"] = np.log1p(prior)

    # ---------------------------------------------------------
    # 4. MEDICATION FEATURES
    # ---------------------------------------------------------
    if "medication_count" in df.columns:
        meds = pd.to_numeric(
            df["medication_count"],
            errors="coerce",
        ).fillna(0)

        # Medication burden
        df["high_medication_burden"] = (meds >= 8).astype(int)

        # Log-transformed medication count
        df["medication_count_log"] = np.log1p(meds)

    # ---------------------------------------------------------
    # 5. COMORBIDITY FEATURES
    # ---------------------------------------------------------
    if "comorbidity_count" in df.columns:
        comorb = pd.to_numeric(
            df["comorbidity_count"],
            errors="coerce",
        ).fillna(0)

        # Do NOT call this Charlson Index.
        # The current dataset does not contain ICD diagnosis data.
        df["high_comorbidity_burden"] = (comorb >= 3).astype(int)

        df["comorbidity_count_log"] = np.log1p(comorb)

    # ---------------------------------------------------------
    # 6. COMORBIDITY COMBINATION
    # ---------------------------------------------------------
    comorbidity_flags = [
        "has_diabetes",
        "has_copd",
        "has_heart_failure",
        "has_renal_disease",
    ]

    available_flags = [
        column for column in comorbidity_flags
        if column in df.columns
    ]

    if available_flags:
        df["total_major_comorbidities"] = (
            df[available_flags]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum(axis=1)
        )

        df["multiple_major_comorbidities"] = (
            df["total_major_comorbidities"] >= 2
        ).astype(int)

    # ---------------------------------------------------------
    # 7. VITAL-SIGN FEATURES
    # ---------------------------------------------------------

    # Heart rate
    if "heart_rate" in df.columns:
        hr = pd.to_numeric(
            df["heart_rate"],
            errors="coerce",
        )

        df["abnormal_heart_rate"] = (
            (hr < 60) | (hr > 100)
        ).astype(int)

    # Systolic blood pressure
    if "systolic_bp" in df.columns:
        sbp = pd.to_numeric(
            df["systolic_bp"],
            errors="coerce",
        )

        df["abnormal_systolic_bp"] = (
            (sbp < 90) | (sbp > 140)
        ).astype(int)

    # Oxygen saturation
    if "oxygen_saturation" in df.columns:
        spo2 = pd.to_numeric(
            df["oxygen_saturation"],
            errors="coerce",
        )

        df["low_oxygen_saturation"] = (
            spo2 < 94
        ).astype(int)

    # Combined vital instability score
    vital_flags = [
        "abnormal_heart_rate",
        "abnormal_systolic_bp",
        "low_oxygen_saturation",
    ]

    available_vital_flags = [
        column for column in vital_flags
        if column in df.columns
    ]

    if available_vital_flags:
        df["vital_instability_count"] = (
            df[available_vital_flags]
            .sum(axis=1)
        )

        df["multiple_vital_abnormalities"] = (
            df["vital_instability_count"] >= 2
        ).astype(int)

    # ---------------------------------------------------------
    # 8. LABORATORY FEATURES
    # ---------------------------------------------------------

    # Creatinine
    if "creatinine" in df.columns:
        creatinine = pd.to_numeric(
            df["creatinine"],
            errors="coerce",
        )

        df["elevated_creatinine"] = (
            creatinine > 1.3
        ).astype(int)

    # Glucose
    if "glucose" in df.columns:
        glucose = pd.to_numeric(
            df["glucose"],
            errors="coerce",
        )

        df["abnormal_glucose"] = (
            (glucose < 70) | (glucose > 180)
        ).astype(int)

    # Hemoglobin
    if "hemoglobin" in df.columns:
        hemoglobin = pd.to_numeric(
            df["hemoglobin"],
            errors="coerce",
        )

        df["low_hemoglobin"] = (
            hemoglobin < 12
        ).astype(int)

    # ---------------------------------------------------------
    # 9. COMBINED CLINICAL RISK INDICATORS
    # ---------------------------------------------------------

    clinical_risk_flags = [
        "age_75_plus",
        "long_stay",
        "frequent_prior_admission",
        "high_medication_burden",
        "high_comorbidity_burden",
        "abnormal_heart_rate",
        "abnormal_systolic_bp",
        "low_oxygen_saturation",
        "elevated_creatinine",
        "abnormal_glucose",
        "low_hemoglobin",
    ]

    available_risk_flags = [
        column for column in clinical_risk_flags
        if column in df.columns
    ]

    if available_risk_flags:
        df["clinical_risk_factor_count"] = (
            df[available_risk_flags]
            .sum(axis=1)
        )

        df["multiple_clinical_risk_factors"] = (
            df["clinical_risk_factor_count"] >= 3
        ).astype(int)

    return df


def validate_input_columns(df: pd.DataFrame) -> None:
    """
    Validate that the expected source columns exist.
    """

    required_columns = [
        "age",
        "length_of_stay",
        "prior_admissions",
        "medication_count",
        "comorbidity_count",
        "heart_rate",
        "systolic_bp",
        "oxygen_saturation",
        "creatinine",
        "glucose",
        "hemoglobin",
        "has_diabetes",
        "has_copd",
        "has_heart_failure",
        "has_renal_disease",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}"
        )


def process_file(
    input_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Read raw data, generate engineered features,
    and save the result.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    validate_input_columns(df)

    original_columns = set(df.columns)

    engineered_df = add_engineered_features(df)

    new_columns = [
        column
        for column in engineered_df.columns
        if column not in original_columns
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    engineered_df.to_csv(
        output_path,
        index=False,
    )

    print("Feature engineering completed.")
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(engineered_df)}")
    print(f"Original columns: {len(original_columns)}")
    print(f"New engineered features: {len(new_columns)}")
    print("\nCreated features:")

    for column in new_columns:
        print(f"  - {column}")

    print(f"\nSaved to: {output_path}")

    return engineered_df


if __name__ == "__main__":
    process_file(
        "data/raw/demo_patients.csv",
        "data/processed/engineered_patients.csv",
    )