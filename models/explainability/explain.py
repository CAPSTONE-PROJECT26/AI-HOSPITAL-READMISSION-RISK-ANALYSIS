"""
Unified Explainability Engine
=============================

Generates SHAP and LIME explanations for patient readmission risk.

SHAP:
- Global feature importance
- Patient-level feature contributions

LIME:
- Patient-level local explanation

The explainability engine explains the best base model.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from lime.lime_tabular import LimeTabularExplainer

from models.explainability.feature_names import (
    FEATURE_NAME_MAP,
    get_display_name,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
    / "best_base_model.joblib"
)

X_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "X_train.csv"
)

X_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "X_test.csv"
)

Y_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "y_test.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load the best trained base model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}\n"
            "Run models/train.py first."
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load training and test datasets."""

    required_files = [
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{file_path}\n"
                "Run preprocessing first."
            )

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    return X_train, X_test, y_test


# ============================================================
# SHAP
# ============================================================

def create_shap_explainer(model, X_train):
    """Create the appropriate SHAP explainer."""

    model_name = type(model).__name__

    if model_name == "LogisticRegression":
        return shap.LinearExplainer(
            model,
            X_train,
        )

    return shap.Explainer(
        model,
        X_train,
    )


def generate_shap_values(explainer, X_test):
    """Generate SHAP values."""

    return explainer(X_test)


def get_global_shap_importance(
    shap_values,
    feature_names,
):
    """Calculate global mean absolute SHAP importance."""

    values = shap_values.values

    if values.ndim == 3:
        values = np.mean(
            np.abs(values),
            axis=2,
        )

    importance = np.mean(
        np.abs(values),
        axis=0,
    )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": importance,
        }
    )

    result["display_name"] = result["feature"].apply(
        get_display_name
    )

    result = result.sort_values(
        by="mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    return result[
        [
            "display_name",
            "feature",
            "mean_abs_shap",
        ]
    ]


def explain_with_shap(
    patient_index,
    X_test,
    shap_values,
):
    """Generate SHAP explanation for one patient."""

    values = shap_values.values

    if values.ndim == 3:
        patient_values = values[
            patient_index
        ].mean(axis=1)
    else:
        patient_values = values[
            patient_index
        ]

    patient_features = X_test.iloc[
        patient_index
    ]

    explanation = pd.DataFrame(
        {
            "feature": X_test.columns,
            "value": patient_features.values,
            "shap_value": patient_values,
        }
    )

    explanation["display_name"] = (
        explanation["feature"].apply(
            get_display_name
        )
    )

    explanation["direction"] = np.where(
        explanation["shap_value"] >= 0,
        "increases risk",
        "decreases risk",
    )

    explanation["absolute_shap"] = np.abs(
        explanation["shap_value"]
    )

    explanation = explanation.sort_values(
        by="absolute_shap",
        ascending=False,
    ).reset_index(drop=True)

    return explanation[
        [
            "display_name",
            "feature",
            "value",
            "shap_value",
            "absolute_shap",
            "direction",
        ]
    ]


# ============================================================
# LIME
# ============================================================

def create_lime_explainer(X_train):
    """Create LIME tabular explainer."""

    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=[
            "Not Readmitted",
            "Readmitted",
        ],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )


def extract_lime_feature_name(
    lime_condition
):
    """
    Identify the original model feature contained
    inside a LIME condition string.
    """

    for model_feature in FEATURE_NAME_MAP:

        if model_feature in lime_condition:
            return model_feature

    return lime_condition


def explain_with_lime(
    patient_index,
    model,
    X_test,
    lime_explainer,
):
    """Generate LIME explanation for one patient."""

    patient = X_test.iloc[
        patient_index
    ].values

    # --------------------------------------------------------
    # LIME passes NumPy arrays to predict_fn.
    # Convert them back to a DataFrame so that the model
    # receives the same feature names it was trained with.
    # --------------------------------------------------------

    def lime_predict(data):

        data = pd.DataFrame(
            data,
            columns=X_test.columns,
        )

        return model.predict_proba(data)

    explanation = lime_explainer.explain_instance(
        data_row=patient,
        predict_fn=lime_predict,
        num_features=10,
    )

    records = []

    for feature, weight in explanation.as_list():

        original_feature = (
            extract_lime_feature_name(
                feature
            )
        )

        display_name = get_display_name(
            original_feature
        )

        direction = (
            "increases risk"
            if weight >= 0
            else "decreases risk"
        )

        records.append(
            {
                "display_name": display_name,
                "feature": feature,
                "weight": weight,
                "direction": direction,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# UNIFIED PATIENT EXPLANATION
# ============================================================

def explain_patient(
    patient_index,
    model,
    X_test,
    shap_values,
    lime_explainer,
):
    """Generate SHAP + LIME explanations."""

    shap_explanation = explain_with_shap(
        patient_index=patient_index,
        X_test=X_test,
        shap_values=shap_values,
    )

    lime_explanation = explain_with_lime(
        patient_index=patient_index,
        model=model,
        X_test=X_test,
        lime_explainer=lime_explainer,
    )

    return {
        "shap": shap_explanation,
        "lime": lime_explanation,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("UNIFIED EXPLAINABILITY ENGINE")
    print("=" * 60)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nLoading model...")

    model = load_model()

    print(
        f"Model: {type(model).__name__}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print("\nLoading data...")

    X_train, X_test, y_test = load_data()

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    print(
        f"Features: {X_test.shape[1]}"
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print("\nCreating SHAP explainer...")

    shap_explainer = create_shap_explainer(
        model=model,
        X_train=X_train,
    )

    print("Generating SHAP values...")

    shap_values = generate_shap_values(
        explainer=shap_explainer,
        X_test=X_test,
    )

    print(
        "SHAP values generated successfully."
    )

    # --------------------------------------------------------
    # GLOBAL SHAP
    # --------------------------------------------------------

    global_importance = (
        get_global_shap_importance(
            shap_values=shap_values,
            feature_names=X_test.columns,
        )
    )

    print("\n" + "=" * 60)
    print("TOP 10 GLOBAL SHAP FEATURES")
    print("=" * 60)

    print(
        global_importance
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # LIME
    # --------------------------------------------------------

    print("\nCreating LIME explainer...")

    lime_explainer = create_lime_explainer(
        X_train=X_train
    )

    print(
        "LIME explainer created successfully."
    )

    # --------------------------------------------------------
    # PATIENT
    # --------------------------------------------------------

    patient_index = 0

    print("\n" + "=" * 60)
    print(
        f"PATIENT {patient_index} EXPLANATION"
    )
    print("=" * 60)

    explanations = explain_patient(
        patient_index=patient_index,
        model=model,
        X_test=X_test,
        shap_values=shap_values,
        lime_explainer=lime_explainer,
    )

    # --------------------------------------------------------
    # SHAP PATIENT
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("SHAP — PATIENT LEVEL")
    print("-" * 60)

    print(
        explanations["shap"]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # LIME PATIENT
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("LIME — PATIENT LEVEL")
    print("-" * 60)

    print(
        explanations["lime"]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    patient = X_test.iloc[
        [patient_index]
    ]

    probability = model.predict_proba(
        patient
    )[0][1]

    prediction = model.predict(
        patient
    )[0]

    print("\n" + "=" * 60)
    print("PATIENT RISK")
    print("=" * 60)

    print(
        f"Readmission probability: "
        f"{probability:.4f}"
    )

    print(
        f"Predicted class: {prediction}"
    )

    print(
        "\nExplainability engine "
        "completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()