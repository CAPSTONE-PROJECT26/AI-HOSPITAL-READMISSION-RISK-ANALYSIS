from pathlib import Path

import joblib
import pandas as pd
import shap


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_DIR = PROJECT_ROOT / "models" / "artifacts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():
    """Load the best base model selected by the ML pipeline."""

    model_path = ARTIFACT_DIR / "best_base_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(model_path)


# =========================================================
# LOAD TEST DATA
# =========================================================

def load_test_data():
    """Load processed test features and target."""

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv"
    )

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    ).iloc[:, 0]

    return X_test, y_test


# =========================================================
# CREATE SHAP EXPLAINER
# =========================================================

def create_explainer(model, X_background):
    """
    Create the appropriate SHAP explainer.

    Logistic Regression uses LinearExplainer.
    Other supported models fall back to the
    general SHAP Explainer.
    """

    model_name = model.__class__.__name__.lower()

    if model_name == "logisticregression":

        return shap.LinearExplainer(
            model,
            X_background
        )

    return shap.Explainer(
        model.predict_proba,
        X_background
    )


# =========================================================
# GENERATE SHAP VALUES
# =========================================================

def generate_shap_values(model, X_test):
    """Generate SHAP explanations for the test dataset."""

    explainer = create_explainer(
        model,
        X_test
    )

    shap_values = explainer(X_test)

    return explainer, shap_values


# =========================================================
# GLOBAL FEATURE IMPORTANCE
# =========================================================

def get_global_feature_importance(
    shap_values,
    X_test
):
    """
    Calculate mean absolute SHAP importance
    for every feature across the test set.
    """

    importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": (
            abs(shap_values.values)
            .mean(axis=0)
        )
    })

    return importance.sort_values(
        "mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)


# =========================================================
# PATIENT-LEVEL EXPLANATION
# =========================================================

def explain_patient(
    shap_values,
    X_test,
    patient_index
):
    """
    Generate SHAP feature contributions
    for one patient.
    """

    patient_values = (
        shap_values.values[patient_index]
    )

    explanation = pd.DataFrame({
        "feature": X_test.columns,
        "feature_value": (
            X_test.iloc[patient_index].values
        ),
        "shap_value": patient_values
    })

    explanation["absolute_shap"] = (
        explanation["shap_value"].abs()
    )

    explanation["direction"] = (
        explanation["shap_value"]
        .apply(
            lambda value:
            "increases risk"
            if value > 0
            else "decreases risk"
        )
    )

    return explanation.sort_values(
        "absolute_shap",
        ascending=False
    ).reset_index(drop=True)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("SHAP EXPLAINABILITY")
    print("=" * 60)

    # Load model
    model = load_model()

    # Load test data
    X_test, y_test = load_test_data()

    print(
        f"\nModel: "
        f"{model.__class__.__name__}"
    )

    print(
        f"Test samples: "
        f"{X_test.shape[0]}"
    )

    print(
        f"Features: "
        f"{X_test.shape[1]}"
    )

    # Generate SHAP explanations
    explainer, shap_values = (
        generate_shap_values(
            model,
            X_test
        )
    )

    print(
        "\nSHAP explanation generated "
        "successfully."
    )

    print(
        f"SHAP output shape: "
        f"{shap_values.values.shape}"
    )

    # -----------------------------------------------------
    # Global importance
    # -----------------------------------------------------

    importance = (
        get_global_feature_importance(
            shap_values,
            X_test
        )
    )

    print(
        "\nTop 10 features by SHAP importance:"
    )

    print(
        importance.head(10)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # Patient explanation
    # -----------------------------------------------------

    patient_explanation = explain_patient(
        shap_values,
        X_test,
        patient_index=0
    )

    print(
        "\nTop 10 factors for test patient 0:"
    )

    print(
        patient_explanation.head(10)
        .to_string(index=False)
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()