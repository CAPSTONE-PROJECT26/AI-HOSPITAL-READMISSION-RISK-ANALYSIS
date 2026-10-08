from pathlib import Path

import joblib
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer


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
# LOAD DATA
# =========================================================

def load_data():
    """Load training and test data."""

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv"
    )

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv"
    )

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    ).iloc[:, 0]

    return X_train, X_test, y_test


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_probability(model, data):
    """
    Return class probabilities.

    LIME requires a prediction function that accepts
    a matrix of samples and returns probabilities
    for each class.
    """

    return model.predict_proba(data)


# =========================================================
# CREATE LIME EXPLAINER
# =========================================================

def create_explainer(X_train):
    """
    Create a LIME tabular explainer.

    Training data is used as the reference distribution
    for generating local perturbations.
    """

    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=[
            "No Readmission",
            "Readmission"
        ],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )


# =========================================================
# EXPLAIN ONE PATIENT
# =========================================================

def explain_patient(
    explainer,
    model,
    X_test,
    patient_index
):
    """
    Generate a LIME explanation for one patient.
    """

    patient = (
        X_test.iloc[patient_index]
        .values
    )

    explanation = (
        explainer.explain_instance(
            patient,
            lambda data:
            predict_probability(
                model,
                pd.DataFrame(
                    data,
                    columns=X_test.columns
                )
            ),
            num_features=10
        )
    )

    return explanation


# =========================================================
# CONVERT LIME OUTPUT
# =========================================================

def format_explanation(explanation):
    """
    Convert LIME output into a structured DataFrame.
    """

    results = []

    for feature, weight in (
        explanation.as_list()
    ):

        results.append({
            "feature": feature,
            "weight": weight,
            "direction": (
                "increases risk"
                if weight > 0
                else "decreases risk"
            )
        })

    return pd.DataFrame(results)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("LIME EXPLAINABILITY")
    print("=" * 60)

    # Load model
    model = load_model()

    # Load data
    X_train, X_test, y_test = load_data()

    print(
        f"\nModel: "
        f"{model.__class__.__name__}"
    )

    print(
        f"Training samples: "
        f"{X_train.shape[0]}"
    )

    print(
        f"Test samples: "
        f"{X_test.shape[0]}"
    )

    print(
        f"Features: "
        f"{X_test.shape[1]}"
    )

    # Create LIME explainer
    explainer = create_explainer(
        X_train
    )

    # Explain first test patient
    patient_index = 0

    explanation = explain_patient(
        explainer,
        model,
        X_test,
        patient_index
    )

    # Format explanation
    results = format_explanation(
        explanation
    )

    print(
        "\nLIME explanation generated "
        "successfully."
    )

    print(
        "\nTop factors for test patient 0:"
    )

    print(
        results.to_string(
            index=False
        )
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()