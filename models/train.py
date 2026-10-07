from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "artifacts"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load processed data
# ---------------------------------------------------------

def load_data():
    """Load preprocessed train, validation and test datasets."""

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0]

    X_val = pd.read_csv(PROCESSED_DIR / "X_validation.csv")
    y_val = pd.read_csv(PROCESSED_DIR / "y_validation.csv").iloc[:, 0]

    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_model(model, X, y, threshold=0.5):
    """
    Calculate classification and probability-based metrics.

    The threshold controls when a predicted probability
    becomes a positive readmission prediction.
    """

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y, probabilities)
        ),
        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "brier_score": float(
            brier_score_loss(y, probabilities)
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# ---------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------

def find_best_threshold(model, X_val, y_val):
    """
    Select the probability threshold that gives the
    highest F1 score on the validation set.

    The test set is not used for threshold selection.
    """

    thresholds = [
        round(threshold, 2)
        for threshold in [
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
        ]
    ]

    threshold_results = []

    for threshold in thresholds:

        metrics = evaluate_model(
            model,
            X_val,
            y_val,
            threshold=threshold,
        )

        threshold_results.append(metrics)

    threshold_results_df = pd.DataFrame(threshold_results)

    best_row = threshold_results_df.loc[
        threshold_results_df["f1_score"].idxmax()
    ]

    best_threshold = float(best_row["threshold"])

    threshold_results_df.to_csv(
        ARTIFACT_DIR / "threshold_analysis.csv",
        index=False,
    )

    return best_threshold, threshold_results_df


# ---------------------------------------------------------
# Model definitions
# ---------------------------------------------------------

def build_models():
    """Create the four candidate ML models."""

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),

        "lightgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        ),
    }


# ---------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("HOSPITAL READMISSION ML MODEL TRAINING")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print("\nDataset shapes:")
    print(f"Train      : {X_train.shape}")
    print(f"Validation : {X_val.shape}")
    print(f"Test       : {X_test.shape}")

    models = build_models()

    validation_results = {}
    trained_models = {}

    # -----------------------------------------------------
    # Train candidate models
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            model,
            X_val,
            y_val,
            threshold=0.5,
        )

        validation_results[name] = metrics
        trained_models[name] = model

        print(
            f"ROC-AUC     : "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC      : "
            f"{metrics['pr_auc']:.4f}"
        )

        print(
            f"Precision   : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Sensitivity : "
            f"{metrics['recall_sensitivity']:.4f}"
        )

        print(
            f"F1 Score    : "
            f"{metrics['f1_score']:.4f}"
        )

        print(
            f"Brier Score : "
            f"{metrics['brier_score']:.4f}"
        )

    # -----------------------------------------------------
    # Model selection
    # -----------------------------------------------------

    best_model_name = max(
        validation_results,
        key=lambda name:
        validation_results[name]["roc_auc"],
    )

    best_base_model = trained_models[
        best_model_name
    ]

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame(
        validation_results
    ).T

    display_columns = [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "brier_score",
    ]

    print(
        comparison[display_columns]
        .sort_values(
            "roc_auc",
            ascending=False,
        )
        .round(4)
    )

    print(
        f"\nSelected model: "
        f"{best_model_name}"
    )

    # -----------------------------------------------------
    # Probability calibration
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("PROBABILITY CALIBRATION")
    print("=" * 60)

    calibrated_model = CalibratedClassifierCV(
        estimator=best_base_model,
        method="sigmoid",
        cv=3,
    )

    calibrated_model.fit(
        X_train,
        y_train,
    )

    calibrated_validation_metrics = evaluate_model(
        calibrated_model,
        X_val,
        y_val,
        threshold=0.5,
    )

    print("\nCalibrated validation metrics:")

    print(
        f"ROC-AUC     : "
        f"{calibrated_validation_metrics['roc_auc']:.4f}"
    )

    print(
        f"Brier Score : "
        f"{calibrated_validation_metrics['brier_score']:.4f}"
    )

    # -----------------------------------------------------
    # Threshold analysis
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION THRESHOLD ANALYSIS")
    print("=" * 60)

    best_threshold, threshold_results = (
        find_best_threshold(
            calibrated_model,
            X_val,
            y_val,
        )
    )

    print(
        "\nThreshold results:"
    )

    print(
        threshold_results[
            [
                "threshold",
                "precision",
                "recall_sensitivity",
                "specificity",
                "f1_score",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    best_threshold_row = threshold_results.loc[
        threshold_results["threshold"]
        == best_threshold
    ].iloc[0]

    print(
        f"\nSelected threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation precision   : "
        f"{best_threshold_row['precision']:.4f}"
    )

    print(
        f"Validation sensitivity : "
        f"{best_threshold_row['recall_sensitivity']:.4f}"
    )

    print(
        f"Validation specificity : "
        f"{best_threshold_row['specificity']:.4f}"
    )

    print(
        f"Validation F1          : "
        f"{best_threshold_row['f1_score']:.4f}"
    )

    # -----------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    print(
        f"Using validation-selected "
        f"threshold: {best_threshold:.2f}"
    )

    test_metrics = evaluate_model(
        calibrated_model,
        X_test,
        y_test,
        threshold=best_threshold,
    )

    for metric, value in test_metrics.items():
        print(
            f"{metric:20s}: {value}"
        )

    # -----------------------------------------------------
    # Save models
    # -----------------------------------------------------

    joblib.dump(
        best_base_model,
        ARTIFACT_DIR /
        "best_base_model.joblib",
    )

    joblib.dump(
        calibrated_model,
        ARTIFACT_DIR /
        "calibrated_model.joblib",
    )

    # -----------------------------------------------------
    # Save evaluation results
    # -----------------------------------------------------

    results = {
        "selected_model": best_model_name,

        "validation_results":
            validation_results,

        "calibrated_validation_results":
            calibrated_validation_metrics,

        "selected_threshold":
            best_threshold,

        "threshold_selection_metric":
            "f1_score",

        "test_results":
            test_metrics,

        "training_config": {
            "random_state": 42,
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "test_rows": len(X_test),
            "num_features": X_train.shape[1],
        },
    }

    with open(
        ARTIFACT_DIR /
        "evaluation_results.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Save metadata
    # -----------------------------------------------------

    metadata = {
        "model_type":
            best_model_name,

        "calibration":
            "sigmoid",

        "selected_threshold":
            best_threshold,

        "threshold_selection_metric":
            "f1_score",

        "num_features":
            X_train.shape[1],

        "feature_columns":
            X_train.columns.tolist(),

        "random_state":
            42,
    }

    with open(
        ARTIFACT_DIR /
        "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Complete
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "\nArtifacts saved to:"
    )

    print(ARTIFACT_DIR)


if __name__ == "__main__":
    main()