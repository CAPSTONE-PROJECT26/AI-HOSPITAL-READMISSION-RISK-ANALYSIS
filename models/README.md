# ML Models and Evaluation

## Overview

This module is responsible for training, comparing, calibrating, and evaluating machine learning models for the 30-day hospital readmission prediction task.

The following four models are evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM

The dataset is split into:

- Training set: 1,750 records
- Validation set: 375 records
- Test set: 375 records

The test set is kept separate and is used only for final evaluation.

---

## Training Pipeline

The training pipeline performs the following steps:

1. Load preprocessed training, validation, and test datasets.
2. Train four candidate classification models.
3. Evaluate each model on the validation set.
4. Compare models using:
   - ROC-AUC
   - PR-AUC
   - Precision
   - Sensitivity/Recall
   - Specificity
   - F1-score
   - Brier score
5. Select the model with the highest validation ROC-AUC.
6. Apply sigmoid probability calibration to the selected model.
7. Perform probability-threshold analysis on the validation set.
8. Select the threshold that produces the highest validation F1-score.
9. Evaluate the calibrated model once on the independent test set using the selected threshold.
10. Save the trained models, evaluation results, metadata, and threshold analysis.

---

## Validation Model Comparison

| Model | ROC-AUC | PR-AUC | Precision | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7495 | 0.4589 | 0.3613 | 0.7089 | 0.6655 | 0.4786 |
| XGBoost | 0.7379 | 0.4337 | 0.5455 | 0.3038 | 0.9324 | 0.3902 |
| Random Forest | 0.7270 | 0.4084 | 0.5429 | 0.2405 | 0.9459 | 0.3333 |
| LightGBM | 0.7211 | 0.4391 | 0.4286 | 0.3797 | 0.8649 | 0.4027 |

### Selected Model

Logistic Regression was selected because it achieved the highest validation ROC-AUC:

**ROC-AUC: 0.7495**

---

## Probability Calibration

The selected Logistic Regression model is calibrated using sigmoid calibration.

### Validation Results

Before calibration:

**Brier score: 0.2170**

After calibration:

**Brier score: 0.1437**

ROC-AUC after calibration:

**0.7487**

The improvement in Brier score indicates better probability calibration while maintaining similar discrimination performance.

---

## Threshold Analysis

The default classification threshold of 0.50 was not used blindly.

Multiple probability thresholds were evaluated on the validation set.

The threshold was selected using the validation F1-score.

### Selected Threshold

**0.20**

Validation performance at threshold 0.20:

- Precision: 34.71%
- Sensitivity: 74.68%
- Specificity: 62.50%
- F1-score: 47.39%

The test set was not used during threshold selection.

---

## Final Test Evaluation

Using the calibrated Logistic Regression model and the validation-selected threshold of 0.20:

- ROC-AUC: 0.7250
- PR-AUC: 0.4653
- Accuracy: 66.13%
- Precision: 34.00%
- Sensitivity: 64.56%
- Specificity: 66.55%
- F1-score: 44.54%
- Brier score: 0.1433

### Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 197 | 99 |
| Actual Positive | 28 | 51 |

Therefore:

- True Negatives: 197
- False Positives: 99
- False Negatives: 28
- True Positives: 51

---

## Generated Artifacts

Training generates the following files inside `models/artifacts/`:

```text
best_base_model.joblib
calibrated_model.joblib
evaluation_results.json
metadata.json
threshold_analysis.csv