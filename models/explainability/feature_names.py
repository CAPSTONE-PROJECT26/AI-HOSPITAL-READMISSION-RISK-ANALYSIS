"""
Clinician-readable feature names.

Maps internal model feature names to human-readable
clinical terminology for SHAP and LIME explanations.
"""

FEATURE_NAME_MAP = {

    # Original numerical features
    "numeric__age": "Age",
    "numeric__length_of_stay": "Length of Stay",
    "numeric__prior_admissions": "Prior Admissions",
    "numeric__medication_count": "Medication Count",
    "numeric__comorbidity_count": "Comorbidity Count",
    "numeric__heart_rate": "Heart Rate",
    "numeric__systolic_bp": "Systolic Blood Pressure",
    "numeric__oxygen_saturation": "Oxygen Saturation",
    "numeric__creatinine": "Creatinine",
    "numeric__glucose": "Glucose",
    "numeric__hemoglobin": "Hemoglobin",

    # Major disease indicators
    "numeric__has_diabetes": "Diabetes",
    "numeric__has_copd": "COPD",
    "numeric__has_heart_failure": "Heart Failure",
    "numeric__has_renal_disease": "Renal Disease",

    # Age-related features
    "numeric__age_75_plus": "Age 75+",

    # Length of stay
    "numeric__length_of_stay_log": "Length of Stay (Log)",
    "numeric__long_stay": "Long Stay",

    # Previous admissions
    "numeric__has_prior_admission": "Has Prior Admission",
    "numeric__frequent_prior_admission": "Frequent Prior Admissions",
    "numeric__prior_admissions_log": "Prior Admissions (Log)",

    # Medication burden
    "numeric__high_medication_burden": "High Medication Burden",
    "numeric__medication_count_log": "Medication Count (Log)",

    # Comorbidity burden
    "numeric__high_comorbidity_burden": "High Comorbidity Burden",
    "numeric__comorbidity_count_log": "Comorbidity Count (Log)",
    "numeric__total_major_comorbidities": "Major Comorbidity Count",
    "numeric__multiple_major_comorbidities": "Multiple Major Comorbidities",

    # Vital signs
    "numeric__abnormal_heart_rate": "Abnormal Heart Rate",
    "numeric__abnormal_systolic_bp": "Abnormal Systolic Blood Pressure",
    "numeric__low_oxygen_saturation": "Low Oxygen Saturation",
    "numeric__vital_instability_count": "Vital Instability Count",
    "numeric__multiple_vital_abnormalities": "Multiple Vital Abnormalities",

    # Laboratory abnormalities
    "numeric__elevated_creatinine": "Elevated Creatinine",
    "numeric__abnormal_glucose": "Abnormal Glucose",
    "numeric__low_hemoglobin": "Low Hemoglobin",

    # Overall clinical risk
    "numeric__clinical_risk_factor_count": "Clinical Risk Factor Count",
    "numeric__multiple_clinical_risk_factors": "Multiple Clinical Risk Factors",

    # Age groups
    "categorical__age_group_elderly": "Age Group: Elderly",
    "categorical__age_group_middle_aged": "Age Group: Middle Aged",
    "categorical__age_group_older": "Age Group: Older",
    "categorical__age_group_young": "Age Group: Young",
}


def get_display_name(feature_name):
    """
    Convert an internal model feature name into
    a clinician-readable name.

    If the feature is not found in the mapping,
    return the original feature name.
    """

    return FEATURE_NAME_MAP.get(
        feature_name,
        feature_name
    )