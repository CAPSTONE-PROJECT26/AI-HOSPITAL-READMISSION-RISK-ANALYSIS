from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")

    age: float = Field(..., ge=0, le=120)
    length_of_stay: float = Field(..., ge=0)
    prior_admissions: float = Field(..., ge=0)
    medication_count: float = Field(..., ge=0)
    comorbidity_count: float = Field(..., ge=0)

    heart_rate: float = Field(..., ge=0)
    systolic_bp: float = Field(..., ge=0)
    oxygen_saturation: float = Field(..., ge=0, le=100)
    creatinine: float = Field(..., ge=0)
    glucose: float = Field(..., ge=0)
    hemoglobin: float = Field(..., ge=0)

    has_diabetes: int = Field(..., ge=0, le=1)
    has_copd: int = Field(..., ge=0, le=1)
    has_heart_failure: int = Field(..., ge=0, le=1)
    has_renal_disease: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    patient_id: str
    readmission_probability: float
    risk_percentage: float
    risk_band: str
    predicted_class: int