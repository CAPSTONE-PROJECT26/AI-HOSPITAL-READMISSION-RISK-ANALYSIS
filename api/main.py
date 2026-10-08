from fastapi import FastAPI, HTTPException

from api.prediction import predict_patient
from api.schemas import PatientInput, PredictionResponse


app = FastAPI(
    title="AI Hospital Readmission Risk API",
    description=(
        "API for predicting 30-day hospital readmission risk "
        "using the trained calibrated ML model."
    ),
    version="1.0.0",
)


@app.get("/health")
def health_check():
    """
    Basic API health check.
    """

    return {
        "status": "healthy",
        "service": "hospital-readmission-risk-api",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(patient: PatientInput):
    """
    Predict 30-day readmission risk for one patient.
    """

    try:
        result = predict_patient(
            patient.model_dump()
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc