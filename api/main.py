from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from src.churn_predictor.schemas import PredictionRequest, PredictionResponse

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn.",
    version="0.1.0"
)

# Load model and feature list at startup
MODEL_PATH = os.getenv("MODEL_PATH", "ml_artifacts/lgbm_churn_model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "ml_artifacts/feature_list.joblib")

model = None
feature_list = None

@app.on_event("startup")
def load_model():
    """Load the model and feature list when the API starts."""
    global model, feature_list
    try:
        model = joblib.load(MODEL_PATH)
        feature_list = joblib.load(FEATURES_PATH)
        print("Model and feature list loaded successfully.")
    except FileNotFoundError:
        print("Error: Model or feature list not found. Ensure training has been run.")
        model = None
        feature_list = None

@app.get("/", tags=["Health Check"])
def read_root():
    """Root endpoint for health checks."""
    return {"status": "ok", "message": "Welcome to the Churn Prediction API"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(request: PredictionRequest) -> PredictionResponse:
    """
    Accepts user features and returns a churn prediction.
    """
    if not model or not feature_list:
        return PredictionResponse(
            error="Model not loaded. Please check server logs."
        )

    # Convert request data to a DataFrame
    input_data = pd.DataFrame([request.dict()])
    
    # One-hot encode categorical features to match training columns
    input_data = pd.get_dummies(input_data)
    
    # Align columns with the training data
    input_df_aligned = input_data.reindex(columns=feature_list, fill_value=0)

    # Make prediction
    probability = model.predict_proba(input_df_aligned)[:, 1]
    prediction = int(probability > 0.5)

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=probability
    )
