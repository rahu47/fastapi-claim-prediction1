from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load model & scaler
model = load_model("claim_ann_model.h5")
scaler = joblib.load("scaler.pkl")

# Initialize FastAPI app
app = FastAPI(title="Customer Claim Prediction API")

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "✅ Customer Claim Prediction API is running!",
        "docs_url": "/docs",
        "predict_example": {
            "age": 35,
            "annual_income": 50000,
            "health_score": 7,
            "number_of_dependents": 2,
            "is_smoker": 0
        }
    }

# Input schema
class CustomerInput(BaseModel):
    age: int
    annual_income: float
    health_score: int
    number_of_dependents: int
    is_smoker: int

# Prediction endpoint
@app.post("/predict")
def predict(data: CustomerInput):
    # Convert input to numpy
    input_data = np.array([[data.age, data.annual_income, data.health_score,
                            data.number_of_dependents, data.is_smoker]])
    
    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict probability
    prob = model.predict(input_scaled)[0][0]
    
    return {"probability_of_claim": float(prob)}




