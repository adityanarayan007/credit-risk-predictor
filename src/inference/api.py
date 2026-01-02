from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import uvicorn
import os

app = FastAPI(
    title="LendGuard: Automated Loan Approval API",
    description="An AI-powered system for real-time loan eligibility and risk assessment.",
    version="1.0.0"
)

# --- 1. PATH RESOLUTION ---
# This ensures paths work on Windows, Mac, and Linux (Render)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_DIR, "models", "registry", "credit_model_latest.joblib")
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "preprocessing_pipeline.joblib")

# Load artifacts
MODEL = joblib.load(MODEL_PATH)
PIPELINE = joblib.load(PIPELINE_PATH)

# --- 2. INPUT SCHEMA ---
class LoanApplication(BaseModel):
    person_age: int = Field(default=25, ge=18, le=100, description="Age of the applicant (18-100)")
    person_income: int = Field(default=50000, gt=0, description="Annual income in USD")
    person_home_ownership: str = Field(default="RENT", description="Options: RENT, MORTGAGE, OWN, OTHER")
    person_emp_length: float = Field(default=2.0, ge=0, le=60, description="Years of employment")
    loan_intent: str = Field(default="EDUCATION", description="Options: EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION")
    loan_grade: str = Field(default="B", description="Internal risk grade (A - G)")
    loan_amnt: int = Field(default=10000, gt=0, description="Requested loan amount")
    loan_int_rate: float = Field(default=11.1, ge=0, description="Agreed interest rate percentage")
    cb_person_default_on_file: str = Field(default="N", description="Historical default record (Y/N)")
    cb_person_cred_hist_length: int = Field(default=3, ge=0, description="Length of credit history")

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 30,
                "person_income": 65000,
                "person_home_ownership": "MORTGAGE",
                "person_emp_length": 5.0,
                "loan_intent": "VENTURE",
                "loan_grade": "A",
                "loan_amnt": 5000,
                "loan_int_rate": 10.5,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 7
            }
        }
    }

# --- 3. ENDPOINTS ---

@app.get("/")
def home():
    """Welcome page to prevent 404 on root URL."""
    return {
        "project": "LendGuard: Automated Loan Approval Platform",
        "status": "online",
        "documentation": "/docs",
        "version": "1.0.0"
    }

@app.post("/predict")
def predict(app_data: LoanApplication):
    try:
        # Convert to dict and then DataFrame
        data_dict = app_data.model_dump()
        input_df = pd.DataFrame([data_dict])
        
        # Auto-calculate derived feature
        input_df['loan_percent_income'] = input_df['loan_amnt'] / input_df['person_income']
        
        # Transform using the production pipeline
        X_processed = PIPELINE.transform(input_df)
        
        # Extract feature names for XGBoost compatibility
        try:
            feature_names = PIPELINE.get_feature_names_out()
        except:
            feature_names = PIPELINE.named_steps['preprocessor'].get_feature_names_out()
        
        X_final = pd.DataFrame(X_processed, columns=feature_names)
        
        # Model Inference
        prob = float(MODEL.predict_proba(X_final)[0][1])
        prediction = 1 if prob > 0.5 else 0
        
        return {
            "application_status": "REJECT" if prediction == 1 else "APPROVE",
            "risk_score": round(prob * 100, 2),
            "probability": round(prob, 4)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)