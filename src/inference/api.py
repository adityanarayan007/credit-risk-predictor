from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import uvicorn
import os

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Professional API for evaluating loan default probability based on customer financial profiles.",
    version="1.0.0"
)

# Get the absolute path to the directory where api.py lives
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Build absolute paths to artifacts
MODEL_PATH = os.path.join(BASE_DIR, "models/registry/credit_model_latest.joblib")
PIPELINE_PATH = os.path.join(BASE_DIR, "models/preprocessing_pipeline.joblib")

# Load artifacts
MODEL = joblib.load(MODEL_PATH)
PIPELINE = joblib.load(PIPELINE_PATH)

# 1. Clean Schema: 'loan_percent_income' is REMOVED from the user input
class LoanApplication(BaseModel):
    person_age: int = Field(
        default=25, 
        ge=18, le=100, 
        description="Age of the applicant (Range: 18-100)",
        examples=[30]
    )
    person_income: int = Field(
        default=50000, 
        gt=0, 
        description="Annual income in USD",
        examples=[65000]
    )
    person_home_ownership: str = Field(
        default="RENT", 
        description="Options: RENT, MORTGAGE, OWN, OTHER",
        examples=["MORTGAGE"]
    )
    person_emp_length: float = Field(
        default=2.0, 
        ge=0, le=60, 
        description="Years of employment (Range: 0-60)",
        examples=[5.5]
    )
    loan_intent: str = Field(
        default="EDUCATION", 
        description="Options: EDUCATION, MEDICAL, VENTURE, PERSONAL, HOMEIMPROVEMENT, DEBTCONSOLIDATION",
        examples=["VENTURE"]
    )
    loan_grade: str = Field(
        default="B", 
        description="Internal risk grade (A - G)(A is safest, G is riskiest)",
        examples=["A"]
    )
    loan_amnt: int = Field(
        default=10000, 
        gt=0, 
        description="Requested loan amount in USD",
        examples=[5000]
    )
    loan_int_rate: float = Field(
        default=11.1, 
        ge=0, 
        description="Agreed interest rate percentage",
        examples=[10.5]
    )
    cb_person_default_on_file: str = Field(
        default="N", 
        description="Historical default record (Y/N)",
        examples=["N"]
    )
    cb_person_cred_hist_length: int = Field(
        default=3, 
        ge=0, 
        description="Length of credit history in years",
        examples=[7]
    )

    # This class allows you to show a complete example in the Swagger UI
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

@app.post("/predict")
def predict(app_data: LoanApplication):
    try:
        # Convert to dict and then DataFrame
        data_dict = app_data.model_dump()
        input_df = pd.DataFrame([data_dict])
        
        # 2. AUTO-CALCULATION: 
        # We calculate the missing column here so the Pipeline doesn't crash
        input_df['loan_percent_income'] = input_df['loan_amnt'] / input_df['person_income']
        
        # 3. Transform and Predict
        X_processed = PIPELINE.transform(input_df)
        
        # Get names for XGBoost
        try:
            feature_names = PIPELINE.get_feature_names_out()
        except:
            feature_names = PIPELINE.named_steps['preprocessor'].get_feature_names_out()
        
        X_final = pd.DataFrame(X_processed, columns=feature_names)
        
        prob = float(MODEL.predict_proba(X_final)[0][1])
        prediction = 1 if prob > 0.5 else 0
        
        return {
            "application_status": "REJECT" if prediction == 1 else "APPROVE",
            "risk_score": round(prob * 100, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)