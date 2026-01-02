import joblib
import pandas as pd
import json
import logging
from sklearn.metrics import f1_score

def check_model_drift():
    # 1. Load the "Production" Model and the Pipeline
    model = joblib.load("models/registry/credit_model_latest.joblib")
    pipeline = joblib.load("models/preprocessing_pipeline.joblib")
    
    # 2. Load the "Future/Validation" Data
    # This data has NOT been through the pipeline yet
    drift_df = pd.read_csv("data/validation/drift_test.csv")
    X_drift = drift_df.drop(columns=['loan_status'])
    y_drift = drift_df['loan_status']
    
    # 3. Transform the "Future" data using the saved pipeline
    # We use only .transform() to simulate real-world arrival
    X_drift_proc = pipeline.transform(X_drift)

    try:
        # For newer sklearn versions
        feature_names = pipeline.get_feature_names_out()
    except:
        # For older sklearn versions, get names from the 'preprocessor' step
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Create the DataFrame using the EXACT names the model expects
    X_drift_df = pd.DataFrame(X_drift_proc, columns=feature_names)
    
    """# Convert to DataFrame with string columns (to match training)
    X_drift_df = pd.DataFrame(X_drift_proc)
    X_drift_df.columns = [str(i) for i in range(X_drift_df.shape[1])]"""
    
    # 4. Predict and Score
    y_pred = model.predict(X_drift_df)
    current_f1 = f1_score(y_drift, y_pred)
    
    # 5. Compare against Training Baseline
    with open("models/evaluation_report.json", "r") as f:
        baseline = json.load(f)
    baseline_f1 = baseline['f1_score']
    
    drift_magnitude = baseline_f1 - current_f1
    
    logging.info(f"Baseline F1: {baseline_f1:.4f}")
    logging.info(f"Drift Set F1: {current_f1:.4f}")
    
    if drift_magnitude > 0.05:
        logging.warning(f"🚨 MODEL DRIFT DETECTED! Performance dropped by {drift_magnitude:.4f}")
    else:
        logging.info("✅ No significant model drift detected.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_model_drift()