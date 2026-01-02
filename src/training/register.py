##register.py

import os
import shutil
import json
import logging

def register_model(threshold=0.80):
    # 1. Load latest report
    with open("models/evaluation_report.json", "r") as f:
        metrics = json.load(f)

    # 2. Check if performance meets the 'Gate'
    if metrics['f1_score'] >= threshold:
        os.makedirs("models/registry", exist_ok=True)
        # Versioning by simple timestamp or 'v1'
        shutil.copy("models/xgboost_model.joblib", "models/registry/credit_model_latest.joblib")
        logging.info(f"Model Registered! F1 Score {metrics['f1_score']} passed threshold {threshold}")
    else:
        logging.warning("Model performance too low. Registration skipped.")

if __name__ == "__main__":
    register_model()