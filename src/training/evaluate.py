##evaluate.py

import joblib
import json
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from src.features.feature_store import FeatureStore

def run_evaluation():
    # 1. Load Model and Test Data
    model = joblib.load("models/xgboost_model.joblib")
    store = FeatureStore()
    test_df = store.load_features("test_processed")
    
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']

    # 2. Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 3. Calculate Metrics
    metrics = {
        "f1_score": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    # 4. Save Report
    with open("models/evaluation_report.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("Evaluation Report Generated:", metrics)
    return metrics

if __name__ == "__main__":
    run_evaluation()