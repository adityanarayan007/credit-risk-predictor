## Data Drift Detection
## it is actually not required as model_drift performance is better, yet adding just for reference

import pandas as pd
import yaml
import logging
from scipy.stats import ks_2samp
import json
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_config():
    with open("configs/data.yaml", "r") as f:
        return yaml.safe_load(f)

def check_data_drift():
    config = load_config()
    
    # 1. Load Baseline (Train) and Current (Drift/Future) Data
    # We compare the RAW features before they were transformed
    train_df = pd.read_csv("data/processed/train_final.csv")
    current_df = pd.read_csv("data/validation/drift_test.csv")
    
    numerical_features = config['schema']['numerical_features']
    drift_report = {}
    drift_detected = False

    logging.info("--- Running Data Drift Analysis (KS Test) ---")

    for feature in numerical_features:
        # 2. Perform Kolmogorov-Smirnov test
        # Null Hypothesis: The two samples are drawn from the same distribution
        stat, p_value = ks_2samp(train_df[feature], current_df[feature])
        
        # P-value < 0.05 means we reject the null hypothesis (Drift exists)
        is_drifted = p_value < 0.05
        drift_report[feature] = {
            "p_value": float(p_value),
            "drift_detected": bool(is_drifted)
        }
        
        if is_drifted:
            logging.warning(f"🚨 DRIFT DETECTED in '{feature}' (p={p_value:.4f})")
            drift_detected = True
        else:
            logging.info(f"✅ Feature '{feature}' is stable (p={p_value:.4f})")

    # 3. Save the report
    os.makedirs("models/monitoring", exist_ok=True)
    with open("models/monitoring/data_drift_report.json", "w") as f:
        json.dump(drift_report, f, indent=4)

    if not drift_detected:
        logging.info("Overall Result: Data remains stable.")
    
    return drift_detected

if __name__ == "__main__":
    check_data_drift()