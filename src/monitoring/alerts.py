## Alerts
import json
import os
import logging

# Set up logging to both console and a specific alert file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - 🚨 ALERT_SYSTEM: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("models/monitoring/alerts.log")
    ]
)

def load_report(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def run_alerts():
    # 1. Check Data Drift Report
    data_drift = load_report("models/monitoring/data_drift_report.json")
    if data_drift:
        drifted_features = [feat for feat, val in data_drift.items() if val['drift_detected']]
        if drifted_features:
            logging.warning(f"CRITICAL: Data drift detected in features: {drifted_features}")
            # Here you would trigger: send_slack_message(f"Data Drift: {drifted_features}")
        else:
            logging.info("Data drift check passed: All features stable.")

    # 2. Check Model Performance Report (from evaluation or model_drift)
    # Let's assume we saved a specific model_drift_report.json earlier
    model_report = load_report("models/evaluation_report.json")
    if model_report:
        f1_score = model_report.get('f1_score', 0)
        # Define a hard "Service Level Agreement" (SLA) threshold
        SLA_THRESHOLD = 0.80 
        
        if f1_score < SLA_THRESHOLD:
            logging.error(f"CRITICAL: Model performance (F1: {f1_score:.4f}) has fallen below SLA ({SLA_THRESHOLD})")
        else:
            logging.info(f"Model performance is healthy (F1: {f1_score:.4f})")

    logging.info("Alert scan complete.")

if __name__ == "__main__":
    run_alerts()