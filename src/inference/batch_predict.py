## Batch Prediction

import pandas as pd
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_batch_inference(input_path, output_path):
    # 1. Load the Production Artifacts
    # We load the REGISTERED model (the one that passed the gatekeeper)
    model = joblib.load("models/registry/credit_model_latest.joblib")
    pipeline = joblib.load("models/preprocessing_pipeline.joblib")
    
    # 2. Load Raw Data
    if not os.path.exists(input_path):
        logging.error(f"Input file {input_path} not found.")
        return

    df_raw = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df_raw)} records for inference.")

    # 3. Transform Data
    # The pipeline handles ratios, imputation, and encoding
    X_processed = pipeline.transform(df_raw)
    
    # Get feature names to avoid XGBoost name mismatch
    try:
        feature_names = pipeline.get_feature_names_out()
    except:
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    X_final = pd.DataFrame(X_processed, columns=feature_names)

    # 4. Generate Predictions
    # We save both the hard class (0/1) and the probability (%)
    df_raw['default_probability'] = model.predict_proba(X_final)[:, 1]
    df_raw['prediction'] = (df_raw['default_probability'] > 0.5).astype(int)

    # 5. Save Results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_raw.to_csv(output_path, index=False)
    logging.info(f"Batch inference complete. Results saved to {output_path}")

if __name__ == "__main__":
    # Test it on our holdout set
    run_batch_inference("data/validation/drift_test.csv", "data/predictions/batch_results.csv")