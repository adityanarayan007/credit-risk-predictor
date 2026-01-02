##validation.py

import pandas as pd
import yaml
import logging
import os

# Setup logging for the portfolio
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_config(path="configs/data.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def validate_data(df, config):
    """
    Performs data quality checks based on the data.yaml config.
    Returns: Cleaned DataFrame and a boolean indicating if it passed.
    """
    is_valid = True
    
    # 1. Column Check
    required_cols = config['schema']['numerical_features'] + \
                    config['schema']['categorical_features'] + \
                    [config['schema']['target']]
    
    if not set(required_cols).issubset(df.columns):
        missing = set(required_cols) - set(df.columns)
        logging.error(f"Missing columns: {missing}")
        return df, False

    # 2. Logical Outlier Removal (Using YAML thresholds)
    # Filter Age
    max_age = config['validation']['max_age']
    age_outliers = df[df['person_age'] > max_age]
    if not age_outliers.empty:
        logging.warning(f"Found {len(age_outliers)} rows where Age > {max_age}. Dropping.")
        df = df[df['person_age'] <= max_age]

    # Filter Employment Length
    max_emp = config['validation']['max_emp_length']
    emp_outliers = df[df['person_emp_length'] > max_emp]
    if not emp_outliers.empty:
        logging.warning(f"Found {len(emp_outliers)} rows where Emp Length > {max_emp}. Dropping.")
        df = df[df['person_emp_length'] <= max_emp]
        
    # Logic check: Employment length cannot exceed age
    # We assume people start working at 14 at the absolute earliest
    logic_error = df[df['person_emp_length'] > (df['person_age'] - 14)]
    if not logic_error.empty:
        logging.warning(f"Found {len(logic_error)} rows where Emp Length > (Age - 14). Dropping.")
        df = df.drop(logic_error.index)

    return df, is_valid

def run_validation():
    #config = load_config()
    with open("configs/data.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load the raw data we just ingested
    raw_path = "data/raw/credit_risk_raw.csv"
    if not os.path.exists(raw_path):
        logging.error(f"Raw data not found at {raw_path}")
        return

    df = pd.read_csv(raw_path)
    
    # --- VALIDATION LOGIC ---
    # (Age check, Emp Length check as we wrote before)
    df = df[df['person_age'] <= config['validation']['max_age']]
    df = df[df['person_emp_length'] <= config['validation']['max_emp_length']]
    
    # --- THE FIX: Save to PROCESSED folder ---
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/train_validated.csv"
    df.to_csv(output_path, index=False)
    
    logging.info(f"Validation complete. Cleaned data saved to {output_path}")
    """
    # Validate both Train and Test raw files
    for split in ['train_raw.csv', 'test_raw.csv']:
        path = os.path.join('data/raw', split)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df_clean, success = validate_data(df, config)
            
            if success:
                # Save to processed as "validated" but not yet "engineered"
                output_path = os.path.join('data/processed', split.replace('raw', 'validated'))
                os.makedirs('data/processed', exist_ok=True)
                df_clean.to_csv(output_path, index=False)
                logging.info(f"Successfully validated {split} -> saved to {output_path}")"""

if __name__ == "__main__":
    run_validation()