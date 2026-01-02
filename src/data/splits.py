import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_config():
    with open("configs/data.yaml", "r") as f:
        return yaml.safe_load(f)

def create_splits():
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # This now matches the output of validation.py
    input_path = "data/processed/train_validated.csv"
    
    if not os.path.exists(input_path):
        logging.error(f"CRITICAL: {input_path} not found! Check validation.py logic.")
        return

    df = pd.read_csv(input_path)
    target = config['schema']['target']

    # 1. Separate 'Future' Drift set (15%)
    train_test_df, validation_df = train_test_split(
        df, test_size=0.15, stratify=df[target], random_state=42
    )

    # 2. Separate Final Train and Test (80/20 of the remainder)
    train_df, test_df = train_test_split(
        train_test_df, test_size=0.20, stratify=train_test_df[target], random_state=42
    )

    # Save everything to data/processed/ (except the drift test)
    train_df.to_csv("data/processed/train_final.csv", index=False)
    test_df.to_csv("data/processed/test_final.csv", index=False)
    
    # Save drift test to data/validation/ as intended
    os.makedirs("data/validation", exist_ok=True)
    validation_df.to_csv("data/validation/drift_test.csv", index=False)

    logging.info("Splits created and stored in data/processed/ and data/validation/")
    '''
    config = load_config()
    target = config['schema']['target']
    
    # We split the 'validated' file to ensure outliers are already gone
    input_path = "data/processed/train_validated.csv"
    
    if not os.path.exists(input_path):
        logging.error(f"Validated data not found. Run validation.py first.")
        return

    df = pd.read_csv(input_path)

    # 1. First Split: Separate the "Future" Validation/Drift set (15%)
    # This simulates data the model has never seen, even during testing.
    train_test_df, validation_df = train_test_split(
        df, 
        test_size=0.15, 
        stratify=df[target], 
        random_state=42
    )

    # 2. Second Split: Separate Training (80% of remainder) and Testing (20% of remainder)
    # The 'Test' set is for your final evaluation before deployment.
    train_df, test_df = train_test_split(
        train_test_df, 
        test_size=0.20, 
        stratify=train_test_df[target], 
        random_state=42
    )

    # 3. Save to paths defined by architect logic
    train_df.to_csv("data/processed/train_final.csv", index=False)
    test_df.to_csv("data/processed/test_final.csv", index=False)
    validation_df.to_csv("data/validation/drift_test.csv", index=False)

    logging.info("Data splitting complete:")
    logging.info(f" - Train Final: {train_df.shape}")
    logging.info(f" - Test Final: {test_df.shape}")
    logging.info(f" - Drift Validation: {validation_df.shape}")
    '''

if __name__ == "__main__":
    create_splits()