##Ingestion.py

import pandas as pd
import yaml
import shutil
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_config():
    with open("configs/data.yaml", "r") as f:
        return yaml.safe_load(f)

def run_ingestion():
    config = load_config()
    source_path = config['data_paths']['raw_path']
    target_dir = "data/raw"
    
    # 1. Create directory structure
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/validation", exist_ok=True)

    # 2. Check if source exists
    if not os.path.exists(source_path):
        logging.error(f"Source file not found at {source_path}. Please download the Kaggle dataset.")
        return

    # 3. Load and Save Initial Copy
    # We load and save to ensure we have a 'working copy' in our raw folder
    df = pd.read_csv(source_path)
    raw_store_path = os.path.join(target_dir, "credit_risk_raw.csv")
    df.to_csv(raw_store_path, index=False)
    
    logging.info(f"Ingestion successful. Raw data stored at {raw_store_path}")

if __name__ == "__main__":
    run_ingestion()