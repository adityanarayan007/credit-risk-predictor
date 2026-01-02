##feature_store.py

import pandas as pd
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

class FeatureStore:
    def __init__(self, base_path="data/processed"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def save_features(self, df, name, version=None):
        """
        Saves a feature set to the store as a Parquet file.
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M")
        
        filename = f"{name}_v{version}.parquet"
        path = os.path.join(self.base_path, filename)
        
        # Save as Parquet for efficiency
        df.to_parquet(path, index=False)
        
        # Also maintain a 'latest' pointer for the training script
        latest_path = os.path.join(self.base_path, f"{name}_latest.parquet")
        df.to_parquet(latest_path, index=False)
        
        logging.info(f"Stored feature set '{name}' to {path}")

    def load_features(self, name, version='latest'):
        """
        Retrieves a feature set by name and version.
        """
        filename = f"{name}_{version}.parquet"
        path = os.path.join(self.base_path, filename)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature set {filename} not found in store.")
            
        return pd.read_parquet(path)

def run_feature_store_demo(train_enriched_df):
    store = FeatureStore()
    
    # Save the training features we just created
    store.save_features(train_enriched_df, "credit_risk_train")
    
    # Reload them to verify
    df_reloaded = store.load_features("credit_risk_train", version='latest')
    print(f"Reloaded {df_reloaded.shape[0]} rows from the Feature Store.")