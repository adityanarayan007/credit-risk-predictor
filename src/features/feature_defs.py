##feature_defs.py

import pandas as pd
import numpy as np
import yaml
import os
from sklearn.base import BaseEstimator, TransformerMixin

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, config_path="configs/features.yaml"):
        # THE FIX: Store the attribute so the object knows its own config path
        self.config_path = config_path
        
        # Load the config immediately
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.feature_config = self.config['feature_engineering']['ratios_to_create']
        else:
            self.feature_config = []
    def set_output(self, transform=None):
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # Apply ratios defined in YAML
        for ratio in self.feature_config:
            name = ratio['name']
            num = ratio['numerator']
            den = ratio['denominator']
            
            # Use np.where to handle division by zero safely
            X_out[name] = X_out[num] / X_out[den].replace(0, np.nan)
            # Fill NaNs resulting from 0/0 or division by zero with 0
            X_out[name] = X_out[name].fillna(0)

        # Static ratio: Credit history relative to age
        if 'cb_person_cred_hist_length' in X_out.columns and 'person_age' in X_out.columns:
            X_out['cred_hist_age_ratio'] = X_out['cb_person_cred_hist_length'] / X_out['person_age']
            X_out['cred_hist_age_ratio'] = X_out['cred_hist_age_ratio'].fillna(0)
        
        return X_out

def run_feature_definitions():
    # Test the class logic
    train_df = pd.read_csv("data/processed/train_final.csv")
    engineer = CreditFeatureEngineer()
    
    # Transform
    train_enriched = engineer.transform(train_df)
    
    print(f"Features before: {train_df.shape[1]}")
    print(f"Features after: {train_enriched.shape[1]}")
    print("New features added:", [c for c in train_enriched.columns if c not in train_df.columns])
    
    return train_enriched

if __name__ == "__main__":
    run_feature_definitions()
