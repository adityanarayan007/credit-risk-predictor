##train.py

import yaml
import joblib
import pandas as pd
from xgboost import XGBClassifier
from src.features.feature_store import FeatureStore

def run_training():
    # 1. Load Config and Data
    with open("configs/training.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    store = FeatureStore()
    train_df = store.load_features("train_processed")
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']

    # 2. Train Model
    print("Training final model...")
    model = XGBClassifier(**config['model']['params'])
    model.fit(X_train, y_train)

    # 3. Save Model
    joblib.dump(model, "models/xgboost_model.joblib")
    print("Model saved to models/xgboost_model.joblib")

if __name__ == "__main__":
    run_training()