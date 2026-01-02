## tune.py

import yaml
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from src.features.feature_store import FeatureStore

def run_tuning():
    store = FeatureStore()
    train_df = store.load_features("train_processed")
    
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']

    # 1. Define Search Space
    param_dist = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'scale_pos_weight': [1, 3, 5] # Vital for credit risk imbalance
    }

    # 2. Run Search
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    search = RandomizedSearchCV(xgb, param_dist, n_iter=10, cv=3, scoring='f1', n_jobs=-1)
    search.fit(X_train, y_train)

    # 3. Update Config with Best Params
    with open("configs/training.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    config['model']['params'] = search.best_params_
    
    with open("configs/training.yaml", 'w') as f:
        yaml.dump(config, f)
    
    print(f"Tuning complete. Best params: {search.best_params_}")

if __name__ == "__main__":
    run_tuning()