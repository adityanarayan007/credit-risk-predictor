import pandas as pd
import yaml
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Import our custom modules
from src.features.feature_defs import CreditFeatureEngineer
from src.features.feature_store import FeatureStore

def load_configs():
    with open("configs/data.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/features.yaml", "r") as f:
        feat_cfg = yaml.safe_load(f)
    return data_cfg, feat_cfg

def build_preprocessing_pipeline(data_cfg, feat_cfg):
    """
    Creates a robust pipeline: 
    1. Custom Ratios -> 2. Impute Nulls -> 3. One-Hot Encode
    """
    num_features = data_cfg['schema']['numerical_features']
    cat_features = data_cfg['schema']['categorical_features']
    
    # Define the transformers for numerical data
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Define the transformers for categorical data
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine them into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ],
        remainder='passthrough'
    )

    # The Full Pipeline including our custom ratios from feature_defs.py
    full_pipeline = Pipeline(steps=[
        ('ratios', CreditFeatureEngineer()), # Our custom class!
        ('preprocessor', preprocessor)
    ])
    
    #full_pipeline.set_output(transform="pandas")

    return full_pipeline

def run_transformations():
    data_cfg, feat_cfg = load_configs()
    store = FeatureStore()
    
    # 1. Load Data
    train_df = pd.read_csv("data/processed/train_final.csv")
    test_df = pd.read_csv("data/processed/test_final.csv")
    
    X_train = train_df.drop(columns=[data_cfg['schema']['target']])
    y_train = train_df[data_cfg['schema']['target']]
    
    X_test = test_df.drop(columns=[data_cfg['schema']['target']])
    y_test = test_df[data_cfg['schema']['target']]

    # 2. Build Pipeline (Remove the .set_output line from build_preprocessing_pipeline)
    pipeline = build_preprocessing_pipeline(data_cfg, feat_cfg)
    
    # 3. Fit and Transform
    # This will return a NumPy Array because set_output isn't working
    X_train_raw = pipeline.fit_transform(X_train)
    X_test_raw = pipeline.transform(X_test)

    # 4. MANUALLY RECONSTRUCT DATAFRAME
    # We use the feature names from the preprocessor step
    # This ensures XGBoost gets strings, not integers
    try:
        # Try to get feature names if the version supports it
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    except:
        # Fallback: create generic string names if get_feature_names_out fails
        feature_names = [f"feature_{i}" for i in range(X_train_raw.shape[1])]

    X_train_proc = pd.DataFrame(X_train_raw, columns=feature_names)
    X_test_proc = pd.DataFrame(X_test_raw, columns=feature_names)

    # 5. Attach target and Save
    X_train_proc['target'] = y_train.values
    X_test_proc['target'] = y_test.values

    store.save_features(X_train_proc, "train_processed")
    store.save_features(X_test_proc, "test_processed")
    
    # Save the pipeline for inference
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/preprocessing_pipeline.joblib")
    
    print("✅ Transformation complete. Column names forced to strings.")

if __name__ == "__main__":
    run_transformations()