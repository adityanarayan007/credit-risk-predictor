##run_pipeline.py

import logging
import sys
from src.data.ingestion import run_ingestion
from src.data.validation import run_validation
from src.data.splits import create_splits
from src.features.transformations import run_transformations
from src.training.tune import run_tuning
from src.training.train import run_training
from src.training.evaluate import run_evaluation
from src.training.register import register_model

# Configure logging to show the flow in the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log")
    ]
)

def main():
    try:
        logging.info("🚀 Starting End-to-End Credit Risk Pipeline")

        # --- DATA PHASE ---
        logging.info("Step 1/8: Ingesting Raw Data...")
        run_ingestion()

        logging.info("Step 2/8: Running Data Validation & Outlier Detection...")
        run_validation()

        logging.info("Step 3/8: Creating Stratified Train/Test/Drift Splits...")
        create_splits()

        # --- FEATURE PHASE ---
        logging.info("Step 4/8: Engineering Ratios & Building Transformation Pipeline...")
        run_transformations()

        # --- MODEL PHASE ---
        logging.info("Step 5/8: Starting Hyperparameter Optimization (Tuning)...")
        run_tuning()

        logging.info("Step 6/8: Training Final Model with Optimal Parameters...")
        run_training()

        logging.info("Step 7/8: Evaluating Model on Holdout Test Set...")
        run_evaluation()

        # --- DEPLOYMENT PHASE ---
        logging.info("Step 8/8: Model Registration & Gatekeeping...")
        register_model(threshold=0.80)

        logging.info("✅ Pipeline Completed Successfully!")

    except Exception as e:
        logging.error(f"❌ Pipeline failed! Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()