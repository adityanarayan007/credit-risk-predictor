# 💳 AI-Driven Credit Decision Engine

An enterprise-grade machine learning pipeline designed to predict loan default risk. This project goes beyond modeling by implementing a full MLOps lifecycle, including automated data validation, feature engineering, hyperparameter tuning, and a complete monitoring suite (Data & Model Drift).

## 🏗️ System Architecture

The platform is built with a modular, "Separation of Concerns" architecture:

- **Data Layer:** Automated ingestion, validation (sanity checks), and stratified splitting.
- **Feature Layer:** Custom Scikit-Learn transformers for financial ratio engineering (Loan-to-Income, etc.) and a Parquet-based Feature Store.
- **Training Pipeline:** Automated hyperparameter optimization (RandomizedSearchCV) with a registration gatekeeper.
- **Monitoring Suite:** Statistical drift detection (Kolmogorov-Smirnov test) and performance decay alerting.
- **Serving Layer:** Real-time REST API via FastAPI and high-throughput Batch Inference.

## 🚀 Key Features

- **Production-Ready Feature Engineering:** Custom transformers ensure no training-serving skew. Ratios are calculated identically in training and production.
- **Automated Model Gatekeeping:** The `register.py` script prevents models from being deployed if they don't meet a specific F1-score threshold (currently set to 0.80).
- **Drift Monitoring:** Early warning systems detect when the population's financial behavior changes (Data Drift) or when model accuracy drops (Model Drift).
- **Deterministic Workflow:** Every step of the pipeline is tracked via a centralized logger and a reproducible Makefile.

## 📊 Performance Summary

- **Best Model:** XGBoost Classifier
- **Baseline F1-Score:** 0.8266
- **Validation (Drift Set) F1-Score:** 0.8378
- **Key Decision Metrics:** Prioritizes Recall for high-risk grades (D-G) to minimize bank capital loss.

## 📁 Repository Structure

```plaintext
├── configs/            # YAML configurations for features/training
├── data/               # Raw, Processed, and Validation data (gitignored)
├── models/             # Model registry and evaluation reports
├── src/
│   ├── data/           # Ingestion, Validation, Splits
│   ├── features/       # Engineering logic & Feature Store
│   ├── training/       # Tuning, Training, Evaluation, Registration
│   ├── monitoring/     # Data/Model Drift & Alerting logic
│   └── inference/      # FastAPI & Batch Prediction
├── run_pipeline.py     # End-to-end orchestrator
└── Makefile            # Shortcut commands for the system
```

## 🛠️ Installation & Usage

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

This command executes ingestion, cleaning, splitting, feature engineering, tuning, and training in one click:

```bash
make run
# or
python run_pipeline.py
```

### 3. Start the API

```bash
python -m src.inference.api
```

Once running, visit `http://127.0.0.1:8000/docs` to interact with the Swagger UI.

> **🌐 Live Demo:** You can access the public API endpoint here: [https://credit-risk-predictor-rsyl.onrender.com/docs](Credit Decision Engine)

### 4. Run Monitoring

```bash
make monitor
```

## 🛡️ Monitoring & Governance

The platform includes a proactive monitoring suite:

- **Data Drift:** Uses the KS-Test to compare feature distributions. If a p-value drops below 0.05, a warning is triggered.
- **Model Drift:** Compares production performance against the training baseline.
- **Alerts:** All findings are logged to `models/monitoring/alerts.log` for auditability.