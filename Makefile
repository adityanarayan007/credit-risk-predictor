# Orchestrate the entire pipeline
run:
	@echo "🚀 Starting Full Credit Risk Pipeline..."
	python run_pipeline.py

# Run only the monitoring suite
monitor:
	@echo "🔍 Running Drift Analysis..."
	python -m src.monitoring.data_drift
	python -m src.monitoring.model_drift
	python -m src.monitoring.alerts

# Start the API
api:
	@echo "🌐 Starting FastAPI Server..."
	python -m src.inference.api

# Clean up temporary files
clean:
	rm -rf __pycache__
	rm -rf src/**/__pycache__
	rm -rf .pytest_cache