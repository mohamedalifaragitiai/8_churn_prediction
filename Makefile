# Define the name for the virtual environment
VENV_NAME = churn
PYTHON = $(VENV_NAME)/bin/python

.PHONY: all install featurize find_best_model train run-api format lint test clean

all: featurize find_best_model train

install:
	@echo "--- Creating virtual environment named '$(VENV_NAME)' ---"
	@python3 -m venv $(VENV_NAME)
	@echo "--- Installing dependencies with uv ---"
	@$(VENV_NAME)/bin/pip install -q "uv>=0.1.0"
	@$(VENV_NAME)/bin/uv pip install -q -e ".[dev]"
	@echo "\nInstallation complete. Activate with: source $(VENV_NAME)/bin/activate"

featurize:
	@echo "\n--- (1/3) Running Feature Engineering ---"
	@$(PYTHON) scripts/featurize.py

find_best_model:
	@echo "\n--- (2/3) Finding Best Model with AutoML (AutoGluon) ---"
	@$(PYTHON) scripts/find_best_model.py

train:
	@echo "\n--- (3/3) Training Final Model ---"
	@$(PYTHON) scripts/train.py
	@echo "--- Final Model Training Complete ---"

run-api:
	@echo "\n--- Building and Running API via Docker ---"
	@docker build -t churn-predictor-api .
	@docker run -p 8000:8000 -v $(shell pwd)/ml_artifacts:/app/ml_artifacts churn-predictor-api

format:
	@echo "Formatting code..."
	@$(VENV_NAME)/bin/ruff format . && $(VENV_NAME)/bin/black .

lint:
	@echo "Linting code..."
	@$(VENV_NAME)/bin/ruff check .

test:
	@echo "Running tests..."
	@$(VENV_NAME)/bin/pytest

clean:
	@echo "--- Cleaning up project ---"
	@rm -rf $(VENV_NAME) __pycache__ .pytest_cache .ruff_cache mlruns
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@echo "Cleanup complete."
