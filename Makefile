# Define the name for the virtual environment
VENV_NAME = churn
# Define the path to the Python interpreter within the virtual environment
PYTHON = $(VENV_NAME)/bin/python

# Phony targets prevent conflicts with files of the same name
.PHONY: all install featurize find_best_model train run-api format lint test clean

# Default target: runs the main sequence for the ML pipeline
all: featurize find_best_model train

# Setup virtual environment and install dependencies
install:
	@echo "--- Creating virtual environment named '$(VENV_NAME)' ---"
	@python3 -m venv $(VENV_NAME)
	@echo "--- Installing dependencies with uv ---"
	@$(VENV_NAME)/bin/pip install -q "uv>=0.1.0"
	@# The '-e' flag installs the project in "editable" mode
	@$(VENV_NAME)/bin/uv pip install -q -e ".[dev]"
	@echo "\nInstallation complete."
	@echo "Activate the virtual environment with the command:"
	@echo "source $(VENV_NAME)/bin/activate"

# Step 1: Run feature engineering to process raw data
featurize:
	@echo "\n--- (1/3) Running Feature Engineering ---"
	@$(PYTHON) scripts/featurize.py

# Step 2: Run AutoML to find the best model candidate
find_best_model:
	@echo "\n--- (2/3) Finding Best Model with AutoML (AutoGluon) ---"
	@$(PYTHON) scripts/find_best_model.py

# Step 3: Run the final training pipeline for the selected model
train:
	@echo "\n--- (3/3) Training Final Model ---"
	@$(PYTHON) scripts/train.py
	@echo "--- Final Model Training Complete ---"

# Build and run the FastAPI application with Docker
run-api:
	@echo "\n--- Building and Running API via Docker ---"
	@# Temporarily disable BuildKit with DOCKER_BUILDKIT=0 to solve the buildx error
	@DOCKER_BUILDKIT=0 docker build -t churn-predictor-api .
	@echo "\n--- Starting API Container (Press CTRL+C to stop) ---"
	@# The --rm flag automatically removes the container when it exits
	@docker run --rm -p 8000:8000 -v "$(shell pwd)/ml_artifacts:/app/ml_artifacts" churn-predictor-api

# --- Utility Commands ---

# Format and lint the code using the tools installed in the venv
format:
	@echo "Formatting code..."
	@$(VENV_NAME)/bin/ruff format .
	@$(VENV_NAME)/bin/black .

lint:
	@echo "Linting code..."
	@$(VENV_NAME)/bin/ruff check .

# Run tests using pytest from the venv
test:
	@echo "Running tests..."
	@$(VENV_NAME)/bin/pytest

# Clean up the virtual environment and other generated files
clean:
	@echo "--- Cleaning up project ---"
	@rm -rf $(VENV_NAME) __pycache__ .pytest_cache .ruff_cache mlruns
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@echo "Cleanup complete."
