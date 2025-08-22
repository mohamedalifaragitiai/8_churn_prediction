# Stage 1: Build stage with dependencies
FROM python:3.10-slim as builder

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies using uv
RUN uv pip install --system --no-cache -e ".[dev]"

# ---

# Stage 2: Final production stage
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application source code
COPY api /app/api
COPY src /app/src
COPY configs /app/configs

# Expose the port the API will run on
EXPOSE 8000

# Set the command to run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
