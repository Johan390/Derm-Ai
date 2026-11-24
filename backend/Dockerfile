# DermAI Backend Production Dockerfile
FROM python:3.11-slim

# System dependencies for OpenCV & image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source & ML model artifacts
COPY backend /app/backend
COPY ml_models /app/ml_models
COPY ml_dataset/dataset_info.json /app/ml_dataset/dataset_info.json
COPY ml_dataset/eda_summary.json /app/ml_dataset/eda_summary.json

WORKDIR /app/backend

ENV PYTHONPATH="/app:/app/backend"

# Create uploads storage directories
RUN mkdir -p uploads/original uploads/preview uploads/preprocessed

# Non-root service user for container hardening
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Production WSGI/ASGI execution with Uvicorn worker pool
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

