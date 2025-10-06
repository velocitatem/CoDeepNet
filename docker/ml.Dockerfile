FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY ml/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML code
COPY ml/ ./
COPY src/ ./src/

# Create models directory
RUN mkdir -p /app/models/weights

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

EXPOSE 8000

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]