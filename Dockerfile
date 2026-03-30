# Multi-stage Dockerfile — K8s Monitoring API
# Stage 1: builder
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: runtime
FROM python:3.11-slim AS runtime
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy source
COPY src/       ./src/
COPY app/       ./app/
COPY simulator/ ./simulator/
COPY prometheus/ ./prometheus/
COPY templates/ ./templates/
COPY config/    ./config/
COPY main.py    .
COPY setup.py   .

# Install package
RUN pip install --no-cache-dir -e .

# Non-root user
RUN useradd -m -u 1000 mlops && chown -R mlops:mlops /app
USER mlops

# Artifacts directory
RUN mkdir -p /app/artifacts /app/logs

ENV PYTHONPATH=/app/src
ENV APP_ENV=production

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python3 -c "import httpx; httpx.get('http://localhost:8080/health/live').raise_for_status()"

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
