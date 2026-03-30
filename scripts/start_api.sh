#!/bin/bash
# Start the K8s Monitoring FastAPI server
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [ ! -f "artifacts/model_trainer/anomaly_detector.joblib" ]; then
  echo "Model not found. Training first..."
  python3 main.py
fi

echo "Starting K8s Monitoring API..."
echo "  Dashboard : http://localhost:8080/"
echo "  Swagger   : http://localhost:8080/docs"
echo "  Health    : http://localhost:8080/health/live"
echo ""

python3 -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8080 \
  --reload \
  --log-level info
