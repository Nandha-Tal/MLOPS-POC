#!/bin/bash
# Setup script for K8s Monitoring MLOps POC
set -e

echo "=============================="
echo " MLOps K8s Monitoring POC Setup"
echo "=============================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "[1/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[2/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "[3/4] Creating directories..."
mkdir -p artifacts/data_ingestion artifacts/prepare_base_model \
         artifacts/model_trainer artifacts/model_evaluation \
         logs mlruns

echo "[4/4] Verifying installation..."
python3 -c "import mlflow, sklearn, fastapi, anthropic; print('  All packages OK ✓')"

echo ""
echo "Setup complete! Next steps:"
echo "  python3 main.py                              # Train the model"
echo "  uvicorn app.main:app --port 8080 --reload   # Start API"
echo "  open http://localhost:8080                   # Dashboard"
