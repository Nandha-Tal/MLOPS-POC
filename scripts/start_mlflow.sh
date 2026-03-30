#!/bin/bash
# Start MLflow tracking server (SQLite backend)
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DB_PATH="$PROJECT_DIR/mlflow.db"
ARTIFACT_PATH="$PROJECT_DIR/mlruns"

echo "Starting MLflow server..."
echo "  Backend : sqlite:///$DB_PATH"
echo "  Artifacts: $ARTIFACT_PATH"
echo "  UI URL   : http://localhost:5000"
echo ""

mlflow server \
  --backend-store-uri "sqlite:///$DB_PATH" \
  --default-artifact-root "$ARTIFACT_PATH" \
  --host 0.0.0.0 \
  --port 5000
