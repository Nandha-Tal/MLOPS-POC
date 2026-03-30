"""Health check endpoints for Kubernetes liveness/readiness probes."""
import platform
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["Health"])

_start_time = datetime.now()


@router.get("/live", summary="Liveness probe")
async def liveness():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@router.get("/ready", summary="Readiness probe")
async def readiness():
    model_ready = Path("artifacts/model_trainer/anomaly_detector.joblib").exists()
    if model_ready:
        return {"status": "ready", "model": "loaded", "timestamp": datetime.now().isoformat()}
    return {"status": "not_ready", "model": "missing — run: python main.py"}, 503


@router.get("/info", summary="Application info")
async def info():
    uptime = (datetime.now() - _start_time).total_seconds()
    return {
        "app_name": "K8s Cluster Monitoring API",
        "version":  "1.0.0",
        "python":   platform.python_version(),
        "uptime_seconds": round(uptime, 1),
        "model_path": "artifacts/model_trainer/anomaly_detector.joblib",
        "model_exists": Path("artifacts/model_trainer/anomaly_detector.joblib").exists(),
    }
