"""Pipeline management endpoints — trigger and monitor ML pipeline stages."""
import subprocess
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

router = APIRouter(prefix="/pipeline", tags=["ML Pipeline"])

_pipeline_status: dict = {"stage": None, "status": "idle", "last_run": None}


def _run_stage(stage: int):
    global _pipeline_status
    _pipeline_status = {"stage": stage, "status": "running", "last_run": datetime.now().isoformat()}
    try:
        result = subprocess.run(
            ["python3", "main.py", "--stage", str(stage)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            _pipeline_status["status"] = "completed"
        else:
            _pipeline_status["status"] = "failed"
            _pipeline_status["error"] = result.stderr[-500:]
    except Exception as e:
        _pipeline_status["status"] = "failed"
        _pipeline_status["error"] = str(e)


@router.post("/run/{stage}", summary="Run a pipeline stage")
async def run_pipeline_stage(stage: int, background_tasks: BackgroundTasks):
    if stage not in (1, 2, 3, 4):
        raise HTTPException(status_code=400, detail="Stage must be 1-4")
    if _pipeline_status.get("status") == "running":
        raise HTTPException(status_code=409, detail="Pipeline already running")
    background_tasks.add_task(_run_stage, stage)
    return {"message": f"Stage {stage} started", "stage": stage, "status": "started"}


@router.post("/run/all", summary="Run all pipeline stages")
async def run_all_stages(background_tasks: BackgroundTasks):
    if _pipeline_status.get("status") == "running":
        raise HTTPException(status_code=409, detail="Pipeline already running")

    def _run_all():
        for s in (1, 2, 3, 4):
            _run_stage(s)
            if _pipeline_status.get("status") == "failed":
                break

    background_tasks.add_task(_run_all)
    return {"message": "All stages started", "status": "started"}


@router.get("/status", summary="Pipeline status")
async def get_pipeline_status():
    artifacts = {
        "train_data":   Path("artifacts/data_ingestion/train.csv").exists(),
        "base_model":   Path("artifacts/prepare_base_model/base_model.joblib").exists(),
        "trained_model": Path("artifacts/model_trainer/anomaly_detector.joblib").exists(),
        "metrics":      Path("artifacts/model_evaluation/metrics.json").exists(),
    }
    return {**_pipeline_status, "artifacts": artifacts, "timestamp": datetime.now().isoformat()}


@router.get("/metrics", summary="Latest pipeline evaluation metrics")
async def get_pipeline_metrics():
    metrics_path = Path("artifacts/model_evaluation/metrics.json")
    if not metrics_path.exists():
        return {"message": "No metrics yet — run the pipeline first", "metrics": {}}
    import json
    with open(metrics_path) as f:
        return {"metrics": json.load(f), "path": str(metrics_path)}
