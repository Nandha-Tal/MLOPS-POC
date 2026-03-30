"""Metrics endpoints — cluster metrics collection and history."""
from collections import deque
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.schemas.k8s_metrics import K8sMetricsInput
from mlops_poc.logging import logger

router = APIRouter(prefix="/metrics", tags=["Cluster Metrics"])

_metrics_history: deque = deque(maxlen=500)


@router.get("/current", summary="Get current cluster metrics")
async def get_current_metrics():
    try:
        from simulator.metrics_simulator import K8sMetricsSimulator
        sim = K8sMetricsSimulator()
        sample = sim.generate_realtime_sample()
        _metrics_history.append(sample)
        return {"source": "simulator", "metrics": sample, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", summary="Metrics history")
async def get_metrics_history(limit: int = 100):
    history = list(_metrics_history)
    history.reverse()
    return {"count": len(history), "metrics": history[:limit]}


@router.post("/ingest", summary="Ingest metrics snapshot")
async def ingest_metrics(metrics: K8sMetricsInput):
    data = metrics.to_dict()
    data["timestamp"] = data.get("timestamp") or datetime.now().isoformat()
    _metrics_history.append(data)
    logger.info("Metrics ingested: cpu=%.1f%% mem=%.1f%%",
                data["cpu_usage_pct"], data["memory_usage_pct"])
    return {"status": "ingested", "timestamp": data["timestamp"]}


@router.get("/summary", summary="Aggregated metrics summary")
async def get_metrics_summary():
    history = list(_metrics_history)
    if not history:
        from simulator.metrics_simulator import K8sMetricsSimulator
        sample = K8sMetricsSimulator().generate_realtime_sample()
        return {
            "sample_count": 0,
            "latest": sample,
            "averages": {k: round(float(v), 2) for k, v in sample.items()
                         if isinstance(v, (int, float))},
        }
    import pandas as pd
    df = pd.DataFrame(history)
    numeric_cols = df.select_dtypes(include="number").columns
    return {
        "sample_count": len(history),
        "latest": history[-1],
        "averages": {col: round(float(df[col].mean()), 2) for col in numeric_cols},
    }
