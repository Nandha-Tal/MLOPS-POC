"""Alert Routes — ML inference endpoints for K8s anomaly detection."""
from collections import deque
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException

from app.schemas.k8s_metrics import AnomalyResponse, BatchAnomalyResponse, SimulatorRequest, K8sMetricsInput
from mlops_poc.logging import logger
from mlops_poc.pipeline.prediction_pipeline import PredictionPipeline

router = APIRouter(prefix="/alerts", tags=["Alerts & Anomaly Detection"])

_pipeline: PredictionPipeline | None = None
_alert_history: deque = deque(maxlen=200)


def get_pipeline() -> PredictionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PredictionPipeline()
    return _pipeline


@router.post("/check", response_model=AnomalyResponse, summary="Check metrics for anomaly")
async def check_anomaly(metrics: K8sMetricsInput) -> AnomalyResponse:
    try:
        result = get_pipeline().detect(metrics.to_dict())
        response = AnomalyResponse(
            timestamp=result.timestamp or datetime.now().isoformat(),
            anomaly_score=result.anomaly_score,
            is_anomaly=result.is_anomaly,
            severity=result.severity,
            triggered_rules=result.triggered_rules,
            metrics=result.metrics,
        )
        _alert_history.append(response.model_dump())
        if result.is_anomaly:
            logger.warning("Alert: [%s] score=%.4f rules=%s", result.severity, result.anomaly_score, result.triggered_rules)
        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check/batch", response_model=BatchAnomalyResponse, summary="Batch anomaly detection")
async def check_anomaly_batch(metrics_list: List[K8sMetricsInput]) -> BatchAnomalyResponse:
    if not metrics_list:
        raise HTTPException(status_code=400, detail="Empty batch")
    try:
        pipeline = get_pipeline()
        predictions = []
        for m in metrics_list:
            result = pipeline.detect(m.to_dict())
            predictions.append(AnomalyResponse(
                timestamp=result.timestamp or datetime.now().isoformat(),
                anomaly_score=result.anomaly_score,
                is_anomaly=result.is_anomaly,
                severity=result.severity,
                triggered_rules=result.triggered_rules,
                metrics=result.metrics,
            ))
        anomaly_count  = sum(1 for p in predictions if p.is_anomaly)
        critical_count = sum(1 for p in predictions if p.severity == "CRITICAL")
        return BatchAnomalyResponse(predictions=predictions, total=len(predictions),
                                    anomaly_count=anomaly_count, critical_count=critical_count)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate", response_model=AnomalyResponse, summary="Simulate metrics and detect anomaly")
async def simulate_and_detect(req: SimulatorRequest) -> AnomalyResponse:
    try:
        from simulator.metrics_simulator import K8sMetricsSimulator
        sim = K8sMetricsSimulator(n_nodes=req.n_nodes, n_pods=req.n_pods)
        raw = sim.generate_realtime_sample(force_anomaly=req.force_anomaly)
        result = get_pipeline().detect(raw)
        return AnomalyResponse(
            timestamp=raw.get("timestamp", datetime.now().isoformat()),
            anomaly_score=result.anomaly_score,
            is_anomaly=result.is_anomaly,
            severity=result.severity,
            triggered_rules=result.triggered_rules,
            metrics=result.metrics,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", summary="Recent alert history")
async def get_alert_history(limit: int = 50):
    history = list(_alert_history)
    history.reverse()
    anomalies = [h for h in history if h["is_anomaly"]]
    return {
        "total_in_buffer": len(history),
        "anomaly_count": len(anomalies),
        "critical_count": sum(1 for h in history if h.get("severity") == "CRITICAL"),
        "recent_alerts": history[:limit],
    }


@router.delete("/history", summary="Clear alert history")
async def clear_alert_history():
    count = len(_alert_history)
    _alert_history.clear()
    return {"cleared": count, "message": "Alert history cleared"}
