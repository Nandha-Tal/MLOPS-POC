"""Tool definitions and executor for the K8s SRE agent."""
import json
from typing import Any

import httpx

API_BASE = "http://localhost:8080"

TOOL_DEFINITIONS = [
    {
        "name": "get_cluster_health",
        "description": "Get overall Kubernetes cluster health status and current metrics.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "detect_anomaly_now",
        "description": "Run real-time anomaly detection using the current cluster metrics.",
        "input_schema": {"type": "object", "properties": {
            "force_anomaly": {"type": "boolean", "description": "Force anomalous sample for testing"}
        }, "required": []},
    },
    {
        "name": "get_alert_history",
        "description": "Get recent alert history from the monitoring system.",
        "input_schema": {"type": "object", "properties": {
            "limit": {"type": "integer", "description": "Number of alerts to return", "default": 20}
        }, "required": []},
    },
    {
        "name": "get_pipeline_status",
        "description": "Get the current ML training pipeline status and artifact availability.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_pipeline_stage",
        "description": "Trigger a specific ML pipeline stage (1=ingest, 2=prepare, 3=train, 4=evaluate).",
        "input_schema": {"type": "object", "properties": {
            "stage": {"type": "integer", "enum": [1, 2, 3, 4]}
        }, "required": ["stage"]},
    },
    {
        "name": "get_model_metrics",
        "description": "Get the latest ML model evaluation metrics (precision, recall, F1, ROC-AUC).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_service_health",
        "description": "Check the health and readiness of the monitoring API service.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_current_metrics",
        "description": "Fetch the latest cluster metrics snapshot.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_k8s_pods",
        "description": "List Kubernetes pods in a namespace with their status.",
        "input_schema": {"type": "object", "properties": {
            "namespace": {"type": "string", "default": "default"}
        }, "required": []},
    },
    {
        "name": "restart_deployment",
        "description": "Restart a Kubernetes deployment (rolling restart).",
        "input_schema": {"type": "object", "properties": {
            "deployment": {"type": "string"},
            "namespace":  {"type": "string", "default": "default"},
        }, "required": ["deployment"]},
    },
    {
        "name": "scale_deployment",
        "description": "Scale a Kubernetes deployment to a specified replica count.",
        "input_schema": {"type": "object", "properties": {
            "deployment": {"type": "string"},
            "replicas":   {"type": "integer", "minimum": 0, "maximum": 20},
            "namespace":  {"type": "string", "default": "default"},
        }, "required": ["deployment", "replicas"]},
    },
]


def execute_tool(name: str, inputs: dict) -> Any:
    with httpx.Client(base_url=API_BASE, timeout=15) as client:
        if name == "get_cluster_health":
            return client.get("/health/info").json()
        elif name == "detect_anomaly_now":
            return client.post("/alerts/simulate", json={"force_anomaly": inputs.get("force_anomaly", False)}).json()
        elif name == "get_alert_history":
            return client.get("/alerts/history", params={"limit": inputs.get("limit", 20)}).json()
        elif name == "get_pipeline_status":
            return client.get("/pipeline/status").json()
        elif name == "run_pipeline_stage":
            return client.post(f"/pipeline/run/{inputs['stage']}").json()
        elif name == "get_model_metrics":
            return client.get("/pipeline/metrics").json()
        elif name == "get_service_health":
            return {"liveness": client.get("/health/live").json(), "readiness": client.get("/health/ready").json()}
        elif name == "get_current_metrics":
            return client.get("/metrics/current").json()
        elif name == "list_k8s_pods":
            return {
                "note": "kubectl integration requires cluster access",
                "namespace": inputs.get("namespace", "default"),
                "simulated_pods": [{"name": f"api-{i}", "status": "Running", "restarts": 0} for i in range(3)],
            }
        elif name == "restart_deployment":
            return {"action": "restart", "deployment": inputs["deployment"],
                    "namespace": inputs.get("namespace", "default"), "status": "Rolling restart initiated (simulated)"}
        elif name == "scale_deployment":
            return {"action": "scale", "deployment": inputs["deployment"],
                    "replicas": inputs["replicas"], "status": "Scaled (simulated)"}
        else:
            return {"error": f"Unknown tool: {name}"}
