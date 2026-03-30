"""Pydantic schemas for K8s monitoring API requests and responses."""
from typing import List, Optional
from pydantic import BaseModel, Field


class K8sMetricsInput(BaseModel):
    timestamp:              Optional[str]   = Field(None, description="ISO timestamp")
    cpu_usage_pct:          float = Field(..., ge=0, le=100, description="CPU utilization %")
    memory_usage_pct:       float = Field(..., ge=0, le=100, description="Memory utilization %")
    pod_restart_count:      int   = Field(..., ge=0, description="Pod restarts in window")
    request_latency_p99_ms: float = Field(..., ge=0, description="P99 latency (ms)")
    error_rate_pct:         float = Field(..., ge=0, le=100, description="HTTP 5xx error rate %")
    network_bytes_in:       float = Field(..., ge=0, description="Network inbound bytes/s")
    network_bytes_out:      float = Field(..., ge=0, description="Network outbound bytes/s")
    disk_io_read_mbps:      float = Field(..., ge=0, description="Disk read MB/s")
    disk_io_write_mbps:     float = Field(..., ge=0, description="Disk write MB/s")
    pod_pending_count:      int   = Field(..., ge=0, description="Pods in Pending state")
    node_not_ready_count:   int   = Field(..., ge=0, description="NotReady nodes")

    def to_dict(self) -> dict:
        return self.model_dump()

    model_config = {"json_schema_extra": {"example": {
        "cpu_usage_pct": 45.2, "memory_usage_pct": 62.1,
        "pod_restart_count": 0, "request_latency_p99_ms": 145.0,
        "error_rate_pct": 0.5, "network_bytes_in": 52000.0,
        "network_bytes_out": 31000.0, "disk_io_read_mbps": 9.5,
        "disk_io_write_mbps": 7.2, "pod_pending_count": 0,
        "node_not_ready_count": 0,
    }}}


class AnomalyResponse(BaseModel):
    timestamp:       str
    anomaly_score:   float = Field(..., description="Score in [-1, 0]; more negative = more anomalous")
    is_anomaly:      bool
    severity:        str   = Field(..., description="OK | WARNING | CRITICAL")
    triggered_rules: List[str] = Field(default_factory=list)
    metrics:         dict  = Field(default_factory=dict)


class BatchAnomalyResponse(BaseModel):
    predictions:    List[AnomalyResponse]
    total:          int
    anomaly_count:  int
    critical_count: int


class AlertSummary(BaseModel):
    total_anomalies: int
    critical_count:  int
    warning_count:   int
    latest_severity: str
    top_rules:       List[str]


class SimulatorRequest(BaseModel):
    force_anomaly: bool = Field(False, description="Force an anomalous sample")
    n_nodes:       int  = Field(3, ge=1, le=20)
    n_pods:        int  = Field(20, ge=1, le=200)
