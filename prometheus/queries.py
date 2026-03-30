"""
Prometheus Client + PromQL queries for K8s metrics.
"""
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from mlops_poc.logging import logger

PROMQL = {
    "cpu_usage_pct":
        "100 - (avg(irate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)",
    "memory_usage_pct":
        "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
    "pod_restart_count":
        "sum(kube_pod_container_status_restarts_total)",
    "request_latency_p99_ms":
        "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) * 1000",
    "error_rate_pct":
        "sum(rate(http_requests_total{status=~'5..'}[5m])) / sum(rate(http_requests_total[5m])) * 100",
    "network_bytes_in":
        "sum(irate(node_network_receive_bytes_total[5m]))",
    "network_bytes_out":
        "sum(irate(node_network_transmit_bytes_total[5m]))",
    "disk_io_read_mbps":
        "sum(irate(node_disk_read_bytes_total[5m])) / 1048576",
    "disk_io_write_mbps":
        "sum(irate(node_disk_written_bytes_total[5m])) / 1048576",
    "pod_pending_count":
        "count(kube_pod_status_phase{phase='Pending'}) or vector(0)",
    "node_not_ready_count":
        "count(kube_node_status_condition{condition='Ready',status='true'} == 0) or vector(0)",
}


class PrometheusClient:
    def __init__(self, base_url: str = "http://localhost:9090", timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _query_instant(self, query: str) -> Optional[float]:
        try:
            r = requests.get(
                f"{self.base_url}/api/v1/query",
                params={"query": query},
                timeout=self.timeout,
            )
            data = r.json()
            if data["status"] == "success" and data["data"]["result"]:
                return float(data["data"]["result"][0]["value"][1])
        except Exception as e:
            logger.warning("Prometheus query failed: %s", e)
        return None

    def fetch_current_metrics(self) -> dict:
        metrics = {}
        for name, query in PROMQL.items():
            val = self._query_instant(query)
            metrics[name] = val if val is not None else 0.0
        metrics["timestamp"] = datetime.now().isoformat()
        return metrics

    def fetch_training_data(self, lookback_hours: int = 24, step: str = "60s") -> pd.DataFrame:
        end = int(time.time())
        start = end - lookback_hours * 3600
        rows = []

        for ts in range(start, end, 60):
            row = {"timestamp": datetime.fromtimestamp(ts).isoformat()}
            for name, query in PROMQL.items():
                try:
                    r = requests.get(
                        f"{self.base_url}/api/v1/query",
                        params={"query": query, "time": ts},
                        timeout=self.timeout,
                    )
                    data = r.json()
                    if data["status"] == "success" and data["data"]["result"]:
                        row[name] = float(data["data"]["result"][0]["value"][1])
                    else:
                        row[name] = 0.0
                except Exception:
                    row[name] = 0.0
            rows.append(row)

        return pd.DataFrame(rows)

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/-/healthy", timeout=5)
            return r.status_code == 200
        except Exception:
            return False
