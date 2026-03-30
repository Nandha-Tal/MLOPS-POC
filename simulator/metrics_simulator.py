"""
K8s Metrics Simulator
=====================
Generates realistic 3-state synthetic Kubernetes cluster metrics.

States:
  - Normal   (80%): healthy cluster baseline
  - Warning  (12%): degraded performance
  - Critical  (8%): severe issues
"""
import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "cpu_usage_pct", "memory_usage_pct", "pod_restart_count",
    "request_latency_p99_ms", "error_rate_pct", "network_bytes_in",
    "network_bytes_out", "disk_io_read_mbps", "disk_io_write_mbps",
    "pod_pending_count", "node_not_ready_count",
]


class K8sMetricsSimulator:
    def __init__(self, n_nodes: int = 3, n_pods: int = 20, random_state: Optional[int] = None):
        self.n_nodes = n_nodes
        self.n_pods = n_pods
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _normal_sample(self) -> dict:
        return {
            "cpu_usage_pct":          np.clip(np.random.normal(40, 12), 5, 75),
            "memory_usage_pct":       np.clip(np.random.normal(55, 10), 20, 80),
            "pod_restart_count":      max(0, int(np.random.poisson(0.3))),
            "request_latency_p99_ms": max(10, np.random.lognormal(4.8, 0.3)),
            "error_rate_pct":         max(0, np.random.exponential(0.3)),
            "network_bytes_in":       abs(np.random.normal(50_000, 15_000)),
            "network_bytes_out":      abs(np.random.normal(30_000, 10_000)),
            "disk_io_read_mbps":      abs(np.random.normal(10, 4)),
            "disk_io_write_mbps":     abs(np.random.normal(7, 3)),
            "pod_pending_count":      max(0, int(np.random.poisson(0.1))),
            "node_not_ready_count":   0,
        }

    def _warning_sample(self) -> dict:
        base = self._normal_sample()
        scenario = random.choice(["cpu_spike", "memory_pressure", "latency_spike", "error_spike"])
        if scenario == "cpu_spike":
            base["cpu_usage_pct"] = np.clip(np.random.normal(82, 5), 75, 92)
        elif scenario == "memory_pressure":
            base["memory_usage_pct"] = np.clip(np.random.normal(87, 4), 82, 94)
            base["pod_pending_count"] = int(np.random.poisson(3))
        elif scenario == "latency_spike":
            base["request_latency_p99_ms"] = np.random.uniform(1000, 1900)
            base["error_rate_pct"] = np.random.uniform(2, 7)
        elif scenario == "error_spike":
            base["error_rate_pct"] = np.random.uniform(5, 12)
            base["pod_restart_count"] = int(np.random.poisson(4))
        return base

    def _critical_sample(self) -> dict:
        base = self._normal_sample()
        scenario = random.choice(["node_failure", "oom_kill", "network_saturation", "full_outage"])
        if scenario == "node_failure":
            base["node_not_ready_count"] = random.randint(1, max(1, self.n_nodes - 1))
            base["pod_pending_count"] = random.randint(10, 30)
            base["cpu_usage_pct"] = np.random.uniform(88, 99)
        elif scenario == "oom_kill":
            base["memory_usage_pct"] = np.random.uniform(94, 100)
            base["pod_restart_count"] = random.randint(10, 30)
            base["pod_pending_count"] = random.randint(5, 20)
        elif scenario == "network_saturation":
            base["network_bytes_in"]  = np.random.uniform(800_000, 2_000_000)
            base["network_bytes_out"] = np.random.uniform(600_000, 1_500_000)
            base["request_latency_p99_ms"] = np.random.uniform(3000, 9000)
            base["error_rate_pct"] = np.random.uniform(20, 60)
        elif scenario == "full_outage":
            base["cpu_usage_pct"] = np.random.uniform(92, 100)
            base["memory_usage_pct"] = np.random.uniform(93, 100)
            base["error_rate_pct"] = np.random.uniform(30, 80)
            base["request_latency_p99_ms"] = np.random.uniform(5000, 15000)
            base["pod_restart_count"] = random.randint(15, 50)
            base["node_not_ready_count"] = random.randint(1, self.n_nodes)
        return base

    def generate(self, n_samples: int = 2000) -> pd.DataFrame:
        rows = []
        base_time = datetime.now() - timedelta(hours=n_samples // 60)

        for i in range(n_samples):
            r = random.random()
            if r < 0.80:
                sample = self._normal_sample()
                is_anomaly = 0
            elif r < 0.92:
                sample = self._warning_sample()
                is_anomaly = 1
            else:
                sample = self._critical_sample()
                is_anomaly = 1

            sample["timestamp"] = (base_time + timedelta(minutes=i)).isoformat()
            sample["is_anomaly"] = is_anomaly
            rows.append(sample)

        df = pd.DataFrame(rows)
        for col in FEATURE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df

    def generate_realtime_sample(self, force_anomaly: bool = False) -> dict:
        if force_anomaly:
            scenario = random.choice(["warning", "critical"])
            sample = self._warning_sample() if scenario == "warning" else self._critical_sample()
        else:
            r = random.random()
            if r < 0.80:
                sample = self._normal_sample()
            elif r < 0.92:
                sample = self._warning_sample()
            else:
                sample = self._critical_sample()

        sample["timestamp"] = datetime.now().isoformat()
        return sample
