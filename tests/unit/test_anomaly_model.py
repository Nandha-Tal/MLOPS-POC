"""Unit tests for the IsolationForest anomaly detection pipeline."""
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from simulator.metrics_simulator import K8sMetricsSimulator

FEATURE_COLS = [
    "cpu_usage_pct", "memory_usage_pct", "pod_restart_count",
    "request_latency_p99_ms", "error_rate_pct", "network_bytes_in",
    "network_bytes_out", "disk_io_read_mbps", "disk_io_write_mbps",
    "pod_pending_count", "node_not_ready_count",
]


@pytest.fixture(scope="module")
def trained_pipeline():
    sim = K8sMetricsSimulator(random_state=42)
    df = sim.generate(n_samples=500)
    normal = df[df["is_anomaly"] == 0][FEATURE_COLS].fillna(0.0)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(n_estimators=50, contamination=0.05, random_state=42)),
    ])
    pipe.fit(normal)
    return pipe


def test_pipeline_predicts_normal(trained_pipeline):
    sim = K8sMetricsSimulator(random_state=1)
    df = sim.generate(n_samples=200)
    normal = df[df["is_anomaly"] == 0][FEATURE_COLS].fillna(0.0)
    preds = trained_pipeline.predict(normal)
    normal_rate = (preds == 1).mean()
    assert normal_rate >= 0.80, f"Normal prediction rate too low: {normal_rate:.2%}"


def test_pipeline_detects_anomalies(trained_pipeline):
    sim = K8sMetricsSimulator(random_state=2)
    df = sim.generate(n_samples=500)
    anomalies = df[df["is_anomaly"] == 1][FEATURE_COLS].fillna(0.0)
    if len(anomalies) == 0:
        pytest.skip("No anomaly samples generated")
    preds = trained_pipeline.predict(anomalies)
    detection_rate = (preds == -1).mean()
    assert detection_rate > 0.0, "Model should detect at least some anomalies"


def test_decision_function_returns_float(trained_pipeline):
    sim = K8sMetricsSimulator(random_state=3)
    sample = sim.generate(n_samples=1)[FEATURE_COLS].fillna(0.0)
    score = trained_pipeline.decision_function(sample)
    assert score.dtype in (np.float32, np.float64)
    assert len(score) == 1


def test_score_range(trained_pipeline):
    sim = K8sMetricsSimulator(random_state=4)
    df = sim.generate(n_samples=200)[FEATURE_COLS].fillna(0.0)
    scores = trained_pipeline.decision_function(df)
    assert scores.min() >= -1.0
    assert scores.max() <= 1.0
