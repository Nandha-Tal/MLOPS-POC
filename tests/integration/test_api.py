"""Integration tests for the K8s Monitoring FastAPI application."""
import pytest
from fastapi.testclient import TestClient

FEATURE_COLS = [
    "cpu_usage_pct", "memory_usage_pct", "pod_restart_count",
    "request_latency_p99_ms", "error_rate_pct", "network_bytes_in",
    "network_bytes_out", "disk_io_read_mbps", "disk_io_write_mbps",
    "pod_pending_count", "node_not_ready_count",
]

NORMAL_PAYLOAD = {
    "cpu_usage_pct": 35.0, "memory_usage_pct": 55.0, "pod_restart_count": 0,
    "request_latency_p99_ms": 120.0, "error_rate_pct": 0.3,
    "network_bytes_in": 40000.0, "network_bytes_out": 25000.0,
    "disk_io_read_mbps": 8.0, "disk_io_write_mbps": 5.0,
    "pod_pending_count": 0, "node_not_ready_count": 0,
}

CRITICAL_PAYLOAD = {
    "cpu_usage_pct": 96.0, "memory_usage_pct": 98.0, "pod_restart_count": 25,
    "request_latency_p99_ms": 9000.0, "error_rate_pct": 45.0,
    "network_bytes_in": 1_500_000.0, "network_bytes_out": 1_200_000.0,
    "disk_io_read_mbps": 120.0, "disk_io_write_mbps": 100.0,
    "pod_pending_count": 20, "node_not_ready_count": 3,
}


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    import os
    import joblib
    from pathlib import Path
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from simulator.metrics_simulator import K8sMetricsSimulator

    tmp = tmp_path_factory.mktemp("artifacts")
    model_dir = tmp / "model_trainer"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "anomaly_detector.joblib"

    sim = K8sMetricsSimulator(random_state=42)
    df = sim.generate(n_samples=300)
    normal = df[df["is_anomaly"] == 0][FEATURE_COLS].fillna(0.0)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(n_estimators=50, contamination=0.05, random_state=42)),
    ])
    pipe.fit(normal)
    joblib.dump(pipe, model_path)
    os.environ["MODEL_PATH"] = str(model_path)
    return model_path


@pytest.fixture(scope="module")
def client(trained_model):
    from mlops_poc.pipeline import prediction_pipeline as pp_module
    original_init = pp_module.PredictionPipeline.__init__

    def patched_init(self):
        import joblib
        self._model = joblib.load(trained_model)

    pp_module.PredictionPipeline.__init__ = patched_init

    import app.routes.alerts as alerts_mod
    alerts_mod._pipeline = None

    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    pp_module.PredictionPipeline.__init__ = original_init


class TestHealthEndpoints:
    def test_liveness(self, client):
        r = client.get("/health/live")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

    def test_readiness(self, client):
        r = client.get("/health/ready")
        assert r.status_code in (200, 503)

    def test_info(self, client):
        r = client.get("/health/info")
        assert r.status_code == 200


class TestAlertCheck:
    def test_check_returns_200(self, client):
        assert client.post("/alerts/check", json=NORMAL_PAYLOAD).status_code == 200

    def test_check_response_schema(self, client):
        body = client.post("/alerts/check", json=NORMAL_PAYLOAD).json()
        for key in ("anomaly_score", "is_anomaly", "severity", "triggered_rules", "timestamp"):
            assert key in body

    def test_severity_values(self, client):
        assert client.post("/alerts/check", json=NORMAL_PAYLOAD).json()["severity"] in ("OK", "WARNING", "CRITICAL")

    def test_critical_payload_flagged(self, client):
        body = client.post("/alerts/check", json=CRITICAL_PAYLOAD).json()
        assert body["is_anomaly"] or body["severity"] in ("WARNING", "CRITICAL")

    def test_invalid_cpu_rejected(self, client):
        assert client.post("/alerts/check", json={**NORMAL_PAYLOAD, "cpu_usage_pct": 150.0}).status_code == 422

    def test_missing_field_rejected(self, client):
        bad = {k: v for k, v in NORMAL_PAYLOAD.items() if k != "memory_usage_pct"}
        assert client.post("/alerts/check", json=bad).status_code == 422


class TestAlertBatch:
    def test_batch_check(self, client):
        r = client.post("/alerts/check/batch", json=[NORMAL_PAYLOAD, CRITICAL_PAYLOAD])
        assert r.status_code == 200
        assert r.json()["total"] == 2

    def test_empty_batch_rejected(self, client):
        assert client.post("/alerts/check/batch", json=[]).status_code == 400


class TestAlertSimulate:
    def test_simulate_returns_200(self, client):
        assert client.post("/alerts/simulate", json={}).status_code == 200

    def test_simulate_force_anomaly(self, client):
        r = client.post("/alerts/simulate", json={"force_anomaly": True})
        assert r.status_code == 200
        assert "anomaly_score" in r.json()


class TestAlertHistory:
    def test_history_returns_200(self, client):
        assert client.get("/alerts/history").status_code == 200

    def test_clear_history(self, client):
        client.post("/alerts/check", json=NORMAL_PAYLOAD)
        assert client.delete("/alerts/history").status_code == 200
        assert client.get("/alerts/history").json()["total_in_buffer"] == 0


class TestMetricsEndpoints:
    def test_summary(self, client):
        assert client.get("/metrics/summary").status_code == 200

    def test_ingest(self, client):
        assert client.post("/metrics/ingest", json=NORMAL_PAYLOAD).status_code in (200, 201, 202)


class TestPipelineEndpoints:
    def test_status(self, client):
        assert client.get("/pipeline/status").status_code == 200

    def test_metrics(self, client):
        assert client.get("/pipeline/metrics").status_code == 200
