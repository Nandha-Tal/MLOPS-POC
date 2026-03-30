"""Unit tests for K8sMetricsSimulator."""
import pytest
import pandas as pd
from simulator.metrics_simulator import K8sMetricsSimulator, FEATURE_COLUMNS


@pytest.fixture
def sim():
    return K8sMetricsSimulator(random_state=42)


def test_generate_returns_dataframe(sim):
    df = sim.generate(n_samples=100)
    assert isinstance(df, pd.DataFrame)


def test_generate_has_all_feature_columns(sim):
    df = sim.generate(n_samples=100)
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_generate_has_label_column(sim):
    df = sim.generate(n_samples=100)
    assert "is_anomaly" in df.columns


def test_anomaly_rate_roughly_20_percent(sim):
    df = sim.generate(n_samples=1000)
    rate = df["is_anomaly"].mean()
    assert 0.10 <= rate <= 0.35, f"Anomaly rate out of expected range: {rate:.2%}"


def test_normal_samples_within_range(sim):
    df = sim.generate(n_samples=500)
    normal = df[df["is_anomaly"] == 0]
    assert (normal["cpu_usage_pct"] <= 100).all()
    assert (normal["cpu_usage_pct"] >= 0).all()
    assert (normal["memory_usage_pct"] <= 100).all()


def test_critical_samples_have_high_values(sim):
    df = sim.generate(n_samples=2000)
    anomalies = df[df["is_anomaly"] == 1]
    assert len(anomalies) > 0
    normal = df[df["is_anomaly"] == 0]
    assert anomalies["error_rate_pct"].mean() >= normal["error_rate_pct"].mean()


def test_realtime_sample_returns_dict(sim):
    sample = sim.generate_realtime_sample()
    assert isinstance(sample, dict)
    assert "timestamp" in sample
    for col in FEATURE_COLUMNS:
        assert col in sample, f"Missing: {col}"


def test_force_anomaly_realtime(sim):
    samples = [sim.generate_realtime_sample(force_anomaly=True) for _ in range(20)]
    max_cpu = max(s["cpu_usage_pct"] for s in samples)
    max_err = max(s["error_rate_pct"] for s in samples)
    assert max_cpu > 60 or max_err > 1, "Forced anomalies should show elevated metrics"


def test_reproducible_with_seed():
    sim1 = K8sMetricsSimulator(random_state=99)
    sim2 = K8sMetricsSimulator(random_state=99)
    df1 = sim1.generate(n_samples=50)
    df2 = sim2.generate(n_samples=50)
    assert df1["cpu_usage_pct"].tolist() == df2["cpu_usage_pct"].tolist()
