"""
Kubeflow Pipelines v2 — K8s Anomaly Detection Pipeline
=======================================================
3-step KFP DAG: collect metrics → train → evaluate & log

Compile:
    python kubeflow/pipeline.py
    # Outputs: kubeflow/k8s_monitoring_pipeline.yaml

Submit to Kubeflow:
    pip install kfp
    python -c "
    import kfp
    client = kfp.Client(host='http://kubeflow-pipelines:8888')
    client.create_run_from_pipeline_package('kubeflow/k8s_monitoring_pipeline.yaml')
    "
"""
from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn"],
)
def collect_metrics_step(
    n_samples: int,
    use_simulator: bool,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    labelled_data: Output[Dataset],
):
    """Stage 1 — Collect K8s metrics from Prometheus or simulator."""
    import random
    import numpy as np
    import pandas as pd

    FEATURE_COLUMNS = [
        "cpu_usage_pct", "memory_usage_pct", "pod_restart_count",
        "request_latency_p99_ms", "error_rate_pct", "network_bytes_in",
        "network_bytes_out", "disk_io_read_mbps", "disk_io_write_mbps",
        "pod_pending_count", "node_not_ready_count",
    ]

    rows = []
    for _ in range(n_samples):
        r = random.random()
        if r < 0.80:
            row = {
                "cpu_usage_pct":          np.clip(np.random.normal(40, 12), 5, 75),
                "memory_usage_pct":       np.clip(np.random.normal(55, 10), 20, 80),
                "pod_restart_count":      max(0, int(np.random.poisson(0.3))),
                "request_latency_p99_ms": max(10, np.random.lognormal(4.8, 0.3)),
                "error_rate_pct":         max(0, np.random.exponential(0.3)),
                "network_bytes_in":       abs(np.random.normal(50000, 15000)),
                "network_bytes_out":      abs(np.random.normal(30000, 10000)),
                "disk_io_read_mbps":      abs(np.random.normal(10, 4)),
                "disk_io_write_mbps":     abs(np.random.normal(7, 3)),
                "pod_pending_count":      max(0, int(np.random.poisson(0.1))),
                "node_not_ready_count":   0,
                "is_anomaly":             0,
            }
        else:
            row = {
                "cpu_usage_pct":          np.random.uniform(85, 99),
                "memory_usage_pct":       np.random.uniform(88, 100),
                "pod_restart_count":      random.randint(8, 30),
                "request_latency_p99_ms": np.random.uniform(2000, 9000),
                "error_rate_pct":         np.random.uniform(10, 50),
                "network_bytes_in":       np.random.uniform(500000, 2000000),
                "network_bytes_out":      np.random.uniform(400000, 1500000),
                "disk_io_read_mbps":      np.random.uniform(50, 150),
                "disk_io_write_mbps":     np.random.uniform(40, 120),
                "pod_pending_count":      random.randint(5, 25),
                "node_not_ready_count":   random.randint(1, 3),
                "is_anomaly":             1,
            }
        rows.append(row)

    df = pd.DataFrame(rows)
    normal = df[df["is_anomaly"] == 0][FEATURE_COLUMNS]
    normal.to_csv(train_data.path, index=False)
    df[FEATURE_COLUMNS + ["is_anomaly"]].to_csv(test_data.path, index=False)
    df.to_csv(labelled_data.path, index=False)
    print(f"Generated {n_samples} samples: {normal.shape[0]} normal, {df['is_anomaly'].sum()} anomalies")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib"],
)
def train_anomaly_detector_step(
    train_data: Input[Dataset],
    trained_model: Output[Model],
    n_estimators: int = 100,
    contamination: float = 0.05,
):
    """Stage 3 — Train IsolationForest on normal-only metrics."""
    import joblib
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(train_data.path)
    X = df.fillna(0.0)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
        )),
    ])
    pipe.fit(X)
    joblib.dump(pipe, trained_model.path)

    preds = pipe.predict(X)
    print(f"Trained on {len(X)} samples. Train anomaly rate: {(preds == -1).mean():.2%}")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib", "mlflow"],
)
def evaluate_and_log_step(
    trained_model: Input[Model],
    labelled_data: Input[Dataset],
    mlflow_uri: str,
    metrics_output: Output[Dataset],
):
    """Stage 4 — Evaluate model and log metrics to MLflow."""
    import json
    import joblib
    import mlflow
    import mlflow.sklearn
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    FEATURE_COLUMNS = [
        "cpu_usage_pct", "memory_usage_pct", "pod_restart_count",
        "request_latency_p99_ms", "error_rate_pct", "network_bytes_in",
        "network_bytes_out", "disk_io_read_mbps", "disk_io_write_mbps",
        "pod_pending_count", "node_not_ready_count",
    ]

    model = joblib.load(trained_model.path)
    df = pd.read_csv(labelled_data.path)
    X = df[FEATURE_COLUMNS].fillna(0.0)
    y_true = df["is_anomaly"].values

    preds = model.predict(X)
    scores = model.decision_function(X)
    y_pred_binary = (preds == -1).astype(int)
    scores_norm = (-scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    metrics = {
        "precision":    round(float(precision_score(y_true, y_pred_binary, zero_division=0)), 4),
        "recall":       round(float(recall_score(y_true, y_pred_binary, zero_division=0)), 4),
        "f1_score":     round(float(f1_score(y_true, y_pred_binary, zero_division=0)), 4),
        "roc_auc":      round(float(roc_auc_score(y_true, scores_norm)), 4),
        "anomaly_rate": round(float(y_pred_binary.mean()), 4),
    }
    print("Metrics:", metrics)

    mlflow.set_tracking_uri(mlflow_uri)
    with mlflow.start_run(run_name="kubeflow-k8s-anomaly"):
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"pipeline": "kubeflow", "model_type": "IsolationForest"})
        mlflow.sklearn.log_model(model, artifact_path="model")

    with open(metrics_output.path, "w") as f:
        json.dump(metrics, f, indent=2)


@dsl.pipeline(
    name="K8s Anomaly Detection Pipeline",
    description="End-to-end K8s cluster anomaly detection: collect → train → evaluate",
)
def k8s_monitoring_pipeline(
    n_samples:     int   = 2000,
    use_simulator: bool  = True,
    n_estimators:  int   = 100,
    contamination: float = 0.05,
    mlflow_uri:    str   = "sqlite:///mlflow.db",
):
    collect = collect_metrics_step(n_samples=n_samples, use_simulator=use_simulator)
    collect.set_cpu_limit("500m").set_memory_limit("512Mi")

    train = train_anomaly_detector_step(
        train_data=collect.outputs["train_data"],
        n_estimators=n_estimators,
        contamination=contamination,
    )
    train.set_cpu_limit("1").set_memory_limit("1Gi")

    evaluate_and_log_step(
        trained_model=train.outputs["trained_model"],
        labelled_data=collect.outputs["labelled_data"],
        mlflow_uri=mlflow_uri,
    ).set_cpu_limit("500m").set_memory_limit("512Mi")


if __name__ == "__main__":
    from pathlib import Path
    from kfp import compiler

    output = Path("kubeflow/k8s_monitoring_pipeline.yaml")
    output.parent.mkdir(exist_ok=True)
    compiler.Compiler().compile(k8s_monitoring_pipeline, str(output))
    print(f"Pipeline compiled to: {output}")
