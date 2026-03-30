# MLOps POC — Kubernetes Cluster Monitoring with Agentic AI

This project started from a simple question: *what happens when a Kubernetes cluster starts behaving oddly at 2am, and nobody is watching?*

The answer we built here is a full-stack MLOps pipeline that learns what "normal" looks like in a K8s cluster, flags anything that deviates from it, and hands off investigation to an AI agent that can actually do something about it — restart deployments, scale services, retrain the model, and file an incident report — all without waking anyone up.

It's a POC, so there's no production cluster required. Everything runs locally using a simulator that generates realistic 3-state cluster behaviour.

---

## What Problem This Solves

Traditional K8s monitoring relies on threshold alerts — if CPU goes above 85%, fire an alert. That works for obvious problems, but it completely misses the subtle ones: a slow latency creep combined with a rising error rate and three pending pods, none of which crossed a threshold alone, but together signal a cascading failure that's 20 minutes away from taking down the service.

This project takes a different approach. We train a machine learning model (IsolationForest) on what healthy cluster behaviour looks like across 11 metrics simultaneously. When something deviates from that learned pattern — even if no single metric crossed a threshold — it gets flagged. We layer rule-based threshold checks on top for the obvious cases. Between the two, almost nothing slips through.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Data Sources                                       │
│                                                                           │
│   Prometheus (real cluster)          K8s Metrics Simulator               │
│   node-exporter / cAdvisor /         (when no cluster available)         │
│   kube-state-metrics                 3 states: Normal / Warning / Critical│
└───────────────────────┬──────────────────────────────────────────────────┘
                        │ 11 metrics every 60s
                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     MLOps Training Pipeline                               │
│                                                                           │
│  Stage 1           Stage 2            Stage 3          Stage 4           │
│  ─────────         ──────────         ──────────        ──────────────   │
│  Data              Prepare            Model             Evaluate          │
│  Ingestion    ───► Base Model    ───► Trainer      ───► + MLflow Log     │
│                                                                           │
│  Pull metrics      Build sklearn      Fit on normal     Precision/Recall  │
│  Split train/test  Pipeline           samples only      F1 / ROC-AUC     │
│  Save CSV          StandardScaler     Save .joblib      Log to MLflow    │
│                  + IsolationForest                      Save metrics.json │
└───────────────────────┬──────────────────────────────────────────────────┘
                        │ anomaly_detector.joblib
                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   Detection Engine (Prediction Pipeline)                  │
│                                                                           │
│   Input: 11 K8s metrics snapshot                                         │
│                                                                           │
│   ┌─────────────────────────┐   ┌────────────────────────────────────┐   │
│   │   ML Model (IsolForest) │   │   Rule Engine (Threshold Checks)   │   │
│   │   Score: [-1.0 to 1.0]  │   │   cpu > 90% → cpu_critical         │   │
│   │   < -0.15 → CRITICAL    │   │   latency > 2000ms → latency_crit  │   │
│   │   < -0.05 → WARNING     │   │   node_not_ready >= 1 → critical   │   │
│   └────────────┬────────────┘   └──────────────┬─────────────────────┘   │
│                └──────────────┬─────────────────┘                        │
│                               ▼                                          │
│                    AnomalyResult { score, severity, triggered_rules }    │
└───────────────────────┬──────────────────────────────────────────────────┘
                        │
           ┌────────────┼─────────────────┐
           ▼            ▼                 ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────────────────────────┐
│  FastAPI     │ │  Claude      │ │  MCP Server                          │
│  :8080       │ │  SRE Agent   │ │  (Claude Desktop)                    │
│              │ │              │ │                                      │
│  Dashboard   │ │  Agentic     │ │  8 tools exposed via stdio transport │
│  /alerts/*   │ │  loop:       │ │  4 prompt templates                  │
│  /metrics/*  │ │  Observe →   │ │  3 live resources                    │
│  /pipeline/* │ │  Diagnose →  │ │                                      │
│  /health/*   │ │  Act →       │ │  k8s://cluster/health                │
│              │ │  Verify →    │ │  k8s://model/metrics                 │
│  Swagger UI  │ │  Report      │ │  k8s://alerts/recent                 │
└──────────────┘ └──────────────┘ └──────────────────────────────────────┘
```

---

## The 11 Metrics We Track

These were chosen because they cover the four main failure modes we've seen in real K8s clusters: resource exhaustion, workload instability, application degradation, and network saturation.

| Metric | What it tells you |
|---|---|
| `cpu_usage_pct` | Node-level CPU — obvious, but combined with latency it tells a different story |
| `memory_usage_pct` | OOM kills start here, long before the node goes NotReady |
| `pod_restart_count` | A crash-looping pod will restart quietly for minutes before anyone notices |
| `request_latency_p99_ms` | P99 catches tail latency that averages hide completely |
| `error_rate_pct` | HTTP 5xx rate — the symptom closest to what users actually experience |
| `network_bytes_in` | Unusual inbound traffic often precedes DDoS or runaway scraping |
| `network_bytes_out` | Data exfiltration, noisy logging pipelines, backup floods |
| `disk_io_read_mbps` | Database queries gone wrong, log file readers stuck in loops |
| `disk_io_write_mbps` | Write amplification, runaway log generation, PVC filling up |
| `pod_pending_count` | Scheduler can't place pods — resource exhaustion or taint mismatch |
| `node_not_ready_count` | The most serious single indicator — any non-zero value is a crisis |

---

## Why IsolationForest

Most anomaly detection approaches start with labelled data — "here are examples of anomalies, learn to find them." That doesn't work well in K8s because every cluster is different, every failure mode is different, and you almost never have a clean labelled dataset of historical incidents.

IsolationForest works the other way around. You train it on normal data only, and it learns to isolate outliers by randomly splitting the feature space. Points that are easy to isolate (require few splits to separate) are anomalous. Points that need many splits to separate look like everything else and are normal.

The practical advantage: you only need healthy cluster data to train it. Anomalies will stand out regardless of whether you've seen that specific failure mode before.

We train on normal samples only (the `is_anomaly == 0` rows from the simulator) and let the model figure out what doesn't fit.

---

## Project Structure

```
MLOPS-poc/
│
├── main.py                          # Run all 4 pipeline stages
├── config/
│   ├── config.yaml                  # Paths, Prometheus URL, simulator toggle
│   ├── params.yaml                  # IsolationForest hyperparams, thresholds
│   └── schema.yaml                  # Feature definitions + PromQL queries
│
├── src/mlops_poc/
│   ├── logging/                     # Rotating file + console logger
│   ├── exception/                   # Custom exception with file/line context
│   ├── utils/                       # YAML reader, joblib save/load, JSON utils
│   ├── constants/                   # Feature column list, Severity levels
│   ├── entity/                      # Typed config dataclasses (frozen)
│   ├── config/                      # ConfigurationManager — vends typed configs
│   ├── components/                  # One file per pipeline stage
│   │   ├── data_ingestion.py
│   │   ├── prepare_base_model.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   └── pipeline/                    # Stage runners + prediction pipeline
│       ├── stage_01_data_ingestion.py
│       ├── stage_02_prepare_base_model.py
│       ├── stage_03_model_trainer.py
│       ├── stage_04_model_evaluation.py
│       └── prediction_pipeline.py
│
├── simulator/
│   └── metrics_simulator.py         # 80% normal / 12% warning / 8% critical
│
├── prometheus/
│   ├── queries.py                   # PrometheusClient + full PromQL dictionary
│   ├── prometheus.yml               # Scrape configs for K8s exporters
│   └── alert_rules.yml              # 11 Prometheus alerting rules
│
├── app/
│   ├── main.py                      # FastAPI app, lifespan, CORS, middleware
│   ├── schemas/k8s_metrics.py       # Pydantic request/response models
│   └── routes/
│       ├── alerts.py                # /alerts/* — ML inference endpoints
│       ├── metrics.py               # /metrics/* — metrics collection
│       ├── health.py                # /health/* — K8s liveness/readiness
│       └── pipeline.py              # /pipeline/* — trigger retraining
│
├── templates/
│   └── index.html                   # Dark-mode monitoring dashboard (pure HTML/JS)
│
├── agentic_ai/
│   ├── tools/k8s_tools.py           # 11 tool definitions + HTTP executor
│   └── agents/k8s_agent.py          # Claude agentic loop, incident response
│
├── mcp_server/
│   └── server.py                    # MCP stdio server for Claude Desktop
│
├── kubeflow/
│   └── pipeline.py                  # KFP v2 pipeline (3-step DAG)
│
├── k8s/                             # Kubernetes manifests
│   ├── namespace.yaml
│   ├── rbac.yaml                    # ServiceAccount + ClusterRole
│   ├── pvc.yaml                     # 5Gi artifacts + 10Gi MLflow storage
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── deployment.yaml              # API (2 replicas) + MLflow + CronJob
│   ├── service.yaml                 # ClusterIP + Ingress
│   └── hpa.yaml                     # Autoscale 2-8 replicas
│
├── tests/
│   ├── unit/
│   │   ├── test_simulator.py        # 9 tests
│   │   └── test_anomaly_model.py    # 4 tests
│   └── integration/
│       └── test_api.py              # 20+ FastAPI endpoint tests
│
├── Dockerfile                       # Multi-stage, non-root user
├── docker-compose.yml               # Prometheus + MLflow + API
└── scripts/
    ├── setup.sh
    ├── start_mlflow.sh
    ├── start_api.sh
    └── deploy_k8s.sh
```

---

## Getting Started Locally

You don't need a Kubernetes cluster, Prometheus, or any cloud infrastructure to run this. The simulator handles all of that.

**Step 1 — Install**

```bash
cd /Users/nandak/Downloads/MLOPS-poc
pip3 install -r requirements.txt
pip3 install -e .
```

**Step 2 — Train the model**

```bash
python3 main.py
```

This runs all four stages. Takes about 30 seconds. When it finishes you'll have a trained model at `artifacts/model_trainer/anomaly_detector.joblib` and metrics logged to MLflow.

**Step 3 — Start the API**

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Open `http://localhost:8080` — that's the monitoring dashboard. Open `http://localhost:8080/docs` to try the API endpoints interactively.

**Step 4 — MLflow UI (optional)**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Shows the experiment run with all logged params, metrics, and the model artifact.

---

## The Four Pipeline Stages

### Stage 1 — Data Ingestion

Pulls 11 metrics either from a real Prometheus instance or from the simulator. Splits the data so the training set contains only normal samples — this is important because IsolationForest needs clean data to learn from. Saves a separate labelled file for supervised evaluation later.

Toggle in `config/config.yaml`:
```yaml
data_ingestion:
  use_simulator: true   # flip to false for real Prometheus
  prometheus_url: http://localhost:9090
```

### Stage 2 — Prepare Base Model

Builds the sklearn Pipeline object: `StandardScaler → IsolationForest`. Supports swapping to RobustScaler or MinMaxScaler via params.yaml if you're dealing with heavy outliers in the training data.

### Stage 3 — Model Trainer

Fits the pipeline on normal samples only. After fitting, it logs the anomaly rate on both train and test sets as a sanity check — if your training set shows 30% anomalies you've probably got a data problem.

### Stage 4 — Model Evaluation + MLflow

Computes Precision, Recall, F1, and ROC-AUC against labelled test data. Logs everything to MLflow — params, metrics, tags, and the model artifact. Saves `metrics.json` for DVC to track. If you have a remote MLflow instance it'll register the model in the registry and transition it to Staging.

---

## API Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Monitoring dashboard |
| POST | `/alerts/check` | Run anomaly detection on submitted metrics |
| POST | `/alerts/check/batch` | Same but for a list of snapshots |
| POST | `/alerts/simulate` | Generate a synthetic snapshot and detect — good for demos |
| GET | `/alerts/history` | Last 200 alerts (in-memory ring buffer) |
| DELETE | `/alerts/history` | Clear the buffer |
| GET | `/metrics/current` | Latest cluster snapshot from simulator |
| GET | `/metrics/summary` | Averages across recent history |
| POST | `/metrics/ingest` | Push your own metrics in |
| POST | `/pipeline/run/{stage}` | Trigger stages 1-4 in the background |
| GET | `/pipeline/status` | What artifacts exist, what's the pipeline doing |
| GET | `/pipeline/metrics` | Latest evaluation metrics |
| GET | `/health/live` | Kubernetes liveness probe |
| GET | `/health/ready` | Kubernetes readiness probe |

---

## The AI Agent

The agent uses Claude with tool use to operate the cluster autonomously. You give it a natural language instruction and it figures out what to do:

```bash
export ANTHROPIC_API_KEY=your-key
python3 -m agentic_ai.agents.k8s_agent

You > check the cluster and tell me if anything looks wrong
You > the latency has been spiking for 10 minutes, what should I do
You > incident      # triggers full incident response workflow
```

The incident response workflow follows a fixed pattern: gather cluster health → run anomaly detection → check recent alerts → identify root cause from triggered rules → propose or execute remediation (restart/scale) → verify recovery → write a report. If it can't resolve something it'll say so clearly and tell you what to escalate.

The 11 tools available to the agent:

| Tool | What the agent uses it for |
|---|---|
| `get_cluster_health` | First thing it checks |
| `detect_anomaly_now` | Run real-time detection |
| `get_alert_history` | Look for patterns across recent alerts |
| `get_pipeline_status` | Check if model artifacts are ready |
| `run_pipeline_stage` | Retrain the model with fresh data |
| `get_model_metrics` | Check if model performance has degraded |
| `get_service_health` | Verify API is up |
| `get_current_metrics` | Pull the latest snapshot |
| `list_k8s_pods` | Inspect pod states |
| `restart_deployment` | Rolling restart a deployment |
| `scale_deployment` | Scale replicas up or down |

---

## MCP Server (Claude Desktop)

If you use Claude Desktop, you can add the monitoring tools directly to your chat sessions.

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "k8s-monitoring": {
      "command": "python3",
      "args": ["/Users/nandak/Downloads/MLOPS-poc/mcp_server/server.py"],
      "env": {
        "API_BASE_URL": "http://localhost:8080"
      }
    }
  }
}
```

Restart Claude Desktop and you'll have four prompt templates available:

- **k8s-incident-response** — structured investigation for a specific namespace
- **k8s-daily-standup** — quick cluster health summary
- **k8s-retrain-model** — guided retraining walkthrough
- **k8s-explain-anomaly** — plain-English explanation of an anomaly score

Plus three live resources that Claude can read at any time: current cluster health, latest model metrics, and recent alerts.

---

## DVC Pipeline

Every training run is reproducible and tracked:

```bash
dvc repro                  # Run the full pipeline
dvc metrics diff           # Compare metrics between runs
dvc push                   # Push artifacts to remote storage
```

The DAG in `dvc.yaml` tracks which files each stage depends on and produces. If you change `params.yaml`, DVC knows stages 2-4 need to rerun. If you only change the evaluation code, only stage 4 reruns. Nothing runs unnecessarily.

---

## Kubernetes Deployment

The stack runs on any cluster with a PVC-capable storage class:

```bash
export IMAGE=your-registry/mlops-monitoring:latest
bash scripts/deploy_k8s.sh
```

What gets deployed:
- **Monitoring API** — 2 replicas with rolling update, liveness/readiness probes, spread across nodes
- **MLflow Server** — single replica with 10Gi persistent volume
- **CronJob** — retraining job every 6 hours, runs `python3 main.py` against the same artifacts PVC
- **HPA** — autoscales the API from 2 to 8 replicas based on CPU (70%) and memory (80%)

---

## Simulator Details

The simulator is what makes this runnable without any infrastructure. It generates data in three states with realistic distributions:

**Normal (80%)** — CPU around 40%, memory around 55%, latency log-normal around 120ms, almost no restarts or pending pods. This is what the model learns from.

**Warning (12%)** — One of four scenarios: CPU spike to 80-92%, memory pressure with pods going pending, latency spike to 1-2 seconds with elevated errors, or an error rate spike with pod restarts.

**Critical (8%)** — One of four scenarios: node failure with NotReady nodes and pending pods, OOM kill with full memory and restarting pods, network saturation with huge byte counts and 9-second latency, or a full outage with everything elevated simultaneously.

The simulator uses the same statistical profiles for each state every time (seeded with `random_state=42` during training) so results are reproducible.

---

## Running Tests

```bash
# Unit tests — fast, no API needed
python3 -m pytest tests/unit/ -v

# Integration tests — spins up a TestClient with a minimal trained model
python3 -m pytest tests/integration/ -v

# Everything with coverage
python3 -m pytest tests/ -v --cov=src/mlops_poc --cov=simulator --cov=app
```

The integration tests are fully self-contained — they train a small IsolationForest in a temp directory, monkey-patch the pipeline to use it, and run the full FastAPI stack in-process with no external dependencies.

---

## Configuration Reference

Everything important lives in three files:

**`config/config.yaml`** — artifact paths, Prometheus URL, simulator settings, MLflow URI. Change `use_simulator: false` and set `prometheus_url` to point at a real cluster.

**`config/params.yaml`** — IsolationForest hyperparameters and all threshold values. Adjust `contamination` if your cluster runs hotter than normal — it controls how aggressively the model flags anomalies during training.

**`config/schema.yaml`** — feature definitions with expected ranges and the PromQL query for each metric when running against real Prometheus.

---

## Troubleshooting

**`python3 main.py` fails on Stage 4 with connection refused**

MLflow is trying to reach `localhost:5000` and nothing is there. The config ships pointing at a local SQLite database now, but if you changed it back:

```yaml
# config/config.yaml
model_evaluation:
  mlflow_uri: sqlite:///Users/nandak/Downloads/MLOPS-poc/mlflow.db
```

**Dashboard loads but shows no data**

The model needs to be trained first. Run `python3 main.py` from the project root before starting the API.

**`pip3: command not found`**

Use `pip3` instead of `pip`. The project was built with Python 3.14 via Homebrew where the binary is `pip3`.

**Jinja2 template error on Python 3.14**

The dashboard uses `FileResponse` instead of Jinja2 templates precisely because of a compatibility issue between Jinja2 3.1.x and Python 3.14's changes to `weakref`. No fix needed — this is already handled in `app/main.py`.
