"""
Prediction Pipeline
====================
Real-time anomaly detection combining ML model + rule-based thresholds.
"""
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from mlops_poc.constants import FEATURE_COLUMNS, Severity
from mlops_poc.exception import MLOpsException
from mlops_poc.logging import logger
from mlops_poc.utils import load_object

MODEL_PATH = Path("artifacts/model_trainer/anomaly_detector.joblib")

# Thresholds for rule-based detection
CPU_WARNING = 80.0
CPU_CRITICAL = 90.0
MEM_WARNING = 85.0
MEM_CRITICAL = 95.0
LATENCY_WARNING = 1000.0
LATENCY_CRITICAL = 2000.0
ERROR_WARNING = 5.0
ERROR_CRITICAL = 15.0
ML_SCORE_WARNING = -0.05
ML_SCORE_CRITICAL = -0.15


@dataclass
class AnomalyResult:
    timestamp: str
    anomaly_score: float
    is_anomaly: bool
    severity: str
    triggered_rules: List[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class PredictionPipeline:
    def __init__(self, model_path: Path = MODEL_PATH):
        self._model_path = model_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not self._model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self._model_path}. Run: python main.py"
                )
            self._model = load_object(self._model_path)
            logger.info("Anomaly detector loaded from %s", self._model_path)
        return self._model

    @staticmethod
    def _check_rules(metrics: dict) -> List[str]:
        rules = []
        cpu = metrics.get("cpu_usage_pct", 0)
        mem = metrics.get("memory_usage_pct", 0)
        lat = metrics.get("request_latency_p99_ms", 0)
        err = metrics.get("error_rate_pct", 0)
        restarts = metrics.get("pod_restart_count", 0)
        pending = metrics.get("pod_pending_count", 0)
        not_ready = metrics.get("node_not_ready_count", 0)

        if cpu >= CPU_CRITICAL:     rules.append("cpu_critical")
        elif cpu >= CPU_WARNING:    rules.append("cpu_warning")
        if mem >= MEM_CRITICAL:     rules.append("memory_critical")
        elif mem >= MEM_WARNING:    rules.append("memory_warning")
        if lat >= LATENCY_CRITICAL: rules.append("latency_critical")
        elif lat >= LATENCY_WARNING: rules.append("latency_warning")
        if err >= ERROR_CRITICAL:   rules.append("error_rate_critical")
        elif err >= ERROR_WARNING:  rules.append("error_rate_warning")
        if restarts >= 10:          rules.append("pod_restart_critical")
        elif restarts >= 5:         rules.append("pod_restart_warning")
        if pending >= 15:           rules.append("pod_pending_critical")
        elif pending >= 5:          rules.append("pod_pending_warning")
        if not_ready >= 1:          rules.append("node_not_ready")
        return rules

    def detect(self, metrics: dict) -> AnomalyResult:
        try:
            import pandas as pd
            feature_vals = {col: metrics.get(col, 0.0) for col in FEATURE_COLUMNS}
            X = pd.DataFrame([feature_vals])[FEATURE_COLUMNS].fillna(0.0)

            score = float(self.model.decision_function(X)[0])
            pred = int(self.model.predict(X)[0])
            rules = self._check_rules(metrics)

            critical_rules = [r for r in rules if "critical" in r]
            warning_rules  = [r for r in rules if "warning" in r]

            is_anomaly = (pred == -1) or bool(rules)

            if score <= ML_SCORE_CRITICAL or critical_rules:
                severity = Severity.CRITICAL
            elif score <= ML_SCORE_WARNING or warning_rules:
                severity = Severity.WARNING
            else:
                severity = Severity.OK

            return AnomalyResult(
                timestamp=metrics.get("timestamp", datetime.now().isoformat()),
                anomaly_score=round(score, 6),
                is_anomaly=is_anomaly,
                severity=severity,
                triggered_rules=rules,
                metrics=feature_vals,
            )

        except Exception as e:
            raise MLOpsException(e, sys) from e
