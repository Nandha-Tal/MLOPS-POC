"""
Model Evaluation Component + MLflow Integration
================================================
Stage 4: Evaluate the anomaly detector and log to MLflow.
"""
import sys
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlops_poc.constants import FEATURE_COLUMNS
from mlops_poc.entity import ModelEvaluationConfig
from mlops_poc.exception import MLOpsException
from mlops_poc.logging import logger
from mlops_poc.utils import load_object, read_yaml, save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    @staticmethod
    def _compute_supervised_metrics(y_true, y_pred, scores) -> dict:
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

        y_pred_binary = (y_pred == -1).astype(int)
        scores_norm = (-scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        try:
            auc = float(roc_auc_score(y_true, scores_norm))
        except Exception:
            auc = 0.0

        return {
            "precision":    round(float(precision_score(y_true, y_pred_binary, zero_division=0)), 4),
            "recall":       round(float(recall_score(y_true, y_pred_binary, zero_division=0)), 4),
            "f1_score":     round(float(f1_score(y_true, y_pred_binary, zero_division=0)), 4),
            "roc_auc":      round(auc, 4),
            "anomaly_rate": round(float(y_pred_binary.mean()), 4),
        }

    def log_into_mlflow(self) -> dict:
        try:
            model = load_object(self.config.model_path)
            all_params = read_yaml(self.config.all_params)

            mlflow.set_tracking_uri(self.config.mlflow_uri)
            tracking_scheme = urlparse(self.config.mlflow_uri).scheme

            has_labels = self.config.labelled_data_path.exists()
            if has_labels:
                eval_df = pd.read_csv(self.config.labelled_data_path)
                y_true = eval_df["is_anomaly"].values if "is_anomaly" in eval_df.columns else None
                feature_cols = [c for c in FEATURE_COLUMNS if c in eval_df.columns]
                X_eval = eval_df[feature_cols].fillna(0.0)
            else:
                test_df = pd.read_csv(self.config.test_data_path)
                feature_cols = [c for c in FEATURE_COLUMNS if c in test_df.columns]
                X_eval = test_df[feature_cols].fillna(0.0)
                y_true = None

            predictions = model.predict(X_eval)
            scores = model.decision_function(X_eval)
            anomaly_rate = float((predictions == -1).mean())

            if y_true is not None and has_labels:
                metrics = self._compute_supervised_metrics(y_true, predictions, scores)
            else:
                metrics = {
                    "anomaly_rate": round(anomaly_rate, 4),
                    "score_mean":   round(float(scores.mean()), 4),
                    "score_std":    round(float(scores.std()), 4),
                    "score_min":    round(float(scores.min()), 4),
                }

            logger.info("Evaluation metrics: %s", metrics)

            with mlflow.start_run(run_name="k8s-anomaly-eval") as run:
                for section, vals in all_params.items():
                    if isinstance(vals, dict):
                        for k, v in vals.items():
                            mlflow.log_param(f"{section}.{k}", v)

                mlflow.log_metrics(metrics)
                mlflow.set_tags({
                    "model_type": "IsolationForest",
                    "use_case":   "k8s-anomaly-detection",
                    "framework":  "scikit-learn",
                    "labelled_eval": str(has_labels),
                })

                if tracking_scheme not in ("file", "sqlite"):
                    mlflow.sklearn.log_model(model, artifact_path="model",
                                             registered_model_name="k8s-anomaly-detector")
                else:
                    mlflow.sklearn.log_model(model, artifact_path="model")

                save_json(path=Path(self.config.metric_file_name), data=metrics)

            return metrics

        except Exception as e:
            raise MLOpsException(e, sys) from e
