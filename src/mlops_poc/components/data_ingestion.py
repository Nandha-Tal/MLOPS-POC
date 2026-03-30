"""
Data Ingestion Component
========================
Stage 1: Collect K8s metrics from Prometheus or the simulator.
Splits data into train (normal only) and test sets.
"""
import sys
from pathlib import Path

import pandas as pd

from mlops_poc.constants import FEATURE_COLUMNS
from mlops_poc.entity import DataIngestionConfig
from mlops_poc.exception import MLOpsException
from mlops_poc.logging import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _collect_from_simulator(self) -> pd.DataFrame:
        from simulator.metrics_simulator import K8sMetricsSimulator
        sim = K8sMetricsSimulator(random_state=42)
        df = sim.generate(n_samples=self.config.simulator_samples)
        logger.info("Simulator generated %d samples", len(df))
        return df

    def _collect_from_prometheus(self) -> pd.DataFrame:
        from prometheus.queries import PrometheusClient
        client = PrometheusClient(self.config.prometheus_url)
        df = client.fetch_training_data(lookback_hours=self.config.lookback_hours)
        logger.info("Prometheus returned %d samples", len(df))
        return df

    def initiate_data_ingestion(self):
        try:
            if self.config.use_simulator:
                df = self._collect_from_simulator()
            else:
                df = self._collect_from_prometheus()

            # Save raw metrics
            df.to_csv(self.config.raw_metrics_file, index=False)
            logger.info("Raw metrics saved: %s", self.config.raw_metrics_file)

            # Save labelled data if available
            if "is_anomaly" in df.columns:
                labelled_path = self.config.root_dir / "labelled_anomalies.csv"
                df.to_csv(labelled_path, index=False)
                logger.info("Labelled anomalies saved: %s", labelled_path)

            # Split: train = normal only, test = everything
            feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

            if "is_anomaly" in df.columns:
                train_df = df[df["is_anomaly"] == 0][feature_cols].copy()
                if len(train_df) < 50:
                    logger.warning("Very few normal samples (%d) — using all data for training", len(train_df))
                    train_df = df[feature_cols].copy()
            else:
                train_df = df[feature_cols].copy()

            test_df = df[feature_cols + (["is_anomaly"] if "is_anomaly" in df.columns else [])].copy()

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logger.info("Train: %d samples | Test: %d samples", len(train_df), len(test_df))
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise MLOpsException(e, sys) from e
