"""
Model Trainer Component
========================
Stage 3: Fit the IsolationForest pipeline on normal-only K8s metrics.
"""
import sys

import pandas as pd

from mlops_poc.constants import FEATURE_COLUMNS
from mlops_poc.entity import ModelTrainerConfig
from mlops_poc.exception import MLOpsException
from mlops_poc.logging import logger
from mlops_poc.utils import load_object, save_object


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self, base_model_path):
        try:
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            feature_cols = [c for c in self.config.feature_columns if c in train_df.columns]
            X_train = train_df[feature_cols].fillna(0.0)
            X_test = test_df[feature_cols].fillna(0.0)

            logger.info("Training on %d normal samples with %d features", len(X_train), len(feature_cols))

            pipe = load_object(base_model_path)
            pipe.fit(X_train)

            # Diagnostics
            train_preds = pipe.predict(X_train)
            test_preds = pipe.predict(X_test)
            train_anomaly_rate = (train_preds == -1).mean()
            test_anomaly_rate = (test_preds == -1).mean()

            logger.info("Train anomaly rate: %.2f%%", train_anomaly_rate * 100)
            logger.info("Test anomaly rate:  %.2f%%", test_anomaly_rate * 100)

            model_path = self.config.root_dir / self.config.model_name
            save_object(model_path, pipe)
            logger.info("Trained model saved: %s", model_path)

            return model_path

        except Exception as e:
            raise MLOpsException(e, sys) from e
