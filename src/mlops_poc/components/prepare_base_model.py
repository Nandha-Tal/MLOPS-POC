"""
Prepare Base Model Component
=============================
Stage 2: Build the sklearn Pipeline (Scaler + IsolationForest).
"""
import sys

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from mlops_poc.entity import PrepareBaseModelConfig
from mlops_poc.exception import MLOpsException
from mlops_poc.logging import logger
from mlops_poc.utils import read_yaml, save_object

SCALERS = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "MinMaxScaler": MinMaxScaler,
}


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def _build_pipeline(self, params) -> Pipeline:
        scaler_name = params.scaler.get("type", "StandardScaler")
        scaler = SCALERS.get(scaler_name, StandardScaler)()

        iso_params = params.isolation_forest
        model = IsolationForest(
            n_estimators=iso_params.n_estimators,
            contamination=iso_params.contamination,
            max_samples=iso_params.get("max_samples", "auto"),
            random_state=iso_params.random_state,
            bootstrap=iso_params.get("bootstrap", False),
        )
        return Pipeline([("scaler", scaler), ("model", model)])

    def get_base_model(self, params_path):
        try:
            params = read_yaml(params_path)
            pipe = self._build_pipeline(params)
            save_object(self.config.base_model_path, pipe)
            logger.info("Base model pipeline saved: %s", self.config.base_model_path)

            # Copy as updated model (same at this stage)
            save_object(self.config.updated_base_model_path, pipe)
            logger.info("Updated base model saved: %s", self.config.updated_base_model_path)

            return pipe
        except Exception as e:
            raise MLOpsException(e, sys) from e
