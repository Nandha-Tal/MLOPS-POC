"""ConfigurationManager — reads YAMLs and vends typed config objects."""
import sys
from pathlib import Path

from mlops_poc.constants import FEATURE_COLUMNS
from mlops_poc.entity import (
    DataIngestionConfig, ModelEvaluationConfig,
    ModelTrainerConfig, PrepareBaseModelConfig,
)
from mlops_poc.exception import MLOpsException
from mlops_poc.utils import create_directories, read_yaml

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")
SCHEMA_FILE_PATH = Path("config/schema.yaml")


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):
        try:
            self.config = read_yaml(config_filepath)
            self.params = read_yaml(params_filepath)
            self.schema = read_yaml(schema_filepath)
            create_directories([self.config.artifacts_root])
        except Exception as e:
            raise MLOpsException(e, sys) from e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config.data_ingestion
        create_directories([cfg.root_dir])
        return DataIngestionConfig(
            root_dir=Path(cfg.root_dir),
            prometheus_url=cfg.prometheus_url,
            lookback_hours=cfg.lookback_hours,
            scrape_interval=cfg.scrape_interval,
            raw_metrics_file=Path(cfg.raw_metrics_file),
            train_data_path=Path(cfg.train_data_path),
            test_data_path=Path(cfg.test_data_path),
            use_simulator=cfg.use_simulator,
            simulator_samples=cfg.simulator_samples,
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        cfg = self.config.prepare_base_model
        create_directories([cfg.root_dir])
        return PrepareBaseModelConfig(
            root_dir=Path(cfg.root_dir),
            base_model_path=Path(cfg.base_model_path),
            updated_base_model_path=Path(cfg.updated_base_model_path),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg = self.config.model_trainer
        create_directories([cfg.root_dir])
        return ModelTrainerConfig(
            root_dir=Path(cfg.root_dir),
            train_data_path=Path(cfg.train_data_path),
            test_data_path=Path(cfg.test_data_path),
            model_name=cfg.model_name,
            feature_columns=list(cfg.feature_columns),
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        cfg = self.config.model_evaluation
        create_directories([cfg.root_dir])
        return ModelEvaluationConfig(
            root_dir=Path(cfg.root_dir),
            test_data_path=Path(cfg.test_data_path),
            model_path=Path(cfg.model_path),
            metric_file_name=Path(cfg.metric_file_name),
            all_params=Path(cfg.all_params),
            mlflow_uri=cfg.mlflow_uri,
            labelled_data_path=Path(cfg.labelled_data_path),
        )
