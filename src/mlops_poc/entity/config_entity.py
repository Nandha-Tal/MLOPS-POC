"""Typed configuration dataclasses — one per pipeline stage."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    prometheus_url: str
    lookback_hours: int
    scrape_interval: int
    raw_metrics_file: Path
    train_data_path: Path
    test_data_path: Path
    use_simulator: bool
    simulator_samples: int


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    feature_columns: list


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    all_params: Path
    mlflow_uri: str
    labelled_data_path: Path
