"""MLflow helper utilities — experiment management, model registry."""
import mlflow
from mlflow.tracking import MlflowClient

from mlops_poc.logging import logger


def set_tracking_uri(uri: str):
    mlflow.set_tracking_uri(uri)
    logger.info("MLflow tracking URI: %s", uri)


def get_or_create_experiment(name: str) -> str:
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        exp_id = client.create_experiment(name)
        logger.info("Created MLflow experiment '%s' (id=%s)", name, exp_id)
    else:
        exp_id = exp.experiment_id
        logger.info("Using MLflow experiment '%s' (id=%s)", name, exp_id)
    return exp_id


def register_model(run_id: str, model_name: str, artifact_path: str = "model") -> int:
    """Register a logged model and return the version number."""
    client = MlflowClient()
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass  # already exists

    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
    logger.info("Registered model '%s' version %s", model_name, mv.version)
    return int(mv.version)


def transition_model_stage(model_name: str, version: int, stage: str):
    """Transition a model version to Staging or Production."""
    client = MlflowClient()
    client.transition_model_version_stage(name=model_name, version=version, stage=stage)
    logger.info("Model '%s' v%d → %s", model_name, version, stage)
