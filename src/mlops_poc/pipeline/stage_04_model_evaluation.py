import sys
from mlops_poc.config import ConfigurationManager
from mlops_poc.components.model_evaluation import ModelEvaluation
from mlops_poc.exception import MLOpsException


class ModelEvaluationPipeline:
    def run(self):
        try:
            cfg = ConfigurationManager().get_model_evaluation_config()
            ModelEvaluation(cfg).log_into_mlflow()
        except Exception as e:
            raise MLOpsException(e, sys) from e


if __name__ == "__main__":
    ModelEvaluationPipeline().run()
