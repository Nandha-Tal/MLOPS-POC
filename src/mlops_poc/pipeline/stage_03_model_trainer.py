import sys
from mlops_poc.config import ConfigurationManager
from mlops_poc.components.model_trainer import ModelTrainer
from mlops_poc.exception import MLOpsException


class ModelTrainerPipeline:
    def run(self):
        try:
            cfg = ConfigurationManager()
            trainer_cfg = cfg.get_model_trainer_config()
            prepare_cfg = cfg.get_prepare_base_model_config()
            ModelTrainer(trainer_cfg).train(prepare_cfg.updated_base_model_path)
        except Exception as e:
            raise MLOpsException(e, sys) from e


if __name__ == "__main__":
    ModelTrainerPipeline().run()
