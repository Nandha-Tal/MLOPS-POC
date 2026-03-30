import sys
from pathlib import Path
from mlops_poc.config import ConfigurationManager
from mlops_poc.components.prepare_base_model import PrepareBaseModel
from mlops_poc.exception import MLOpsException


class PrepareBaseModelPipeline:
    def run(self):
        try:
            cfg = ConfigurationManager()
            model_cfg = cfg.get_prepare_base_model_config()
            PrepareBaseModel(model_cfg).get_base_model(Path("config/params.yaml"))
        except Exception as e:
            raise MLOpsException(e, sys) from e


if __name__ == "__main__":
    PrepareBaseModelPipeline().run()
