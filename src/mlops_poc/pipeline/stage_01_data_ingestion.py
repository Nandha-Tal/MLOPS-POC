import sys
from mlops_poc.config import ConfigurationManager
from mlops_poc.components.data_ingestion import DataIngestion
from mlops_poc.exception import MLOpsException
from mlops_poc.logging import logger


class DataIngestionPipeline:
    def run(self):
        try:
            cfg = ConfigurationManager().get_data_ingestion_config()
            DataIngestion(cfg).initiate_data_ingestion()
        except Exception as e:
            raise MLOpsException(e, sys) from e


if __name__ == "__main__":
    DataIngestionPipeline().run()
