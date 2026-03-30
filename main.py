"""
main.py — K8s Monitoring Pipeline Orchestrator
================================================
Runs all 4 pipeline stages to train the K8s anomaly detector.

Usage:
    python main.py              # Run all stages
    python main.py --stage 1    # Data Ingestion only
    python main.py --stage 3    # Model Training only
"""
import argparse
import sys

from mlops_poc.exception import MLOpsException
from mlops_poc.logging import logger
from mlops_poc.pipeline import (
    DataIngestionPipeline, PrepareBaseModelPipeline,
    ModelTrainerPipeline, ModelEvaluationPipeline,
)

STAGES = {
    1: ("Data Ingestion (Prometheus/Simulator)", DataIngestionPipeline),
    2: ("Prepare Base Model (IsolationForest Pipeline)", PrepareBaseModelPipeline),
    3: ("Model Trainer (Fit on Normal Metrics)", ModelTrainerPipeline),
    4: ("Model Evaluation + MLflow Logging", ModelEvaluationPipeline),
}


def run_stage(stage_num: int):
    name, PipelineClass = STAGES[stage_num]
    logger.info("=" * 60)
    logger.info(">>> Stage %d: %s — STARTED", stage_num, name)
    logger.info("=" * 60)
    try:
        PipelineClass().run()
        logger.info(">>> Stage %d: COMPLETED ✓", stage_num)
    except Exception as e:
        logger.exception(">>> Stage %d: FAILED ✗", stage_num)
        raise MLOpsException(e, sys) from e


def main(stages=None):
    for s in (stages or list(STAGES.keys())):
        run_stage(s)
    logger.info("=" * 60)
    logger.info("K8s Anomaly Detector training COMPLETE")
    logger.info("Start the monitoring API: uvicorn app.main:app --port 8080")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K8s Monitoring Pipeline")
    parser.add_argument("--stage", type=int, action="append", choices=list(STAGES.keys()))
    args = parser.parse_args()
    main(stages=args.stage)
