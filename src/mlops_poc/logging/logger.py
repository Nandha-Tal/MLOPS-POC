"""Rotating file + console logger for the MLOps POC."""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "mlops_poc.log"

LOG_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("mlops_poc")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

if not logger.handlers:
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(ch)

    # Rotating file handler (5 MB × 3 backups)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(fh)
