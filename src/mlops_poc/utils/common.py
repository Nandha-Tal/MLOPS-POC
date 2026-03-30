"""Shared utility functions used across all pipeline stages."""
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import yaml
from box import ConfigBox
from ensure import ensure_annotations

from mlops_poc.logging import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML file and return a ConfigBox (dot-access dict)."""
    try:
        with open(path_to_yaml) as f:
            content = yaml.safe_load(f)
        logger.info("YAML loaded: %s", path_to_yaml)
        return ConfigBox(content)
    except Exception as e:
        from mlops_poc.exception import MLOpsException
        raise MLOpsException(e, sys) from e


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """Create a list of directories if they don't exist."""
    for path in path_to_directories:
        Path(path).mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info("Directory created: %s", path)


def save_json(path: Path, data: dict):
    """Save a dict as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info("JSON saved: %s", path)


def load_json(path: Path) -> ConfigBox:
    """Load a JSON file as ConfigBox."""
    with open(path) as f:
        content = json.load(f)
    logger.info("JSON loaded: %s", path)
    return ConfigBox(content)


def save_object(path: Path, obj: Any):
    """Serialize any Python object with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Object saved: %s", path)


def load_object(path: Path) -> Any:
    """Load a joblib-serialized object."""
    obj = joblib.load(path)
    logger.info("Object loaded: %s", path)
    return obj


def get_size(path: Path) -> str:
    """Return human-readable file size."""
    size_kb = round(Path(path).stat().st_size / 1024, 2)
    return f"~ {size_kb} KB"
