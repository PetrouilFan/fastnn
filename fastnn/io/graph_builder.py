"""Build runnable models from .fnn headers.

Supports both Sequential models (from PyTorch export) and
DAG models (from ONNX import).
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastnn.io import read_fnn_header, read_fnn_parameters, SerializationError, MODEL_MAGIC

logger = logging.getLogger(__name__)


def build_model_from_fnn(path: str) -> Any:
    """Build a runnable model from a .fnn file.

    Automatically detects whether the file contains a sequential
    layer list (PyTorch export) or a full graph topology (ONNX import).

    Args:
        path: Path to .fnn file.

    Returns:
        A fastnn model (Sequential for PyTorch-exported, DAGModel for ONNX-imported).
    """
    with open(path, "rb") as f:
        magic, file_version, header, num_params = read_fnn_header(f)
        if magic != MODEL_MAGIC:
            raise SerializationError("Invalid .fnn file: missing magic bytes")

        if "graph" in header:
            return build_dag_model(header, f, num_params)
        elif "layers" in header:
            return build_sequential_model(path)
        else:
            raise ValueError("Unknown .fnn format: header has neither 'graph' nor 'layers'")


def build_dag_model(header: dict, f, num_params: int) -> Any:
    """Build a DAGModel from an ONNX-imported .fnn file."""
    from fastnn.io.dag_model import DAGModel
    params = read_fnn_parameters(f, num_params)
    return DAGModel.from_header(header, params)


def build_sequential_model(path: str) -> Any:
    """Build a Sequential model from a PyTorch-exported .fnn file."""
    from fastnn.io.export import load_fnn_model
    return load_fnn_model(path)
