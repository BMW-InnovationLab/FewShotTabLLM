"""
Data-loading helpers for the MCP layer.

Thin wrappers around the evaluator's own loaders so the service layer
doesn't need to know CSV paths vs. dicts vs. metadata files.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from synthetic_data_evaluator.core.models import EvaluationParameters
from synthetic_data_evaluator.engine.evaluation_engine import (
    load_csv_data,
    load_metadata_from_file,
    prepare_data,
)
from synthetic_data_evaluator.helpers.data_helper import generate_metadata


def load_and_prepare(
    real_data_path: str,
    synth_data_path: str,
    params: Optional[Dict],
    metadata: Optional[Dict] = None,
    table_names: Optional[List[str]] = None,
) -> Tuple[Any, Any, Any, str, str, EvaluationParameters]:
    
    """Load CSVs, build metadata if needed, and prepare data.

    Returns:
        (real_data, synthetic_data, dataset_metadata, table_name, modality, params_obj)
    """
    real_data, synthetic_data, table_name, modality = load_csv_data(
        [real_data_path], [synth_data_path], table_names=table_names,
    )

    params_obj = EvaluationParameters(**(params or {}))

    if metadata:
        dataset_metadata = load_metadata_from_file(metadata)
    elif isinstance(real_data, dict):
        dataset_metadata = generate_metadata(real_data)
    else:
        dataset_metadata = generate_metadata({table_name: real_data})

    real_data, synthetic_data, dataset_metadata, _ = prepare_data(
        real_data, synthetic_data, dataset_metadata, table_name, params_obj, modality,
    )

    return real_data, synthetic_data, dataset_metadata, table_name, modality, params_obj


def build_metadata(real_data_paths: List[str]) -> Dict[str, Any]:
    """Generate metadata from one or more CSV paths."""
    if isinstance(real_data_paths, str):
        real_data_paths = [real_data_paths]

    datasets = {
        f"table_{i}": pd.read_csv(path)
        for i, path in enumerate(real_data_paths)
    }
    return generate_metadata(datasets)
