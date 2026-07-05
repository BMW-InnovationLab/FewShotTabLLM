"""
Validation logic for MCP tool parameters.

Moved from synthetic_data_evaluator/helpers/mcp_helper.py so the evaluator
package stays CLI-focused and validation lives next to the server that needs it.
"""

from typing import Any, Dict, List, Optional

from synthetic_data_evaluator.core.evaluation_info import (
    EVALUATION_HELPER,
    EVALUATION_REPORT_HELPER,
)
from synthetic_data_evaluator.core.param_registry import (
    METRIC_PARAMS,
    PARAM_DEFINITIONS,
    REPORT_METRICS,
    REPORT_PARAM_OVERRIDES,
)

# Parameters exposed as top-level MCP tool arguments (not inside ``params``).
_MCP_TOP_LEVEL_PARAMS = {"table_names", "primary_key", "foreign_key"}

# ---------------------------------------------------------------------------
# Required-parameter registries
# ---------------------------------------------------------------------------

def _build_report_required_params() -> Dict[str, Dict[str, Any]]:
    """Auto-generate per-report required-param registry from param_registry."""
    result: Dict[str, Dict[str, Any]] = {}
    for report, metrics in REPORT_METRICS.items():
        # Only keep params required by *every* metric in the report.
        #
        # Report execution tolerates per-metric failures and continues, so
        # report-level pre-validation should only block on truly global
        # requirements shared across all metrics.
        metric_param_sets = [set(METRIC_PARAMS.get(m, [])) for m in metrics]
        if not metric_param_sets:
            shared_params: set[str] = set()
        else:
            shared_params = set.intersection(*metric_param_sets)

        # Apply report-level overrides (add/remove swaps)
        overrides = REPORT_PARAM_OVERRIDES.get(report, {})
        shared_params -= set(overrides.get("remove", []))
        shared_params |= set(overrides.get("add", []))
        # Filter to required, non-common params
        params_list: List[str] = []
        top_level_list: List[str] = []
        for p in sorted(shared_params):
            defn = PARAM_DEFINITIONS.get(p, {})
            if not defn.get("required", False) or defn.get("common", False):
                continue
            if p in _MCP_TOP_LEVEL_PARAMS:
                top_level_list.append(p)
            else:
                params_list.append(p)
        if params_list or top_level_list:
            entry: Dict[str, Any] = {}
            if params_list:
                entry["params"] = params_list
            if top_level_list:
                entry["top_level"] = top_level_list
            result[report] = entry
    return result


REPORT_REQUIRED_PARAMS: Dict[str, Dict[str, Any]] = _build_report_required_params()


def _build_metric_required_params() -> Dict[str, Dict[str, List[str]]]:
    """Auto-generate per-metric required-param registry from param_registry."""
    result: Dict[str, Dict[str, List[str]]] = {}
    for metric, param_names in METRIC_PARAMS.items():
        params_list: List[str] = []
        top_level_list: List[str] = []
        for p in param_names:
            defn = PARAM_DEFINITIONS.get(p, {})
            if not defn.get("required", False) or defn.get("common", False):
                continue
            if p in _MCP_TOP_LEVEL_PARAMS:
                top_level_list.append(p)
            else:
                params_list.append(p)
        if params_list or top_level_list:
            entry: Dict[str, List[str]] = {}
            if params_list:
                entry["params"] = params_list
            if top_level_list:
                entry["top_level"] = top_level_list
            result[metric] = entry
    return result


METRIC_REQUIRED_PARAMS: Dict[str, Dict[str, List[str]]] = _build_metric_required_params()


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_metric_params(
    metric: str,
    params: Optional[Dict],
    table_names: Optional[List[str]] = None,
    primary_key: Optional[str] = None,
    foreign_key: Optional[str] = None,
) -> Optional[Dict]:
    """Return an error dict if required params are missing, else ``None``."""
    required = METRIC_REQUIRED_PARAMS.get(metric)
    if not required:
        return None

    missing: List[Dict] = []

    for key in required.get("params", []):
        value = (params or {}).get(key)
        if not value:
            desc = EVALUATION_HELPER.get(metric, {}).get("parameters", {}).get(key, "")
            missing.append({"parameter": key, "location": "params", "description": desc})

    top_level_values = {
        "table_names": table_names,
        "primary_key": primary_key,
        "foreign_key": foreign_key,
    }
    for key in required.get("top_level", []):
        if not top_level_values.get(key):
            desc = EVALUATION_HELPER.get(metric, {}).get("parameters", {}).get(key, "")
            missing.append({"parameter": key, "location": "top-level argument", "description": desc})

    if missing:
        return {
            "error": "MISSING_REQUIRED_PARAMETERS",
            "metric": metric,
            "message": (
                f"The metric '{metric}' requires parameters that were not "
                f"provided. ASK THE USER for the following values before retrying:"
            ),
            "missing_parameters": missing,
        }
    return None


def validate_report_params(
    report: str,
    params: Optional[Dict],
    table_names: Optional[List[str]] = None,
    primary_key: Optional[str] = None,
    foreign_key: Optional[str] = None,
) -> Optional[Dict]:
    
    """Return an error dict if a report's key required params are missing, else ``None``."""

    required = REPORT_REQUIRED_PARAMS.get(report)
    if not required:
        return None

    missing: List[Dict] = []
    for key in required.get("params", []):
        value = (params or {}).get(key)
        if not value:
            desc = (
                EVALUATION_REPORT_HELPER
                .get(report, {})
                .get("parameters", {})
                .get(key, "")
            )
            missing.append({"parameter": key, "location": "params", "description": desc})

    if missing:
        return {
            "error": "MISSING_REQUIRED_PARAMETERS",
            "report": report,
            "message": (
                f"The report '{report}' contains metrics that require "
                f"parameters not provided. ASK THE USER for the following "
                f"values before retrying:"
            ),
            "hint": required.get("description", ""),
            "missing_parameters": missing,
        }
    return None


def validate_ml_eval_params(target: str, task_type: str, train_source) -> Optional[Dict]:

    """Return an error dict if ml_eval required params are missing/invalid, else ``None``."""
    
    missing: List[Dict] = []

    if not target or not target.strip():
        missing.append({
            "parameter": "target",
            "type": "str",
            "description": (
                "The name of the column to predict. ASK THE USER which "
                "column from their dataset should be the prediction target."
            ),
        })

    valid_task_types = {"classification", "regression"}
    if task_type and task_type.strip() and task_type.strip().lower() not in valid_task_types:
        missing.append({
            "parameter": "task_type",
            "type": "str",
            "description": (
                f"Must be 'classification' or 'regression'. Got: '{task_type}'. "
                "ASK THE USER whether they want classification or regression."
            ),
        })

    valid_train_source ={"synthetic", "real"}

    if train_source and train_source.strip() and train_source.strip().lower() not in valid_train_source:

        missing.append({
            "parameter": "train_source",
            "type": "str",
            "description":(
                f"Must be 'real' or 'synthetic'. Got: {train_source}."
                "Ask the user weather they want to train the models on real or synthetic dataset"
            ),
        })

    if missing:
        return {
            "error": "MISSING_REQUIRED_PARAMETERS",
            "tool": "ml_eval",
            "message": (
                "ML evaluation requires parameters that were not provided or "
                "are invalid. ASK THE USER for the following values before retrying:"
            ),
            "missing_parameters": missing,
        }
    return None

