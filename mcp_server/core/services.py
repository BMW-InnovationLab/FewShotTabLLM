"""
Service layer — pure business logic for each MCP tool.

Each function takes validated, typed arguments and returns a plain dict.
No FastMCP types, no HTTP concepts, no tool descriptions here.
"""

from typing import Any, Dict, List, Optional, Literal

from synthetic_data_evaluator.core.evaluation_info import (
    EVALUATION_HELPER,
    EVALUATION_REPORT_HELPER,
)
from synthetic_data_evaluator.ml_eval.ml_info import (
    ML_EVALUATION_HELPER,
    ML_EVALUATION_MODELS,
)
from synthetic_data_evaluator.engine.evaluation_mapping import METRIC_CLASSES
from synthetic_data_evaluator.core.param_registry import REPORT_METRICS
from synthetic_data_evaluator.engine.evaluation_engine import (
    evaluate_single_metric,
    evaluate_report,
)
from synthetic_data_evaluator.ml_eval.ml_eval import run_ml_eval

from .loaders import load_and_prepare, build_metadata
from .validation import (
    METRIC_REQUIRED_PARAMS,
    REPORT_REQUIRED_PARAMS,
    validate_metric_params,
    validate_report_params,
    validate_ml_eval_params,
)

import os
import io
import base64

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Listing helpers ───────────────────────────────────────────────────────

def list_metrics() -> Dict[str, Any]:
    return {"metrics": sorted(METRIC_CLASSES.keys()), "total": len(METRIC_CLASSES)}


def list_reports() -> Dict[str, Any]:
    return {
        "reports": {name: list(metrics) for name, metrics in REPORT_METRICS.items()},
        "total": len(REPORT_METRICS),
    }


# ── Parameter introspection ──────────────────────────────────────────────

def get_metric_parameters(metric: str) -> Dict[str, Any]:
    info = EVALUATION_HELPER.get(metric)
    if info is None:
        raise ValueError(f"Unknown metric: {metric}")

    required = METRIC_REQUIRED_PARAMS.get(metric, {})
    required_keys = set(required.get("params", []) + required.get("top_level", []))
    params = info.get("parameters", {})

    if not required_keys:
        return {
            **info,
            "optional_parameters": params,
            "note": "All parameters are optional with sensible defaults.",
        }

    return {
        **{k: v for k, v in info.items() if k != "parameters"},
        "required_parameters": {k: params[k] for k in required_keys if k in params},
        "optional_parameters": {k: v for k, v in params.items() if k not in required_keys},
        "note": (
            "IMPORTANT: You MUST ask the user for required parameter "
            "values before running the evaluation."
        ),
    }


def get_report_parameters(report: str) -> Dict[str, Any]:
    info = EVALUATION_REPORT_HELPER.get(report)
    if info is None:
        raise ValueError(f"Unknown report: {report}")

    required = REPORT_REQUIRED_PARAMS.get(report, {})
    required_keys = set(required.get("params", []) + required.get("top_level", []))
    params = info.get("parameters", {})

    if not required_keys:
        return {
            **info,
            "optional_parameters": params,
            "note": "All parameters are optional with sensible defaults.",
        }

    return {
        **{k: v for k, v in info.items() if k != "parameters"},
        "required_parameters": {k: params[k] for k in required_keys if k in params},
        "optional_parameters": {k: v for k, v in params.items() if k not in required_keys},
        "note": (
            "IMPORTANT: You MUST ask the user for required parameter "
            "values before running the evaluation."
        ),
    }


def get_ml_eval_parameters() -> Dict[str, Any]:
    result = {**ML_EVALUATION_HELPER}
    result["note"] = (
        "IMPORTANT: 'target' and 'task_type' and 'train_source' are REQUIRED. "
        "You MUST ask the user which column to predict (target) and whether they want the 'train_source' to be 'real' or 'synthetic'"
        "If the user picks the 'train_source' as 'synthetic', then 'synthetic_data_path' becomes mandatory"
        "the task is 'classification' or 'regression'."
    )
    return result


def list_ml_models() -> Dict[str, Any]:
    return ML_EVALUATION_MODELS


# ── Metadata ─────────────────────────────────────────────────────────────

def get_metadata(real_data_paths: List[str]) -> Dict[str, Any]:
    return build_metadata(real_data_paths)


# ── Core evaluation ──────────────────────────────────────────────────────

def _extra_init(
    columns: Optional[List[str]] = None,
    table_names: Optional[List[str]] = None,
    primary_key: Optional[str] = None,
    foreign_key: Optional[str] = None,
) -> Dict[str, Any]:
    
    """Build the ``**extra_init`` kwargs dict from optional fields."""
    
    kw: Dict[str, Any] = {}
    if columns:
        kw["column_names"] = columns
    if table_names:
        kw["table_names"] = table_names
    if primary_key:
        kw["primary_key"] = primary_key
    if foreign_key:
        kw["foreign_key"] = foreign_key
    return kw


def run_single_metric(
    metric: str,
    real_data_path: str,
    synth_data_path: str,
    params: Optional[Dict],
    metadata: Optional[Dict] = None,
    columns: Optional[List[str]] = None,
    table_names: Optional[List[str]] = None,
    primary_key: Optional[str] = None,
    foreign_key: Optional[str] = None,
) -> Dict[str, Any]:
    
    """Validate, load, and evaluate a single metric."""

    validation_error = validate_metric_params(
        metric, params, table_names, primary_key, foreign_key,
    )
    if validation_error:
        return validation_error

    real, synth, meta, table_name, modality, params_obj = load_and_prepare(
        real_data_path, synth_data_path, params, metadata, table_names,
    )

    return evaluate_single_metric(
        metric, real, synth, meta, table_name, params_obj, modality,
        **_extra_init(columns, table_names, primary_key, foreign_key),
    )


def run_report(
    report: str,
    real_data_path: str,
    synth_data_path: Optional[str],
    params: Optional[Dict],
    metadata: Optional[Dict] = None,
    columns: Optional[List[str]] = None,
    table_names: Optional[List[str]] = None,
    primary_key: Optional[str] = None,
    foreign_key: Optional[str] = None,
) -> Dict[str, Any]:
    
    """Validate, load, and evaluate a full report."""

    validation_error = validate_report_params(
        report, params, table_names, primary_key, foreign_key,
    )
    if validation_error:
        return validation_error

    real, synth, meta, table_name, modality, params_obj = load_and_prepare(
        real_data_path, synth_data_path, params, metadata, table_names,
    )

    return evaluate_report(
        [report], real, synth, meta, table_name, params_obj, modality,
        **_extra_init(columns, table_names, primary_key, foreign_key),
    )


def run_ml(
    real_data_path: str,
    synth_data_path: Optional[str],
    target: str,
    task_type: Literal["classification","regression"] = "classification",
    train_source: Literal["synthetic", "real"] = "real",
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    id_column: Optional[str] = None,
    date_columns: Optional[List[str]] = None,
    test_split: float = 0.2,
    random_state: int = 42,
    learning_rate: float = 0.01,
    n_estimators: int = 100,
    max_depth: int = 10,
    C: float = 1.0,
    kernel: str = "rbf",
    max_iter: int = 200
) -> Dict[str, Any]:
    
    """Validate params, then delegate to the ML evaluator."""

    validation_error = validate_ml_eval_params(target, task_type, train_source)
    if validation_error:
        return validation_error

    return run_ml_eval(
        real_path=real_data_path,
        synthetic_path=synth_data_path,
        target_column=target,
        task_type=task_type,
        train_source =train_source,
        train_size=train_size,
        test_size=test_size,
        id_column=id_column,
        date_columns=date_columns,
        test_split=test_split,
        random_state = random_state,
        learning_rate = learning_rate,
        n_estimators = n_estimators,
        max_depth= max_depth,
        C = C,
        kernel = kernel,
        max_iter= max_iter,
    )

# ── Data inspection ───────────────────────────────────────────────────────

def inspect_data(real_data_path: str, target=None):
    """Enrich agent context infomation about dataset"""

    df = pd.read_csv(real_data_path)
    n_rows, n_cols = df.shape


    metadata = build_metadata([real_data_path])
    table_name = list(metadata["tables"].keys())[0]
    col_sdtypes = {
        col: info["sdtype"]
        for col, info in metadata["tables"][table_name]["columns"].items()
    }

    report: Dict[str, Any] = {}

    report["overview"] = {
        "rows": n_rows,
        "columns": n_cols,
        "memory_usage_mb": round(float(df.memory_usage(deep=True).sum() / (1024**2)), 3),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_values": int(df.isnull().sum().sum()),
        "total_missing_percent": round(float(df.isnull().sum().sum() / (n_rows * n_cols) * 100), 2),
    }
    columns_info = []

    for col in df.columns:
        col_data = df[col]
        sdtype = col_sdtypes.get(col, "categorical")
        missing_count = int(col_data.isnull().sum())

        info: Dict[str, Any] = {
            "column": col,
            "pandas_dtype": str(col_data.dtype),
            "sdtype": sdtype,
            "unique_values": int(col_data.nunique()),
            "missing_count": missing_count,
            "missing_percent": round(float(missing_count / n_rows * 100), 2),
        }

        if sdtype == "numerical":
            numeric_col = pd.to_numeric(col_data, errors="coerce")
            info.update({
                "mean": round(float(numeric_col.mean()), 4) if pd.notna(numeric_col.mean()) else None,
                "std": round(float(numeric_col.std()), 4) if pd.notna(numeric_col.std()) else None,
                "min": round(float(numeric_col.min()), 4) if pd.notna(numeric_col.min()) else None,
                "25%": round(float(numeric_col.quantile(0.25)), 4),
                "50%": round(float(numeric_col.quantile(0.50)), 4),
                "75%": round(float(numeric_col.quantile(0.75)), 4),
                "max": round(float(numeric_col.max()), 4) if pd.notna(numeric_col.max()) else None,
                "skew": round(float(numeric_col.skew()), 4) if pd.notna(numeric_col.skew()) else None,
            })

        elif sdtype == "boolean":
            vc = col_data.value_counts(normalize=True)
            info["value_distribution"] = {
                str(k): round(float(v), 4) for k, v in vc.items()
            }

        elif sdtype == "categorical":
            vc = col_data.value_counts()
            if col_data.nunique() <= 15:
                info["value_distribution"] = {
                    str(k): int(v) for k, v in vc.items()
                }
            else:
                info["top_10_values"] = {
                    str(k): int(v) for k, v in vc.head(10).items()
                }
            info["sample_values"] = [
                str(v) for v in col_data.dropna().unique()[:5]
            ]

        elif sdtype == "datetime":
            try:
                dt_col = pd.to_datetime(col_data, errors="coerce")
                info["min_date"] = str(dt_col.min())
                info["max_date"] = str(dt_col.max())
            except Exception:
                pass

        columns_info.append(info)

    report["columns"] = columns_info

    type_counts: Dict[str, int] = {}
    for c in columns_info:
        t = c["sdtype"]
        type_counts[t] = type_counts.get(t, 0) + 1
    report["type_summary"] = type_counts

    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        corr = numeric_df.corr()
        strong = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iloc[i, j]
                if abs(r) > 0.5:
                    strong.append({
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "correlation": round(float(r), 4),
                    })
        if strong:
            report["notable_correlations"] = strong

    # target analysis
    if target and target in df.columns:
        target_data = df[target]
        target_sdtype = col_sdtypes.get(target, "categorical")
        target_info: Dict[str, Any] = {"column": target}

        if target_sdtype in ("boolean", "categorical") or target_data.nunique() <= 20:
            target_info["task_suggestion"] = "classification"
            vc = target_data.value_counts(normalize=True)
            target_info["class_distribution"] = {
                str(k): round(float(v), 4) for k, v in vc.items()
            }
            target_info["n_classes"] = int(target_data.nunique())
            if vc.max() > 0.8:
                target_info["warning"] = "Highly imbalanced target — dominant class has >80% of rows."
        else:
            target_info["task_suggestion"] = "regression"
            numeric_target = pd.to_numeric(target_data, errors="coerce")
            target_info["stats"] = {
                "mean": round(float(numeric_target.mean()), 4) if pd.notna(numeric_target.mean()) else None,
                "std": round(float(numeric_target.std()), 4) if pd.notna(numeric_target.std()) else None,
                "min": round(float(numeric_target.min()), 4) if pd.notna(numeric_target.min()) else None,
                "max": round(float(numeric_target.max()), 4) if pd.notna(numeric_target.max()) else None,
                "skew": round(float(numeric_target.skew()), 4) if pd.notna(numeric_target.skew()) else None,
            }

        report["target_analysis"] = target_info

    return report


# ── Distribution plotting ─────────────────────────────────────────────────

_DEFAULT_PLOTS_DIR = os.path.join(os.getcwd(), "plots")


def plot_distributions(
    real_data_path: str,
    synth_data_path: str,
    columns: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    return_base64: bool = False,
    max_columns: int = 20,
) -> Dict[str, Any]:
    """
    Generate per-column distribution comparison plots (real vs synthetic).

    Numerical columns  → overlaid density histograms
    Categorical columns → grouped bar charts of value counts
    """

    if not os.path.isfile(real_data_path):
        raise FileNotFoundError(f"Real data not found: {real_data_path}")
    if not os.path.isfile(synth_data_path):
        raise FileNotFoundError(f"Synthetic data not found: {synth_data_path}")

    real = pd.read_csv(real_data_path)
    synth = pd.read_csv(synth_data_path)

    # ── pick columns ──────────────────────────────────────
    cols = columns or [c for c in real.columns if c in synth.columns]
    cols = cols[:max_columns]  # guard against huge column counts

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(real[c])]
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(real[c])]

    out_dir = output_dir or _DEFAULT_PLOTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    saved_paths: List[str] = []
    base64_images: Dict[str, str] = {}

    # ── numerical: density histograms ─────────────────────
    if num_cols:
        n = len(num_cols)
        n_rows = (n + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        axes = np.atleast_2d(axes)

        for i, col in enumerate(num_cols):
            ax = axes[i // 3, i % 3]
            ax.hist(
                real[col].dropna(), bins=40, alpha=0.55,
                density=True, label="Real", color="#2196F3",
            )
            ax.hist(
                synth[col].dropna(), bins=40, alpha=0.55,
                density=True, label="Synthetic", color="#FF9800",
            )
            ax.set_title(col, fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

        # hide unused subplots
        for j in range(n, n_rows * 3):
            axes[j // 3, j % 3].set_visible(False)

        fig.suptitle(
            "Numerical Distributions \u2014 Real vs Synthetic",
            fontsize=14, y=1.02,
        )
        fig.tight_layout()

        path = os.path.join(out_dir, "numerical_distributions.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved_paths.append(os.path.abspath(path))

        if return_base64:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            base64_images["numerical"] = base64.b64encode(buf.read()).decode()

        plt.close(fig)

    # ── categorical: grouped bar charts ───────────────────
    if cat_cols:
        n = len(cat_cols)
        n_rows = (n + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        axes = np.atleast_2d(axes)

        for i, col in enumerate(cat_cols):
            ax = axes[i // 3, i % 3]
            real_vc = real[col].value_counts(normalize=True).sort_index()
            synth_vc = synth[col].value_counts(normalize=True).sort_index()
            all_cats = sorted(set(real_vc.index) | set(synth_vc.index))

            x = np.arange(len(all_cats))
            w = 0.35
            ax.bar(
                x - w / 2,
                [real_vc.get(c, 0) for c in all_cats],
                w, label="Real", color="#2196F3", alpha=0.8,
            )
            ax.bar(
                x + w / 2,
                [synth_vc.get(c, 0) for c in all_cats],
                w, label="Synthetic", color="#FF9800", alpha=0.8,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(all_cats, rotation=45, ha="right", fontsize=7)
            ax.set_title(col, fontsize=11)
            ax.set_ylabel("Proportion")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2, axis="y")

        for j in range(n, n_rows * 3):
            axes[j // 3, j % 3].set_visible(False)

        fig.suptitle(
            "Categorical Distributions \u2014 Real vs Synthetic",
            fontsize=14, y=1.02,
        )
        fig.tight_layout()

        path = os.path.join(out_dir, "categorical_distributions.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved_paths.append(os.path.abspath(path))

        if return_base64:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            base64_images["categorical"] = base64.b64encode(buf.read()).decode()

        plt.close(fig)

    return {
        "plots_saved": saved_paths,
        "numerical_columns_plotted": num_cols,
        "categorical_columns_plotted": cat_cols,
        "output_dir": os.path.abspath(out_dir),
        "base64_images": base64_images if return_base64 else None,
        "status": f"Generated {len(saved_paths)} plot(s).",
    }

