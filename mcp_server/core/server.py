"""
MCP tool definitions for synthetic data evaluator

Synthetic Data Evaluator provides different ways to evaluate tabular synthetic data.
It exposes tools for running individual metrics, grouped report evaluations, and
machine learning tests that train models on synthetic data and score them
on real data.  All evaluation tools accept CSV file paths and return structured
JSON results.
"""

import logging
from typing import Any, Dict, List, Optional, Literal

from fastmcp import FastMCP
from fastmcp.server.context import Context
from fastmcp.dependencies import CurrentContext

from .schemas import success, failure

logger = logging.getLogger("mcp_server")
from .tool_descriptions import (
    LIST_METRICS_DESC,
    LIST_REPORTS_DESC,
    GET_METRIC_PARAMS_DESC,
    GET_REPORT_PARAMS_DESC,
    LIST_ML_MODELS_DESC,
    GET_ML_EVAL_PARAMS_DESC,
    GET_METADATA_DESC,
    EVALUATE_METRIC_DESC,
    EVALUATE_REPORT_DESC,
    ML_EVAL_DESC,
    INSPECT_DATA_DESC,
    GET_TRAIN_PARAMS_DESC,
    GET_GENERATE_PARAMS_DESC,
    TRAIN_SYNTHESIZER_DESC,
    GENERATE_SYNTHETIC_DATA_DESC,
    LIST_TRAINED_MODELS_DESC,
    PLOT_DISTRIBUTIONS_DESC,
)
from . import services
from . import synth_services

mcp = FastMCP("synthetic-data-evaluator",
              instructions="""
You are an assistant that evaluates the quality of tabular synthetic data and can also
train CTGAN synthesizer models to generate synthetic data.

General rules:
- Always use absolute file paths or paths relative to the working directory when supplying real_data_path or synth_data_path.
- Never infer, fabricate, or guess parameter values. If a required value is missing, ask the user before proceeding.
- Metric and report names are case-sensitive. There are exactly 6 report names: Similarity, Privacy, Nearest Neighbor, Machine Learning Efficacy, Likelihood Fitness, Distance, Correlation. Everything else the user asks to run is a metric — default to list_evaluation_metrics when unsure.
- Data is always a needed parameter, dont run an evaluation without it.

Single-metric evaluation:
- If you already know the metric name and all required parameters, call evaluate_using_single_metric directly.
- If you are unsure of the exact metric name, call list_evaluation_metrics first.
- If you are unsure what parameters a metric requires, call get_evaluation_metric_parameters before evaluating.

Report evaluation:
- Only use report tools when the user explicitly mentions one of the 6 report names: Similarity, Privacy, Nearest Neighbor, Machine Learning Efficacy, Likelihood Fitness, Distance, Correlation.
- If the user's request does NOT match one of these 6 names, treat it as a metric — do NOT call list_evaluation_reports.
- If you already know the report name and all required parameters, call evaluate_using_report directly.
- If you are unsure what parameters a report requires, call get_report_parameters before evaluating.

ML evaluation:
- Requires target (column to predict), task_type ('classification' or 'regression'), and train_source ('real' or 'synthetic').
- ALWAYS ask the user for all three before calling ml_eval. Do NOT assume or default train_source to 'synthetic'.
- If train_source is 'synthetic', also ensure synth_data_path is provided.

Synthesizer — Training:
- Requires real_data_path (CSV to train on) and model_name (name for the saved model).
- All CTGAN hyper-parameters (epochs, batch_size, learning rates, architecture dims, etc.) are optional with sensible defaults.
- Ask the user for real_data_path and model_name before calling train_synthesizer.
-Call get_train_parameters first so the user can see available paramters
-Only CTGAN synthesizer is avilable for now

Synthesizer — Generation:
- Requires model_name (a previously trained model) and num_rows (how many rows to generate).
- If you are unsure which models exist, call list_trained_models first.
- Ask the user for model_name and num_rows before calling generate_synthetic_data.

Visualization:
- plot_distributions compares real vs synthetic data visually with per-column distribution plots.
- Requires real_data_path and synth_data_path.
- Optionally accepts specific columns, output_dir, return_base64, and max_columns.
- Suggest this tool after generating synthetic data or running evaluations so the user can visually inspect quality.
""")


# ── Listing ───────────────────────────────────────────────────────────────

@mcp.tool(description=LIST_METRICS_DESC)
async def list_evaluation_metrics(ctx: Context = CurrentContext()) -> Dict[str, Any]:

    """Lists available metrics"""
    try:

        await ctx.info("Fetching available evaluation metrics...")
        result = success(services.list_metrics())

        await ctx.info("✅ Metrics list retrieved successfully.")
        return result
    except Exception as e:

        await ctx.error(f"❌ Failed to list metrics: {e}")
        return failure(str(e))


@mcp.tool(description=LIST_REPORTS_DESC)
async def list_evaluation_reports(ctx: Context = CurrentContext()) -> Dict[str, Any]:

    """
    Lists available report categories and available metrics for each category.
    Example of report categories: Similarity, Distance, Privacy etc...
    """

    try:

        await ctx.info("Fetching available evaluation reports...")
        result = success(services.list_reports())

        await ctx.info("✅ Reports list retrieved successfully.")
        return result
    except Exception as e:

        await ctx.error(f"❌ Failed to list reports: {e}")
        return failure(str(e))


@mcp.tool(description=LIST_ML_MODELS_DESC)

async def list_ml_evaluation_models(ctx: Context = CurrentContext()) -> Dict[str, Any]:

    """Displays what models machine learning evaluation uses"""
    try:

        await ctx.info("Fetching available ML models...")
        result = success(services.list_ml_models())

        await ctx.info("✅ ML models list retrieved successfully.")
        return result
    except Exception as e:

        await ctx.error(f"❌ Failed to list ML models: {e}")
        return failure(str(e))


# ── Parameter introspection ──────────────────────────────────────────────

@mcp.tool(description=GET_METRIC_PARAMS_DESC)
async def get_evaluation_metric_parameters(evaluation_metric: str, ctx: Context = CurrentContext()) -> Dict[str, Any]:
    """Gets parameters for a specific metric"""
    try:

        await ctx.info(f"Fetching parameters for metric '{evaluation_metric}'...")
        result = success(services.get_metric_parameters(evaluation_metric))

        await ctx.info(f"✅ Parameters for '{evaluation_metric}' retrieved.")
        return result
    except Exception as e:

        await ctx.error(f"❌ Failed to get parameters for '{evaluation_metric}': {e}")
        return failure(str(e))


@mcp.tool(description=GET_REPORT_PARAMS_DESC)
async def get_report_parameters(evaluation_report: str, ctx: Context = CurrentContext()) -> Dict[str, Any]:
    """Gets parameters for specific report category """
    try:

        await ctx.info(f"Fetching parameters for report '{evaluation_report}'...")
        result = success(services.get_report_parameters(evaluation_report))
        await ctx.info(f"✅ Parameters for '{evaluation_report}' retrieved.")
        return result
    except Exception as e:
        await ctx.error(f"❌ Failed to get parameters for '{evaluation_report}': {e}")
        return failure(str(e))


@mcp.tool(description=GET_ML_EVAL_PARAMS_DESC)

async def get_ml_evaluation_parameters(ctx: Context = CurrentContext()) -> Dict[str, Any]:

    """Gets parameters needed to run machine learning evaluation"""
    try:
        await ctx.info("Fetching ML evaluation parameters...")
        result = success(services.get_ml_eval_parameters())
        await ctx.info("✅ ML evaluation parameters retrieved.")
        return result
    except Exception as e:
        await ctx.error(f"❌ Failed to get ML evaluation parameters: {e}")
        return failure(str(e))


# ── Metadata ─────────────────────────────────────────────────────────────

@mcp.tool(description=GET_METADATA_DESC)
async def get_dataset_metadata(real_data_paths: List[str], ctx: Context = CurrentContext()) -> Dict[str, Any]:
    """Gets the metadata for a dataset"""
    try:
        await ctx.info(f"Generating metadata for {len(real_data_paths)} dataset(s)...")
        result = success(services.get_metadata(real_data_paths))
        await ctx.info("✅ Metadata generated successfully.")
        return result
    except Exception as e:
        await ctx.error(f"❌ Failed to generate metadata: {e}")
        return failure(str(e))


# ── Evaluation ───────────────────────────────────────────────────────────

@mcp.tool(description=EVALUATE_METRIC_DESC)
async def evaluate_using_single_metric(
    metric: str,
    real_data_path: str,
    synth_data_path: str,
    params: Optional[Dict],
    metadata: Optional[Dict] = None,
    columns: Optional[List[str]] = None,
    table_names: Optional[List[str]] = None,
    primary_key: Optional[str] = None,
    foreign_key: Optional[str] = None,
    ctx: Context = CurrentContext(),
) -> Dict[str, Any]:
    
    """Runs a single evaluation metric on the data"""

    try:
        await ctx.info(f"⏳ Starting evaluation with metric '{metric}'...")
        result = services.run_single_metric(
            metric=metric,
            real_data_path=real_data_path,
            synth_data_path=synth_data_path,
            params=params,
            metadata=metadata,
            columns=columns,
            table_names=table_names,
            primary_key=primary_key,
            foreign_key=foreign_key,
        )
        await ctx.info(f"✅ Metric '{metric}' evaluation completed.")
        return success(result)
    except Exception as e:
        await ctx.error(f"❌ Metric '{metric}' evaluation failed: {e}")
        return failure(str(e))


@mcp.tool(description=EVALUATE_REPORT_DESC)

async def evaluate_using_report(
    report: str,
    real_data_path: str,
    synth_data_path: str,
    metadata: Optional[Dict],
    params: Optional[Dict],
    columns: Optional[List[str]] = None,
    table_names: Optional[List[str]] = None,
    primary_key: Optional[str] = None,
    foreign_key: Optional[str] = None,
    ctx: Context = CurrentContext(),
) -> Dict[str, Any]:
    
    """Runs report evaluation on data, reports are categorized and include certain metrics for each category """

    try:

        await ctx.info(f"⏳ Starting '{report}' report evaluation...")
        result = services.run_report(
            report=report,
            real_data_path=real_data_path,
            synth_data_path=synth_data_path,
            params=params,
            metadata=metadata,
            columns=columns,
            table_names=table_names,
            primary_key=primary_key,
            foreign_key=foreign_key,
        )
        await ctx.info(f"✅ '{report}' report evaluation completed.")
        return success(result)
    except Exception as e:
        await ctx.error(f"❌ '{report}' report evaluation failed: {e}")
        return failure(str(e))


# ── ML Evaluation ────────────────────────────────────────────────────────

@mcp.tool(description=ML_EVAL_DESC)

async def ml_eval(
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
    max_iter: int = 200,
    ctx: Context = CurrentContext(),
) -> Dict[str, Any]:
    
    """Trains a list of machine learning models on synthetic data (either regression or classification models), and tests them on the real data"""


    try:
        await ctx.info(f"⏳ Starting ML evaluation — task: {task_type}, target: '{target}', train on: {train_source} data...")
        result = services.run_ml(
        real_data_path=real_data_path,
        synth_data_path=synth_data_path,
        target=target,
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
        max_iter = max_iter
        )
        await ctx.info(f"✅ ML evaluation completed — target: '{target}', task: {task_type}.")
        return success(result)
    except Exception as e:

        await ctx.error(f"❌ ML evaluation failed: {e}")
        return failure(str(e))


@mcp.tool(description=INSPECT_DATA_DESC)

async def inspect_data(real_data_path: str, target: Optional[str], ctx: Context = CurrentContext()):

    """Enrich agent context with information about data"""

    try:
        await ctx.info(f"⏳ Inspecting dataset: {real_data_path}...")
        report = services.inspect_data(real_data_path=real_data_path,
                                        target=target)
        await ctx.info("✅ Dataset inspection completed.")
        return success(report)
    
    except Exception as e:
        await ctx.error(f"❌ Dataset inspection failed: {e}")
        return failure(str(e))



# ── Synthesizer: Introspection ────────────────────────────────────────────

@mcp.tool(description=GET_TRAIN_PARAMS_DESC)
async def get_train_parameters(ctx: Context = CurrentContext()) -> Dict[str, Any]:
    """Gets the parameter schema for training a CTGAN synthesizer."""
    try:
        await ctx.info("Fetching CTGAN training parameters...")
        result = success(synth_services.get_train_parameters())
        await ctx.info("✅ Training parameters retrieved.")
        return result
    except Exception as e:
        await ctx.error(f"❌ Failed to get training parameters: {e}")
        return failure(str(e))


@mcp.tool(description=GET_GENERATE_PARAMS_DESC)
async def get_generate_parameters(ctx: Context = CurrentContext()) -> Dict[str, Any]:
    """Gets the parameter schema for generating synthetic data."""
    try:
        await ctx.info("Fetching generation parameters...")
        result = success(synth_services.get_generate_parameters())
        await ctx.info("✅ Generation parameters retrieved.")
        return result
    except Exception as e:
        await ctx.error(f"❌ Failed to get generation parameters: {e}")
        return failure(str(e))


@mcp.tool(description=LIST_TRAINED_MODELS_DESC)
async def list_trained_models(models_dir: Optional[str] = None, ctx: Context = CurrentContext()) -> Dict[str, Any]:
    """Lists all trained synthesizer models on disk."""
    try:
        await ctx.info("Scanning for trained models...")
        result = success(synth_services.list_trained_models(models_dir))
        await ctx.info("✅ Trained models list retrieved.")
        return result
    except Exception as e:
        await ctx.error(f"❌ Failed to list trained models: {e}")
        return failure(str(e))


# ── Synthesizer: Train & Generate ─────────────────────────────────────────

@mcp.tool(description=TRAIN_SYNTHESIZER_DESC)
async def train_synthesizer(
    real_data_path: str,
    model_name: str,
    models_dir: Optional[str] = None,
    # ── data-processing ───────────────────────────────────
    enforce_min_max_values: bool = True,
    enforce_rounding: bool = True,
    locales: Optional[List[str]] = None,
    # ── architecture ──────────────────────────────────────
    embedding_dim: int = 128,
    generator_dim: Optional[List[int]] = None,
    discriminator_dim: Optional[List[int]] = None,
    # ── optimiser ─────────────────────────────────────────
    generator_lr: float = 2e-4,
    generator_decay: float = 1e-6,
    discriminator_lr: float = 2e-4,
    discriminator_decay: float = 1e-6,
    # ── training ──────────────────────────────────────────
    batch_size: int = 500,
    discriminator_steps: int = 1,
    log_frequency: bool = True,
    verbose: bool = False,
    epochs: int = 300,
    pac: int = 10,
    cuda: bool = True,
    ctx: Context = CurrentContext(),
) -> Dict[str, Any]:
    """Train a CTGAN model on a real CSV dataset."""
    try:
        await ctx.info(f"⏳ Training CTGAN model '{model_name}' for {epochs} epochs (batch_size={batch_size})...")
        result = synth_services.train_model(
            real_data_path=real_data_path,
            model_name=model_name,
            models_dir=models_dir,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            generator_decay=generator_decay,
            discriminator_lr=discriminator_lr,
            discriminator_decay=discriminator_decay,
            batch_size=batch_size,
            discriminator_steps=discriminator_steps,
            log_frequency=log_frequency,
            verbose=verbose,
            epochs=epochs,
            pac=pac,
            cuda=cuda,
        )
        await ctx.report_progress(progress=70, total = 100)
        await ctx.info(f"✅ Model '{model_name}' trained successfully.")
        return success(result)
    except Exception as e:
        error_msg = str(e) or f"{type(e).__name__}: {repr(e)}"
        await ctx.error(f"❌ Training model '{model_name}' failed: {error_msg}")
        return failure(error_msg)


@mcp.tool(description=GENERATE_SYNTHETIC_DATA_DESC)
async def generate_synthetic_data(
    model_name: str,
    num_rows: int,
    output_filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    ctx: Context = CurrentContext(),
) -> Dict[str, Any]:
    """Generate synthetic data from a trained CTGAN model."""
    try:
        await ctx.info(f"⏳ Generating {num_rows} synthetic rows from model '{model_name}'...")
        result = synth_services.generate_data(
            model_name=model_name,
            num_rows=num_rows,
            output_filename=output_filename,
            output_dir=output_dir,
            models_dir=models_dir,
        )
        await ctx.info(f"✅ {num_rows} synthetic rows generated from '{model_name}'.")
        return success(result)
    except Exception as e:
        await ctx.error(f"❌ Synthetic data generation failed: {e}")
        return failure(str(e))


# ── Visualization ─────────────────────────────────────────────────────────

@mcp.tool(description=PLOT_DISTRIBUTIONS_DESC)
async def plot_distributions(
    real_data_path: str,
    synth_data_path: str,
    columns: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    return_base64: bool = False,
    max_columns: int = 20,
    ctx: Context = CurrentContext(),
) -> Dict[str, Any]:
    """Plot per-column real vs synthetic distributions."""
    try:
        await ctx.info("⏳ Generating distribution comparison plots...")
        result = services.plot_distributions(
            real_data_path=real_data_path,
            synth_data_path=synth_data_path,
            columns=columns,
            output_dir=output_dir,
            return_base64=return_base64,
            max_columns=max_columns,
        )
        await ctx.info("✅ Distribution plots generated successfully.")
        return success(result)
    except Exception as e:
        await ctx.error(f"❌ Plot generation failed: {e}")
        return failure(str(e))

##TODO: add progress handler 