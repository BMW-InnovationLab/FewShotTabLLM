import glob
import json
import logging
import os
import re
import shutil

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import pandas as pd

from services.parsers import ml_eval_parse_args

# Use the canonical evaluator for ML evaluation.
# Falls back to the legacy data_handler path if the evaluator is not installed.
try:
    from synthetic_data_evaluator.ml_eval.ml_eval import run_ml_eval as _evaluator_ml_eval

    _USE_EVALUATOR = True
except ImportError:
    _USE_EVALUATOR = False

logger = logging.getLogger(__name__)


def _extract_k_shot(synth_name: str):
    """
    Extract the k-shot value from a synthetic dataset name.

    Matches patterns like ``synthetic_llm_40_shots`` -> 40.
    Returns ``None`` if the name does not follow this convention.
    """
    match = re.search(r"(\d+)_shots?$", synth_name)
    return int(match.group(1)) if match else None


def _safe_name(experiment: str) -> str:
    """Sanitize an experiment name for use as a filename component.

    Replaces path separators (``/``) with underscores so that names like
    ``qwen3.5-9b/adult-langfuse-e2e`` become ``qwen3.5-9b_adult-langfuse-e2e``.
    """
    return experiment.replace("/", "_").replace("\\", "_")


def _find_trace_id(experiment: str, k_shot):
    """
    Attempt to load a Langfuse trace ID from a generation config file.

    Looks for ``experiments/{experiment}/config_k{k_shot}.json``
    when *k_shot* is known, otherwise scans for any ``config_k*.json`` in the
    experiment directory.

    Returns the trace ID string or ``None``.
    """
    if k_shot is not None:
        config_path = f"experiments/{experiment}/config_k{k_shot}.json"
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    return json.load(f).get("last_trace_id")
            except Exception:
                return None

    # Fallback: scan for any config_k*.json in the experiment dir
    exp_dir = f"experiments/{experiment}"
    if not os.path.isdir(exp_dir):
        return None

    for cfg_file in sorted(glob.glob(os.path.join(exp_dir, "config_k*.json"))):
        try:
            with open(cfg_file) as f:
                tid = json.load(f).get("last_trace_id")
                if tid:
                    return tid
        except Exception:
            continue
    return None


def _push_scores_to_langfuse(trace_id: str, scores: dict, experiment: str, synth_name: str):
    """
    Push evaluation scores to a Langfuse trace using LangfuseManager.

    Fail-safe: if Langfuse is not installed or not configured, this is a no-op.
    """
    try:
        from flash_tabgen_tensorrt.core.langfuse_utils import LangfuseManager
    except ImportError:
        logger.debug("[Langfuse] langfuse_utils not importable — score push skipped")
        return

    lf = LangfuseManager(
        enabled=True,
        session_id=experiment,
        tags=["ml_eval"],
        metadata={"synth_name": synth_name, "experiment": experiment},
    )

    if not lf.init():
        return

    comment = f"ml_eval: {experiment}/{synth_name}"
    for metric_name, value in scores.items():
        lf.score_trace_by_id(
            trace_id=trace_id,
            name=metric_name,
            value=float(value),
            comment=comment,
        )

    lf.flush()
    logger.info(f"[Langfuse] Pushed {len(scores)} score(s) to trace {trace_id}")


def _save_results_csv(scores: dict, save_dir: str, filename: str):
    """Write a Model/Score CSV from a scores dict."""
    os.makedirs(save_dir, exist_ok=True)
    rows = [(name, score) for name, score in scores.items()]
    df = pd.DataFrame(rows, columns=["Model", "Score"])
    path = os.path.join(save_dir, filename)
    df.to_csv(path, index=False)
    print(f"Saving to {path}")


def run(
    dataset_path: str,
    experiments: list,
    synthetic_datasets: list,
    target_column: str,
    task_type: str = "classification",
    train_size=5000,
    test_size=1000,
    id_column: str = None,
    date_columns: list = None,
    disable_langfuse: bool = False,
):
    """
    Run ML evaluation on synthetic datasets.

    Uses the canonical ``synthetic_data_evaluator`` package for ML evaluation
    when available, falling back to the legacy ``services/data_handler`` path
    otherwise.

    Args:
        dataset_path: Path to the real dataset CSV file
        experiments: List of experiment names
        synthetic_datasets: List of synthetic dataset names to evaluate
                          e.g., ["real", "ctgan", "tvae", "synthetic_llm_40_shots"]
        target_column: Name of the target column
        task_type: "classification" or "regression"
        train_size: Number of training samples
        test_size: Number of test samples
        id_column: ID column to drop (optional)
        date_columns: List of date columns to process (optional)
        disable_langfuse: If True, skip pushing scores to Langfuse traces.
    """
    if date_columns is None:
        date_columns = []

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    os.makedirs("ML_EVALUATIONS/datasets", exist_ok=True)
    os.makedirs("ML_EVALUATIONS/results", exist_ok=True)
    os.makedirs(f"ML_EVALUATIONS/datasets/{dataset_name}", exist_ok=True)

    # Evaluate each experiment-dataset combination
    for experiment in experiments:
        safe_exp = _safe_name(experiment)

        for synth_name in synthetic_datasets:
            print(f"Evaluating: {experiment}/{synth_name}")

            # Determine synthetic data path
            if synth_name == "real":
                synth_csv_path = None
                train_source = "real"
            else:
                source_path = (
                    f"experiments/{experiment}/datasets/synthetic/{synth_name}.csv"
                )
                if not os.path.exists(source_path):
                    print(f"Warning: Could not find {source_path} — skipping")
                    continue
                synth_csv_path = source_path
                train_source = "synthetic"

            # ── Use canonical evaluator ──────────────────────────────
            if _USE_EVALUATOR:
                try:
                    result = _evaluator_ml_eval(
                        real_path=dataset_path,
                        synthetic_path=synth_csv_path,
                        target_column=target_column,
                        task_type=task_type,
                        train_source=train_source,
                        train_size=train_size,
                        test_size=test_size,
                        id_column=id_column,
                        date_columns=date_columns if date_columns else None,
                    )
                    scores = result.get("scores", {})
                except Exception as e:
                    print(f"Warning: Evaluation failed for {experiment}/{synth_name}: {e}")
                    continue

            # ── Legacy fallback ──────────────────────────────────────
            else:
                logger.warning(
                    "synthetic_data_evaluator not installed — using legacy "
                    "data_handler. Install the evaluator for correct results."
                )
                from services.data_handler import (
                    split_real_dataset,
                    load_dataset,
                    process_data_for_ml,
                    concat_df,
                    encode_datasets,
                    split_datasets,
                    encode_y,
                    convert_to_np,
                    init_models,
                    train_predict,
                )

                # Copy synthetic CSVs for legacy path
                if synth_csv_path:
                    dest_path = (
                        f"ML_EVALUATIONS/datasets/{dataset_name}/"
                        f"{safe_exp}_{synth_name}.csv"
                    )
                    shutil.copy(synth_csv_path, dest_path)

                data = pd.read_csv(dataset_path)
                train_df, test_df = split_real_dataset(
                    data, dataset_name, train_size, test_size,
                )

                synthesizer_name = (
                    f"{safe_exp}_{synth_name}" if synth_name != "real" else "real"
                )
                try:
                    data, test, metadata, _, _ = load_dataset(
                        dataset_name, synthesizer_name, "ML_EVALUATIONS",
                    )
                except FileNotFoundError as e:
                    print(
                        f"Warning: Skipping {experiment}/{synth_name} "
                        f"- file not found: {e}"
                    )
                    continue

                data = data[:train_size]
                test = test[:test_size]
                data, test, y, y_test = process_data_for_ml(
                    data, test, dataset_name, metadata,
                    target_column, date_columns, id_column,
                )

                results_df = concat_df(data, test)
                results_df = encode_datasets(results_df, target_column)
                results_df.fillna(0, inplace=True)

                data, test, y, y_test = split_datasets(
                    results_df, y, y_test, target_column, train_size,
                )
                y, y_test = encode_y(
                    y, y_test, dataset_name, metadata, target_column,
                )
                X_train, X_test, y_train, y_test = convert_to_np(
                    data, test, y, y_test,
                )

                models = init_models(task_type)
                train_predict(
                    X_train, X_test, y_train, y_test,
                    target_column, models, dataset_name,
                    synthesizer_name, task_type, "ML_EVALUATIONS", metadata,
                )
                # Legacy path saves its own CSV — skip further result handling
                continue

            # ── Save results CSV (evaluator path) ────────────────────
            synthesizer_name = (
                f"{safe_exp}_{synth_name}" if synth_name != "real" else "real"
            )
            save_dir = f"ML_EVALUATIONS/results/{dataset_name}"
            _save_results_csv(scores, save_dir, f"{synthesizer_name}.csv")

            # ── Langfuse push-back ───────────────────────────────────
            if not disable_langfuse and synth_name != "real":
                k_shot = _extract_k_shot(synth_name)
                trace_id = _find_trace_id(experiment, k_shot)

                if trace_id:
                    langfuse_scores = {
                        f"ml_eval/{name}": float(val)
                        for name, val in scores.items()
                        if isinstance(val, (int, float))
                    }
                    if langfuse_scores:
                        _push_scores_to_langfuse(
                            trace_id, langfuse_scores, experiment, synth_name,
                        )
                else:
                    logger.debug(
                        f"[Langfuse] No trace ID found for {experiment}/{synth_name} "
                        f"(k_shot={k_shot}) — score push skipped"
                    )

    # Copy results back to experiments
    for experiment in experiments:
        safe_exp = _safe_name(experiment)
        for synth_name in synthetic_datasets:
            if synth_name == "real":
                source_result = f"ML_EVALUATIONS/results/{dataset_name}/real.csv"
            else:
                source_result = (
                    f"ML_EVALUATIONS/results/{dataset_name}/"
                    f"{safe_exp}_{synth_name}.csv"
                )

            if os.path.exists(source_result):
                dest_result = (
                    f"experiments/{experiment}/evaluation_reports/{synth_name}.csv"
                )
                os.makedirs(os.path.dirname(dest_result), exist_ok=True)
                shutil.copy(source_result, dest_result)
                print(f"Copied results to {dest_result}")
            else:
                print(f"Warning: Results file not found: {source_result}")


if __name__ == "__main__":
    args = ml_eval_parse_args()
    run(
        dataset_path=args.dataset,
        experiments=args.experiments,
        synthetic_datasets=args.synthetic_datasets,
        target_column=args.target_column,
        task_type=args.task_type,
        train_size=args.train_size,
        test_size=args.test_size,
        id_column=args.id_column,
        date_columns=args.date_columns,
        disable_langfuse=args.disable_langfuse,
    )
