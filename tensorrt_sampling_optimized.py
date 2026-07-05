from __future__ import annotations

import os
import subprocess
import re
import time
import json
import sys
import signal
import logging
import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Import both backends - will be selected at runtime
from services.sampling import select_representative_samples
from services.parsers import tensorrt_parse_args
from services.logger_config import setup_logging

os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"

logger = logging.getLogger(__name__)

# ==========================================
# HELPER FUNCTIONS
# ==========================================


def configure_mpi_network():
    """Auto-configure MPI network settings"""
    keys_to_unset = ["OMPI_MCA_btl_tcp_if_include", "OMPI_MCA_btl_tcp_if_exclude", "OMPI_MCA_btl"]
    for key in keys_to_unset:
        os.environ.pop(key, None)
    try:
        result = subprocess.check_output(["ip", "route", "get", "8.8.8.8"], text=True)
        match = re.search(r"dev\s+(\S+)", result)
        if match:
            primary_interface = match.group(1)
            os.environ["OMPI_MCA_btl_tcp_if_include"] = f"{primary_interface},lo"
            os.environ["NCCL_SOCKET_IFNAME"] = primary_interface
    except Exception:
        pass


configure_mpi_network()


def load_tensorrt_config(config_path: str = None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "tensorrt_config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def _load_tokenizer(model_name: str):
    """Load a tokenizer, working around Gemma-4 extra_special_tokens incompatibility.

    Some model configs (e.g. google/gemma-4-*) ship ``extra_special_tokens`` as a
    **list** while ``transformers < 5.0`` expects a **dict**.  We patch the config
    JSON on-the-fly so ``AutoTokenizer`` can load without crashing.
    """
    from transformers import AutoTokenizer
    import json as _json

    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except (AttributeError, TypeError) as exc:
        if "list" not in str(exc) and "extra_special_tokens" not in str(exc).lower():
            raise
        # Patch: download config, remove extra_special_tokens (not needed for token counting)
        from huggingface_hub import snapshot_download

        logger.info(f"[Tokenizer] Patching extra_special_tokens for {model_name}")
        local_dir = snapshot_download(model_name, allow_patterns=["tokenizer*", "special_tokens*"])
        for cfg_name in ("tokenizer_config.json", "special_tokens_map.json"):
            cfg_path = os.path.join(local_dir, cfg_name)
            if not os.path.exists(cfg_path):
                continue
            with open(cfg_path, "r") as f:
                cfg = _json.load(f)
            if isinstance(cfg.get("extra_special_tokens"), list):
                del cfg["extra_special_tokens"]
                with open(cfg_path, "w") as f:
                    _json.dump(cfg, f, indent=2)
        return AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)


def estimate_tokens_with_tokenizer(train_data, model_name, k_shot, rows_per_prompt):
    try:
        tokenizer = _load_tokenizer(model_name)

        # Sample data for estimation
        sample_size = min(100, len(train_data))
        sample_rows = train_data.sample(n=sample_size)

        # Convert rows to string representation to estimate tokens
        # We assume a verbose "Column: Value" format to be safe (over-estimate slightly)
        row_strings = []
        for _, row in sample_rows.iterrows():
            # Construct "Col is Val, Col is Val" string
            s = ", ".join([f"{col} is {val}" for col, val in row.items()])
            row_strings.append(s)

        # Get token counts for each sampled row
        # We use the tokenizer directly
        row_token_counts = [len(tokenizer.encode(s)) for s in row_strings]

        if not row_token_counts:
            return None, None

        # --- Output Tokens Calculation ---
        # User request: "check the original trainning data 25 rows on average how many token on max they take and add a buffer of 20 to 30 %"
        # We take chunks of `rows_per_prompt` and find the max token usage among them to be safe.
        chunk_sums = []
        for i in range(0, len(row_token_counts), rows_per_prompt):
            chunk = row_token_counts[i : i + rows_per_prompt]
            # Scale up partial chunks to normalize
            if len(chunk) > 0:
                chunk_sum = sum(chunk) * (rows_per_prompt / len(chunk))
                chunk_sums.append(chunk_sum)

        if chunk_sums:
            # Use the max chunk size we found + 30% buffer
            base_output_tokens = max(chunk_sums)
        else:
            # Fallback to average * rows
            avg_row = sum(row_token_counts) / len(row_token_counts)
            base_output_tokens = avg_row * rows_per_prompt

        estimated_output_len = int(base_output_tokens * 1.1)

        # --- Input Tokens Calculation ---
        # User request: "use the model tokenizer to check the total number of tokens of the input prompt with a small buffer of 20%"
        # Input ≈ Instructions (Overhead) + k-shot examples

        avg_row_tokens = sum(row_token_counts) / len(row_token_counts)
        instruction_overhead = 1000  # Safety buffer for system prompt/instructions

        input_raw = instruction_overhead + (k_shot * avg_row_tokens)
        estimated_input_len = int(input_raw * 1.1)

        return estimated_input_len, estimated_output_len

    except Exception as e:
        logger.warning(f"[Token Estimation] Warning: Failed to use tokenizer for estimation: {e}")
        return None, None


def _adapt_rows_per_prompt(
    batch_size: int,
    train_data,
    model_name: str,
    k_shot: int,
    batch_params: dict,
    safety_margin: float = 1.35,
    min_rows: int = 5,
) -> int:
    """Adapt rows-per-prompt so output fits within the token budget.

    For wide schemas (many columns), the default ``batch_size`` (25) may cause
    the LLM output to be truncated — the model generates 25 JSON objects but
    hits ``max_tokens`` before finishing, resulting in dropped incomplete rows.

    This function estimates the true per-row token cost from the training data
    and reduces ``batch_size`` if necessary.

    Args:
        batch_size: Current (user-specified or default) rows per prompt.
        train_data: Representative training DataFrame.
        model_name: HuggingFace model identifier (for tokenizer).
        k_shot: Number of few-shot examples (affects input size, not output).
        batch_params: Output of ``calculate_optimal_batch_params``.
        safety_margin: Multiply per-row tokens by this factor (>1) for safety.
        min_rows: Never go below this many rows per prompt.

    Returns:
        Adjusted ``batch_size`` (may be the same if no reduction needed).
    """
    try:
        tokenizer = _load_tokenizer(model_name)

        # Sample real rows and measure per-row token cost
        sample_size = min(200, len(train_data))
        sample_rows = train_data.sample(n=sample_size, random_state=42)

        row_tokens = []
        for _, row in sample_rows.iterrows():
            # JSON format: {"col1": "val1", "col2": "val2", ...}
            row_str = json.dumps({str(c): str(v) for c, v in row.items()})
            row_tokens.append(len(tokenizer.encode(row_str)))

        if not row_tokens:
            return batch_size

        # Use the 90th percentile (not max, not mean) to be robust to outliers
        row_tokens_sorted = sorted(row_tokens)
        p90_idx = int(0.9 * len(row_tokens_sorted))
        tokens_per_row_p90 = row_tokens_sorted[min(p90_idx, len(row_tokens_sorted) - 1)]

        # Available output budget
        output_budget = batch_params["estimated_output_tokens"]
        # For remote-vllm, the floor is applied later (8192 or 65536),
        # so use max of estimated and 8192 as a conservative budget
        effective_budget = max(output_budget, 8192)

        # How many rows fit safely?
        safe_tokens_per_row = int(tokens_per_row_p90 * safety_margin)
        max_rows = max(min_rows, effective_budget // safe_tokens_per_row)

        n_cols = len(train_data.columns)

        if max_rows < batch_size:
            logger.info(
                f"[Adaptive] Reducing rows_per_prompt: {batch_size} → {max_rows} "
                f"(cols={n_cols}, ~{tokens_per_row_p90} tok/row p90, "
                f"safe={safe_tokens_per_row}, budget={effective_budget})"
            )
            return max_rows
        else:
            logger.info(
                f"[Adaptive] rows_per_prompt={batch_size} OK "
                f"(cols={n_cols}, ~{tokens_per_row_p90} tok/row p90, "
                f"budget={effective_budget}, max_safe={max_rows})"
            )
            return batch_size

    except Exception as e:
        logger.warning(f"[Adaptive] Could not adapt rows_per_prompt: {e}")
        return batch_size


def calculate_optimal_batch_params(
    k_shot,
    rows_per_prompt,
    hardware="B200",
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
    train_data=None,
    config_path=None,
):
    config = load_tensorrt_config(config_path)

    # Try to estimate using the actual tokenizer and data first
    est_input, est_output = None, None
    if train_data is not None:
        logger.info(
            f"[Auto-Tuning] Estimating token usage using tokenizer for model: {model_name}..."
        )
        est_input, est_output = estimate_tokens_with_tokenizer(
            train_data, model_name, k_shot, rows_per_prompt
        )

    if est_input is not None and est_output is not None:
        logger.info(
            f"[Auto-Tuning] Calculated from data -> Input: {est_input}, Output: {est_output}"
        )
        estimated_input_tokens = est_input
        estimated_output_tokens = est_output
    else:
        logger.info(f"[Auto-Tuning] Fallback to heuristic estimation")
        # Fallback logic
        estimated_input_tokens = 500 + (k_shot * 80)
        base_output_tokens = rows_per_prompt * config["token_estimation"]["fallback_tokens_per_row"]
        buffer_config = config["token_buffers"]
        estimated_output_tokens = int(
            base_output_tokens * buffer_config["output_buffer_multiplier"]
            + buffer_config["output_buffer_additive"]
        )

    # Apply Hardware/Config Constraints
    max_input_len = 65536
    # Match input length to thresholds if defined
    for threshold_config in config["token_length_thresholds"]["input"]:
        threshold = threshold_config["threshold"]
        if threshold is None or estimated_input_tokens <= threshold:
            max_input_len = threshold_config["max_input_len"]
            break

    # Ensure calculated max_input_len respects the estimate (don't cut it off if possible)
    # But also respect the hard limits of the engine/hardware if strictly defined in config.
    # However, usually max_input_len in config is a "bucket" size.
    # If our estimate is larger than the bucket, we might need a larger bucket.
    # For now, we assume the logic "if threshold is None or estimated <= threshold" picks the right bucket.
    # If estimated > largest threshold, it picks the last one (if threshold is None).

    output_config = config["output_tokens"]
    # Clamp output length
    max_output_len = max(output_config["min_value"], estimated_output_tokens)
    max_output_len = min(max_output_len, output_config["max_value"])

    hw_config = config["hardware_configs"].get(hardware, config["hardware_configs"]["default"])
    max_batch_size = hw_config["max_batch_size"]

    if "concurrent_prompts_thresholds" in hw_config:
        concurrent_prompts = 8
        for threshold_config in hw_config["concurrent_prompts_thresholds"]:
            threshold = threshold_config["threshold"]
            if threshold is None or estimated_input_tokens <= threshold:
                concurrent_prompts = threshold_config["concurrent_prompts"]
                break
    else:
        concurrent_prompts = hw_config.get("concurrent_prompts", 8)

    return {
        "max_input_len": max_input_len,
        "max_output_len": max_output_len,
        "max_batch_size": max_batch_size,
        "concurrent_prompts": concurrent_prompts,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
    }


# ==========================================
# WORKER FUNCTION (Runs inside Child Process)
# ==========================================
def run_worker(
    experiment: str,
    train_data_path: str,
    target_column: str,
    k_shots: list[int],
    model="Qwen/Qwen3-Coder-Next-FP8",
    permute=False,
    use_correlation_matrix=True,
    batch_size=25,
    device: int = 0,
    n_rows: int = 5000,
    top_p=0.8,
    top_k=20,
    temperature=0.7,
    float_precision=4,
    min_p=0,
    conditionals=[],
    date_columns=[],
    type_overrides_raw=[],
    max_context_len=None,
    tensor_parallel=1,
    verbose=False,
    backend="vllm",
    max_input_len=16384,
    max_output_len=16384,
    # Remote vLLM options
    server_url="",
    api_key="",
    concurrent_requests=32,
    request_timeout=600.0,
    enable_thinking=False,
    # Observability
    disable_langfuse=False,
    # Penalty parameters
    presence_penalty=1.5,
    repetition_penalty=1.0,
    # Encoding format
    encoding_format="json",
):
    # Only set single GPU if not using tensor parallelism
    # For tensor_parallel > 1, rely on CUDA_VISIBLE_DEVICES from the shell

    if tensor_parallel == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    k_shot = k_shots[0]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments/{experiment}/logs/experiment_k{k_shot}_{timestamp}.log"
    setup_logging(log_file=log_file, verbose=verbose)
    logger.info(f"{'=' * 80}")
    logger.info(f"[Worker] Starting Experiment for k_shot={k_shot}")
    logger.info(f"{'=' * 80}")

    # Log all experiment parameters
    logger.info("Experiment Parameters:")
    logger.info(f"  Experiment Name: {experiment}")
    logger.info(f"  Dataset: {train_data_path}")
    logger.info(f"  Target Column: {target_column}")
    logger.info(f"  K-Shots: {k_shot}")
    logger.info(f"  Model: {model}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Device: {device}")
    logger.info(f"  N Rows: {n_rows}")
    logger.info(f"  Top P: {top_p}")
    logger.info(f"  Top K: {top_k}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Float Precision: {float_precision}")
    logger.info(f"  Min P: {min_p}")
    logger.info(f"  Presence Penalty: {presence_penalty}")
    logger.info(f"  Repetition Penalty: {repetition_penalty}")
    logger.info(f"  Encoding Format: {encoding_format}")
    logger.info(f"  Conditionals: {conditionals}")
    logger.info(f"  Date Columns: {date_columns}")
    logger.info(f"  Type Overrides: {type_overrides_raw}")
    logger.info(f"  Max Context Length: {max_context_len}")
    logger.info(f"  Tensor Parallel: {tensor_parallel} GPU(s)")
    logger.info(f"  Permute: {permute}")
    logger.info(f"  Use Correlation Matrix: {use_correlation_matrix}")
    logger.info(f"  Backend: {backend}")
    if backend == "remote-vllm":
        logger.info(f"  Server URL: {server_url}")
        logger.info(f"  Concurrent Requests: {concurrent_requests}")
        logger.info(f"  Request Timeout: {request_timeout}s")
        logger.info(f"  Enable Thinking: {enable_thinking}")
    logger.info(f"{'=' * 80}")

    try:
        train_df_filtered = select_representative_samples(
            input_file=train_data_path,
            output_file=f"experiments/{experiment}/datasets/real/examples_pool.csv",
            sample_size=None,
            feature_threshold=0.7,
            enable_visualization=False,
        )
        logger.info(f"Selected {len(train_df_filtered)} rows from the training data")
        logger.info(f"Training data sample: {train_df_filtered.head().to_string()}")

        initial_params = calculate_optimal_batch_params(
            k_shot, batch_size, hardware="B200", model_name=model, train_data=train_df_filtered
        )
        logger.info("Optimal Batch Params calculated:")
        logger.info(initial_params)

        # Calculate max_model_len, cap it if max_context_len is specified
        calculated_len = initial_params["max_input_len"] + initial_params["max_output_len"]
        if max_context_len:
            actual_max_len = min(calculated_len, max_context_len)
            logger.info(f"Capping max_model_len from {calculated_len} to {actual_max_len}")
        else:
            actual_max_len = calculated_len

        # Initialize the appropriate backend
        if backend == "vllm":
            from flash_tabgen_tensorrt.core.tabgen_vllm import TabGenVLLM

            logger.info("[Backend] Using vLLM backend")
            tabgen = TabGenVLLM(
                model=model,
                mode=encoding_format,
                float_precision=float_precision,
                tensor_parallel_size=tensor_parallel,
                max_model_len=actual_max_len,
                gpu_memory_utilization=0.80,
                max_batch_size=initial_params["max_batch_size"],
                langfuse_enabled=not disable_langfuse,
                session_id=experiment,
                k_shots=k_shot,
                verbose=verbose,
            )
        elif backend == "tensorrt":
            from flash_tabgen_tensorrt.core.tabgen_tensorrt import TabGenTensorRT

            logger.info("[Backend] Using TensorRT-LLM backend")
            tabgen = TabGenTensorRT(
                model=model,
                mode=encoding_format,
                float_precision=float_precision,
                max_batch_size=initial_params["max_batch_size"],
                max_input_len=initial_params["max_input_len"],
                max_output_len=initial_params["max_output_len"],
                dtype="bfloat16",
                use_paged_attention=True,
                use_inflight_batching=True,
                max_concurrent_prompts=initial_params["concurrent_prompts"],
                k_shots=k_shot,
                verbose=verbose,
            )
        elif backend == "remote-vllm":
            from flash_tabgen_tensorrt.core.tabgen_remote_vllm import TabGenRemoteVLLM

            if not server_url:
                raise ValueError(
                    "remote-vllm backend requires --server-url (or VLLM_SERVER_URL env var)"
                )
            logger.info(f"[Backend] Using remote vLLM backend → {server_url}")
            tabgen = TabGenRemoteVLLM(
                model=model,
                base_url=server_url,
                api_key=api_key if api_key else None,
                mode=encoding_format,
                float_precision=float_precision,
                concurrent_requests=concurrent_requests,
                request_timeout=request_timeout,
                enable_thinking=enable_thinking,
                langfuse_enabled=not disable_langfuse,
                session_id=experiment,
                k_shots=k_shot,
                verbose=verbose,
            )
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Use 'vllm', 'tensorrt', or 'remote-vllm'"
            )

        # Build type_overrides: merge --date-columns and --type-overrides
        type_overrides = {col: "datetime" for col in date_columns} if date_columns else {}
        valid_types = {"categorical", "integer", "float", "boolean", "datetime", "text", "id"}
        for pair in type_overrides_raw:
            if "=" not in pair:
                raise ValueError(
                    f"Invalid --type-overrides format: '{pair}'. Expected col=type "
                    f"(e.g., modelTypeCode=text)"
                )
            col_name, col_type = pair.split("=", 1)
            if col_type not in valid_types:
                raise ValueError(
                    f"Invalid type '{col_type}' for column '{col_name}'. "
                    f"Valid types: {sorted(valid_types)}"
                )
            type_overrides[col_name] = col_type
        type_overrides = type_overrides if type_overrides else None
        profile = tabgen.profile(
            train_df_filtered, target_column=target_column, type_overrides=type_overrides
        )
        batch_params = initial_params

        # ------------------------------------------------------------------
        # Adaptive rows-per-prompt: reduce batch_size for wide schemas so
        # that the generated output fits within the token budget, avoiding
        # truncation and row loss.
        # ------------------------------------------------------------------
        batch_size = _adapt_rows_per_prompt(
            batch_size=batch_size,
            train_data=train_df_filtered,
            model_name=model,
            k_shot=k_shot,
            batch_params=batch_params,
        )

        output_path = (
            f"experiments/{experiment}/datasets/synthetic/synthetic_llm_{k_shot}_shots.csv"
        )
        os.makedirs(f"experiments/{experiment}/datasets/synthetic", exist_ok=True)

        logger.info(f"[Worker] Generating {n_rows} rows...")
        gen_start = time.time()

        if backend == "vllm":
            synthetic = tabgen.generate(
                n_rows=n_rows,
                conditional=conditionals,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                max_tokens=initial_params["estimated_output_tokens"],
                batch_size=batch_size,
                seed=42,
            )
        elif backend == "remote-vllm":
            # For the remote path we need a generous token budget because the
            # auto-estimation can undershoot for large batch sizes (25 rows of a
            # wide dataset can easily require 5k–8k tokens).
            # Floor: 8192 tokens (covers ~25 rows for most datasets).
            # If the model is a thinking model with enable_thinking=True, the
            # reasoning phase is emitted in a separate ``message.reasoning``
            # field but *still counts* towards the ``max_tokens`` budget.
            # Reasoning can easily consume 10k–30k+ tokens for complex
            # tabular prompts (wide schemas, many k-shots, correlation
            # matrices).  Since modern reasoning models (Qwen3.5, DeepSeek-R1)
            # support 128K+ context windows, we use a generous 65536 floor
            # to avoid truncation.
            _remote_min_tokens = 65536 if enable_thinking else 8192
            remote_max_tokens = max(initial_params["estimated_output_tokens"], _remote_min_tokens)
            logger.info(
                f"[Remote vLLM] max_tokens for generation: {remote_max_tokens} "
                f"(estimated={initial_params['estimated_output_tokens']}, "
                f"floor={_remote_min_tokens})"
            )
            synthetic = tabgen.generate(
                n_rows=n_rows,
                conditional=conditionals,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                max_tokens=remote_max_tokens,
                batch_size=batch_size,
                prompts_per_batch=concurrent_requests,
            )
        else:  # tensorrt
            synthetic = tabgen.generate(
                n_rows=n_rows,
                conditional=conditionals,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_new_tokens=initial_params["estimated_output_tokens"],
                batch_size=batch_size,
                seed=42,
                use_batch_generation=True,
                permute=permute,
                use_correlation_matrix=use_correlation_matrix,
            )

        synthetic.to_csv(output_path, index=False)
        logger.info(f"[Worker] ✓ Saved to {output_path} (Time: {time.time() - gen_start:.2f}s)")

        # -------------------------------------------------------------
        # Langfuse post-generation: register dataset, prompt, log trace
        # -------------------------------------------------------------
        if backend in ("remote-vllm", "vllm") and hasattr(tabgen, "register_dataset"):
            try:
                dataset_name = os.path.splitext(os.path.basename(train_data_path))[0]
                tabgen.register_dataset(
                    dataset_name=f"{experiment}/{dataset_name}",
                    description=(
                        f"Real dataset for experiment {experiment}, "
                        f"{tabgen.dataset_profile.n_rows} rows × "
                        f"{tabgen.dataset_profile.n_cols} cols"
                    ),
                    sample_items=5,
                )
            except Exception as exc:
                logger.debug(f"[Langfuse] Dataset registration skipped: {exc}")

            try:
                tabgen.register_prompt_template()
            except Exception as exc:
                logger.debug(f"[Langfuse] Prompt registration skipped: {exc}")

            if tabgen.last_trace_id:
                logger.info(f"[Langfuse] Trace ID: {tabgen.last_trace_id}")
            if tabgen.last_trace_url:
                logger.info(f"[Langfuse] Trace URL: {tabgen.last_trace_url}")

        # Save config (includes trace_id for later score attachment)
        config_path = f"experiments/{experiment}/config_k{k_shot}.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        tabgen.save_config(config_path)

        # -------------------------------------------------------------
        # SIGNAL PARENT TO KILL US
        # -------------------------------------------------------------
        print(">>>>WORKER_COMPLETED_SUCCESSFULLY<<<<")  # KEEP PRINT for parent signal detection
        sys.stdout.flush()  # Force print to buffer so parent sees it immediately

        # Go to sleep and wait for death (Parent will kill us)
        time.sleep(30)

    except Exception as e:
        logger.error(f"[Worker] ✗ Fatal Error: {e}", exc_info=True)
        os._exit(1)


# ==========================================
# MAIN LAUNCHER LOGIC
# ==========================================
if __name__ == "__main__":
    # Check if we are running as a Child Worker
    is_worker = "--worker-mode" in sys.argv

    if is_worker:
        # ---------------------------------------------------------
        # CHILD PROCESS CODE (WORKER)
        # ---------------------------------------------------------
        sys.argv.remove("--worker-mode")
        args = tensorrt_parse_args()

        current_k_shots = args.k_shots if isinstance(args.k_shots, list) else [args.k_shots]
        if isinstance(current_k_shots, list) and len(current_k_shots) > 0:
            current_k_shots = [current_k_shots[0]]

        run_worker(
            experiment=args.experiment,
            train_data_path=args.dataset,
            target_column=args.target_column,
            k_shots=current_k_shots,
            model=args.model,
            permute=args.permute,
            use_correlation_matrix=args.use_correlation_matrix,
            batch_size=args.batch_size,
            device=args.device,
            n_rows=args.n_rows,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            float_precision=args.float_precision,
            min_p=args.min_p,
            conditionals=args.conditionals,
            date_columns=args.date_columns,
            type_overrides_raw=getattr(args, "type_overrides", []),
            max_context_len=args.max_context_len,
            tensor_parallel=args.tensor_parallel,
            verbose=getattr(args, "verbose", False),
            backend=args.backend,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            server_url=getattr(args, "server_url", ""),
            api_key=getattr(args, "api_key", ""),
            concurrent_requests=getattr(args, "concurrent_requests", 300),
            request_timeout=getattr(args, "request_timeout", 600.0),
            enable_thinking=getattr(args, "enable_thinking", False),
            disable_langfuse=getattr(args, "disable_langfuse", False),
            presence_penalty=getattr(args, "presence_penalty", 1.5),
            repetition_penalty=getattr(args, "repetition_penalty", 1.0),
            encoding_format=getattr(args, "encoding_format", "json"),
        )

    else:
        # ---------------------------------------------------------
        # PARENT LAUNCHER CODE (THE EXECUTOR)
        # ---------------------------------------------------------
        args = tensorrt_parse_args()
        script_path = os.path.abspath(__file__)
        total_start = time.time()

        print(f"\n[Launcher] Starting Experiment Sequence: {args.k_shots}")
        print(f"[Launcher] Backend: {args.backend.upper()}")

        k_shots_list = args.k_shots if isinstance(args.k_shots, list) else [args.k_shots]

        for k_shot in k_shots_list:
            print(f"\n[Launcher] >>> Spawning clean process for k_shot={k_shot}")

            # Build command
            cmd = [sys.executable, script_path, "--worker-mode"]

            # Robust Argument Filtering
            raw_args = sys.argv[1:]
            i = 0
            while i < len(raw_args):
                arg = raw_args[i]
                if arg == "--k-shots":
                    i += 1
                    while i < len(raw_args) and not raw_args[i].startswith("-"):
                        i += 1
                else:
                    cmd.append(arg)
                    i += 1
            cmd.extend(["--k-shots", str(k_shot)])

            # ------------------------------------------------------------------
            # NUCLEAR LAUNCHER OPTION:
            # We read output real-time. If we see "COMPLETED", we kill the PID.
            # ------------------------------------------------------------------
            try:
                # Use Popen instead of run to get handle on process
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    text=True,
                    bufsize=1,  # Line buffered
                )

                worker_finished_successfully = False

                # Read stdout line by line
                while True:
                    line = proc.stdout.readline()
                    if not line and proc.poll() is not None:
                        break  # Process died on its own

                    if line:
                        print(line, end="")  # Echo to console

                        # Check for the secret signal
                        if ">>>>WORKER_COMPLETED_SUCCESSFULLY<<<<" in line:
                            print(
                                f"[Launcher] Detected success signal. FORCE KILLING PID {proc.pid}..."
                            )
                            worker_finished_successfully = True

                            # Give it a split second to flush logs
                            time.sleep(0.5)

                            # KILL IT WITH FIRE
                            proc.kill()
                            break

                # Ensure process is collected
                proc.wait()

                if (
                    not worker_finished_successfully
                    and proc.returncode != 0
                    and proc.returncode != -9
                ):
                    print(f"[Launcher] ⚠ Worker exited abnormally with code {proc.returncode}")

                # Wait for OS cleanup
                time.sleep(5)

            except Exception as e:
                print(f"[Launcher] ✗ Error running worker: {e}")
                import traceback

                traceback.print_exc()

        print(f"\n[Launcher] All experiments finished in {time.time() - total_start:.2f}s")
        sys.exit(0)
