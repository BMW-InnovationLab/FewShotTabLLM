import argparse
import os


def tensorrt_parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic tabular data using TensorRT-LLM, vLLM, or a remote vLLM server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using TensorRT-LLM backend (default)
  python tensorrt_sampling_optimized.py --dataset adult.csv --experiment experiment_C2 \\
                              --target-column income --k-shots 40 60 80 --backend tensorrt

  # Using local vLLM backend
  python tensorrt_sampling_optimized.py --dataset adult.csv --experiment experiment_C2 \\
                              --target-column income --k-shots 40 60 80 --backend vllm

  # Using a remote vLLM server (no local GPU required)
  python tensorrt_sampling_optimized.py --dataset adult.csv --experiment experiment_C2 \\
                              --target-column income --k-shots 40 60 80 \\
                              --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
                              --backend remote-vllm --server-url http://my-server:8000 \\
                              --concurrent-requests 64 --batch-size 25

  # Remote vLLM with API key, via env var for URL
  export VLLM_SERVER_URL=http://my-server:8000
  export VLLM_API_KEY=my-secret-token
  python tensorrt_sampling_optimized.py --dataset adult.csv --experiment experiment_C2 \\
                              --target-column income --k-shots 40 60 80 \\
                              --backend remote-vllm --concurrent-requests 32

  # Full example with all parameters
  python tensorrt_sampling_optimized.py --dataset adult.csv --experiment experiment_C2 \\
                              --target-column income --k-shots 40 60 80 100 120 140 160 \\
                              --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
                              --n-rows 5000 --batch-size 25 \\
                              --device 0 --top-p 0.8 --top-k 20 --temperature 0.7 \\
                              --float-precision 4 --min-p 0 --use-correlation-matrix

  # Disable correlation matrix and enable permutation
  python tensorrt_sampling_optimized.py --dataset adult.csv --experiment experiment_C2 \\
                              --target-column income --k-shots 40 60 \\
                              --no-correlation-matrix --permute
        """,
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "tensorrt", "remote-vllm"],
        default="tensorrt",
        help=(
            "Inference backend to use: 'tensorrt' (default, highest local throughput), "
            "'vllm' (local in-process vLLM), or 'remote-vllm' (HTTP to a deployed vLLM server)."
        ),
    )

    # ------------------------------------------------------------------
    # Remote vLLM server options
    # ------------------------------------------------------------------
    parser.add_argument(
        "--server-url",
        type=str,
        default=os.environ.get("VLLM_SERVER_URL", ""),
        help=(
            "URL of the remote vLLM server (remote-vllm backend only). "
            "Example: http://my-server:8000. "
            "Can also be set via the VLLM_SERVER_URL environment variable."
        ),
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("VLLM_API_KEY", ""),
        help=(
            "Optional Bearer API key for the remote vLLM server. "
            "Can also be set via the VLLM_API_KEY environment variable."
        ),
    )

    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=300,
        help=(
            "Maximum number of concurrent HTTP requests sent to the remote vLLM server "
            "(remote-vllm backend only). The server supports up to ~500 in-flight requests; "
            "default 300 keeps a safe margin. Higher values increase throughput."
        ),
    )

    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help=(
            "Per-request HTTP timeout in seconds for the remote vLLM backend. "
            "Increase for very large batch sizes or slow networks. Default: 600."
        ),
    )

    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help=(
            "Enable chain-of-thought / reasoning mode for thinking models "
            "(e.g. Qwen3-thinking, DeepSeek-R1) when using the remote-vllm backend. "
            "When disabled (default) the model skips the <think>...</think> block, "
            "which keeps the token budget focused on tabular output and avoids "
            "zero-row decode failures on long prompts."
        ),
    )

    parser.add_argument(
        "--disable-lang-fuse",
        "--disable-langfuse",
        action="store_true",
        default=False,
        dest="disable_langfuse",
        help=(
            "Disable Langfuse observability tracing even when LANGFUSE_* environment "
            "variables are configured. By default Langfuse is enabled automatically "
            "when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set in the environment."
        ),
    )

    parser.add_argument(
        "--encoding-format",
        type=str,
        choices=["json", "predictive"],
        default="json",
        help=(
            "Encoding format used for few-shot examples and expected model output. "
            "'json' (default) uses JSON objects for both examples and output, enabling "
            "guided JSON decoding on vLLM backends for higher decode success rates. "
            "'predictive' uses the legacy GReaT-style 'column is value' format with "
            "target-last ordering."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="adult.csv",
        help="Path to the training data CSV file (default: adult.csv)",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment_C2",
        help="Name of the experiment (default: experiment_C2)",
    )

    parser.add_argument(
        "--target-column",
        type=str,
        default="income",
        help="Name of the target column (default: income)",
    )

    parser.add_argument(
        "--k-shots",
        nargs="+",
        type=int,
        default=[40, 60, 80, 100, 120, 140, 160],
        help="List of k-shot values to use for generation (default: 40 60 80 100 120 140 160)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model name or path (default: Qwen/Qwen3-30B-A3B-Instruct-2507)",
    )

    parser.add_argument(
        "--n-rows",
        type=int,
        default=5000,
        help="Number of synthetic rows to generate (default: 5000)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=25, help="Batch size for generation (default: 25)"
    )

    parser.add_argument("--device", type=int, default=0, help="GPU device ID to use (default: 0)")

    parser.add_argument(
        "--top-p", type=float, default=0.8, help="Top-p sampling parameter (default: 0.8)"
    )

    parser.add_argument(
        "--top-k", type=int, default=20, help="Top-k sampling parameter (default: 20)"
    )

    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for sampling (default: 0.7)"
    )

    parser.add_argument(
        "--float-precision",
        type=int,
        default=4,
        help="Float precision for generated numbers (default: 4)",
    )

    parser.add_argument(
        "--min-p", type=float, default=0.0, help="Minimum p value for sampling (default: 0.0)"
    )

    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=1.5,
        help=(
            "Penalise tokens that have already appeared in the prompt/output. "
            "Higher values encourage the model to use a wider vocabulary. "
            "Supported by vLLM and remote-vLLM backends. (default: 1.5)"
        ),
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help=(
            "Multiplicative penalty applied to repeated tokens. "
            "Values > 1.0 discourage repetition. (default: 1.0)"
        ),
    )

    parser.add_argument(
        "--max-input-len",
        type=int,
        default=16384,
        help="Maximum input sequence length for TensorRT engine (prompt length limit). This is an engine configuration parameter (default: 16384)",
    )

    parser.add_argument(
        "--max-output-len",
        type=int,
        default=16384,
        help="Maximum output sequence length for TensorRT engine (generation length limit). This is an engine configuration parameter (default: 16384)",
    )

    parser.add_argument(
        "--permute",
        action="store_true",
        default=False,
        help="Enable column permutation during generation (default: False)",
    )

    parser.add_argument(
        "--no-permute",
        action="store_false",
        dest="permute",
        help="Disable column permutation during generation",
    )

    parser.add_argument(
        "--use-correlation-matrix",
        action="store_true",
        default=True,
        help="Use correlation matrix during generation (default: True)",
    )

    parser.add_argument(
        "--no-correlation-matrix",
        action="store_false",
        dest="use_correlation_matrix",
        help="Disable correlation matrix during generation",
    )

    parser.add_argument(
        "--conditionals",
        nargs="*",
        default=[],
        help="List of conditional constraints (default: [])",
    )

    parser.add_argument(
        "--date-columns",
        nargs="*",
        default=[],
        help="List of column names to treat as datetime (e.g., --date-columns assemblyDate orderDate)",
    )

    parser.add_argument(
        "--type-overrides",
        nargs="*",
        default=[],
        help=(
            "Override inferred column types as col=type pairs. "
            "Valid types: categorical, integer, float, boolean, datetime, text, id. "
            "Example: --type-overrides modelTypeCode=text vin10=id"
        ),
    )

    parser.add_argument(
        "--max-context-len",
        type=int,
        default=None,
        help="Maximum context length for the model. Use to reduce memory for large models (e.g., 32768)",
    )

    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1). Use 2 for large models across 2 GPUs.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output including prompt printing (default: False)",
    )

    return parser.parse_args()


def plot_evaluation_parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots and data statistics visualizations for experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default parameters
  python plot_evaluation_graphs.py --dataset adult.csv --experiments experiment_C2 \\
                                    --synthetic-datasets ctgan tvae synthetic_llm_40_shots

  # Multiple experiments with various synthetic datasets
  python plot_evaluation_graphs.py --dataset adult.csv \\
                                    --experiments experiment_C2 experiment_C3 \\
                                    --synthetic-datasets ctgan tvae begreat \\
                                                        synthetic_llm_40_shots \\
                                                        synthetic_llm_60_shots \\
                                                        synthetic_llm_80_shots
        """,
    )

    parser.add_argument(
        "--synthetic-datasets",
        nargs="+",
        type=str,
        required=True,
        help="List of synthetic dataset names (without path or extension) to evaluate. "
        "Examples: ctgan, tvae, begreat, synthetic_llm_40_shots",
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        type=str,
        default=["experiment_C2"],
        help="List of experiment names to plot (default: experiment_C2)",
    )

    parser.add_argument("--dataset", type=str, help="Path to the dataset CSV file", required=True)

    return parser.parse_args()


def ml_eval_parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ML evaluation experiments on synthetic tabular data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default parameters
  python ml_eval.py --dataset adult.csv --experiments experiment_A2 experiment_B2 \\
                    --synthetic-datasets real ctgan tvae synthetic_llm_40_shots

  # Full example with all parameters
  python ml_eval.py --dataset adult.csv \\
                    --experiments experiment_A2 experiment_B2 \\
                    --synthetic-datasets real ctgan tvae begreat \\
                                        synthetic_llm_40_shots \\
                                        synthetic_llm_60_shots \\
                                        synthetic_llm_80_shots \\
                    --target-column income \\
                    --task-type classification \\
                    --train-size 5000 --test-size 1000
        """,
    )

    parser.add_argument("--dataset", type=str, help="Path to the dataset CSV file", required=True)

    parser.add_argument(
        "--id-column",
        type=str,
        help="Name of the ID column in the dataset (optional)",
        required=False,
    )

    parser.add_argument(
        "--synthetic-datasets",
        nargs="+",
        type=str,
        required=True,
        help="List of synthetic dataset names to evaluate. "
        "Examples: real, ctgan, tvae, begreat, synthetic_llm_40_shots",
    )

    parser.add_argument(
        "--date-columns",
        nargs="+",
        type=str,
        default=[],
        help="List of date column names to process (default: empty list)",
        required=False,
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        type=str,
        default=["experiment_A2", "experiment_B2"],
        help="List of experiment names to evaluate (default: experiment_A2 experiment_B2)",
        required=True,
    )

    parser.add_argument(
        "--target-column", type=str, help="Name of the target column for prediction", required=True
    )

    parser.add_argument(
        "--task-type", type=str, choices=["classification", "regression"], help="Type of ML task"
    )

    parser.add_argument(
        "--train-size", type=int, default=5000, help="Training dataset size (default: 5000)"
    )

    parser.add_argument(
        "--test-size", type=int, default=1000, help="Test dataset size (default: 1000)"
    )

    parser.add_argument(
        "--disable-lang-fuse",
        "--disable-langfuse",
        dest="disable_langfuse",
        action="store_true",
        default=False,
        help=(
            "Disable Langfuse score push-back. By default, if a generation trace "
            "ID is found in the experiment config, evaluation scores are pushed to "
            "the corresponding Langfuse trace."
        ),
    )

    return parser.parse_args()
