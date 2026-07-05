# AGENTS.md — Coding Agent Reference

This file provides instructions for AI coding agents working in this repository.

## Project Overview

**FewShotTabLLM** is a Python library and set of scripts for generating synthetic tabular data
using LLMs (Large Language Models). It supports three inference backends through a single CLI
entry point (`tensorrt_sampling_optimized.py`):

- **TensorRT-LLM** (`--backend tensorrt`) — highest throughput, builds an engine on first run
- **vLLM (local)** (`--backend vllm`) — no engine build, works with any HuggingFace model
- **Remote vLLM** (`--backend remote-vllm`) — HTTP calls to a deployed vLLM server, no local GPU needed

Evaluation and visualization are handled by `ml_eval.py`, `plot_evaluation_graphs.py`, and the
standalone `synthetic_data_evaluator` package (maintained in a separate repository).

---

## Repository Structure

```
flash_tabgen_tensorrt/       # Core library package
  core/                      # Data profiler, encoders, prompt builder,
                             #   generators (vllm/tensorrt/remote), tabgen classes
services/                    # Shared utility modules (parsers, sampling,
                             #   data_handler, logger_config, plotting helpers)
mcp_server/                  # Optional MCP server integration
tensorrt_sampling_optimized.py  # Main CLI entry-point (all backends)
ml_eval.py / ml_eval.ipynb      # Evaluation scripts
plot_evaluation_graphs.py       # Plotting scripts
pyproject.toml                  # Project metadata and tool config
```

> **Note:** The `synthetic_data_evaluator` package is maintained in a separate
> repository: <https://github.com/sordi-ai/Synthetic-Data-Generation-Evaluator>.
> Install it with `pip install -e /path/to/Synthetic-Data-Generation-Evaluator/synthetic_data_evaluator/`.

---

## Environment Setup

Python **3.12** is recommended (3.9–3.11 also supported per pyproject.toml).

### vLLM (local) backend

```bash
python3 -m venv venv && source venv/bin/activate
pip install uv
uv pip install -r requirements_vllm.txt
pip install vllm
uv pip install flash-attn --no-build-isolation
pip install -e ".[dev]"
```

### Remote vLLM backend

No local GPU required. The heavy dependencies run on the server side.

```bash
python3 -m venv venv && source venv/bin/activate
pip install uv
uv pip install -r requirements_vllm.txt
pip install -e ".[dev]"
```

### TensorRT-LLM backend

```bash
pip install uv
uv pip install -r requirements_tensorrt.txt
uv pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130
apt-get update && apt-get -y install libopenmpi-dev libzmq3-dev
pip install --upgrade pip setuptools && pip3 install tensorrt_llm
uv pip install flash-attn --no-build-isolation
pip install -e ".[dev]"
```

### Evaluator package

```bash
git clone https://github.com/sordi-ai/Synthetic-Data-Generation-Evaluator.git
pip install -e Synthetic-Data-Generation-Evaluator/synthetic_data_evaluator/
```

Copy `example.env` to `.env` and set `CUDA_VISIBLE_DEVICES` as needed. For remote-vllm, also
set `VLLM_SERVER_URL` and optionally `VLLM_API_KEY`. For Langfuse observability, set
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`.

Both `tensorrt_sampling_optimized.py` and `ml_eval.py` call `dotenv.load_dotenv()` at startup,
so `.env` is loaded automatically — no need to export variables manually.

---

## Build / Lint / Test Commands

### Formatting

```bash
# Format code (line-length 100)
black .

# Lint with ruff
ruff check .

# Auto-fix ruff issues
ruff check --fix .
```

### Type checking

```bash
mypy flash_tabgen_tensorrt/ services/
```

### Running tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=flash_tabgen_tensorrt --cov=services --cov-report=term-missing

# Run a single test file
pytest mcp_server/tests/test_validation.py

# Run a single test by keyword
pytest mcp_server/tests/test_validation.py -k "test_name"
```

Tests use the stdlib `unittest.TestCase` style and are discovered by pytest automatically.

---

## Code Style Guidelines

### General

- **Line length**: 100 characters (enforced by `black` and `ruff`).
- **Python version**: Target Python 3.9+ syntax; avoid features exclusive to 3.10+.
- **Formatter**: `black` (do not manually reformat — let black handle it).
- **Linter**: `ruff` for import order and common issues.

### Imports

- Use absolute imports; avoid relative imports except within the same sub-package.
- Group imports in the standard order: stdlib → third-party → local. `ruff` enforces this.
- Import only what is needed; avoid wildcard imports (`from x import *`).
- Heavy optional dependencies (e.g., `vllm`, `tensorrt_llm`, `httpx`, `langfuse`) must be imported lazily
  inside functions/methods so the module can be imported without them installed:

```python
# Good — lazy import of optional backend
def generate(self, ...):
    from vllm import LLM
    ...
```

### Type Annotations

- Annotate all public function signatures with type hints.
- Use `Optional[X]` (not `X | None`) for Python 3.9 compatibility.
- Use `List`, `Dict`, `Tuple` from `typing` (not the built-in generics) for 3.9 compat.
- `mypy` is configured with `disallow_untyped_defs = false`; annotations are encouraged but
  not strictly required on internal/private helpers.
- Use `@dataclass` for plain data containers (see `ColumnProfile`, `DatasetProfile`).

### Naming Conventions

- **Modules/packages**: `snake_case` (e.g., `data_profiler.py`, `generator_vllm.py`).
- **Classes**: `PascalCase` (e.g., `PromptBuilder`, `GeneratorVLLM`, `DatasetProfile`).
- **Functions/methods**: `snake_case` (e.g., `build_generation_prompt`, `select_representative_samples`).
- **Constants / config parameters**: `UPPER_SNAKE_CASE` for module-level constants.
- **Private helpers**: prefix with a single underscore (`_stratified_sample`).
- CLI argument names use `--kebab-case`; the corresponding Python variable uses `snake_case`.

### Docstrings

- Use Google-style docstrings for all public classes and functions.
- Include `Args:`, `Returns:`, and `Raises:` sections where relevant.
- Module-level docstrings should be a short summary of the module's purpose.

```python
def build_generation_prompt(self, demo_data: pd.DataFrame, n_samples: int = 1) -> str:
    """
    Build a few-shot generation prompt.

    Args:
        demo_data: DataFrame of real rows used as demonstrations.
        n_samples: Number of synthetic rows to request.

    Returns:
        Formatted prompt string ready to be sent to the model.
    """
```

### Error Handling

- Raise specific exceptions (`ValueError`, `RuntimeError`, `FileNotFoundError`) with
  informative messages rather than bare `Exception`.
- Use `logging` (not `print`) for diagnostic messages inside library code.
  Get loggers with `logging.getLogger(__name__)`.
- Use `print` only in top-level CLI scripts (`tensorrt_sampling_optimized.py`, etc.) and
  notebooks where direct user output is expected.
- Do not swallow exceptions silently; log the error before re-raising or returning a fallback.

### pandas / numpy conventions

- Prefer explicit `copy(deep=True)` when modifying a slice to avoid `SettingWithCopyWarning`.
- Use `pd.DataFrame` / `pd.Series` type hints, not bare `Any`.
- Avoid chained indexing; use `.loc[]` or `.iloc[]`.

### Configuration

- Experiment parameters are set directly in scripts/notebooks (no external config files required
  for basic use). `hydra-core` / `omegaconf` are available for advanced config management.
- Environment variables are loaded from `.env` (see `example.env`).

---

## Key Patterns

- **Backend abstraction**: `GeneratorTensorRT`, `GeneratorVLLM`, and `GeneratorRemoteVLLM` all
  expose a `.generate()` method returning `List[str]`. New backends should follow the same
  interface. The corresponding `TabGen*` classes (`TabGenVLLM`, `TabGenTensorRT`,
  `TabGenRemoteVLLM`) orchestrate profiling, prompt building, generation, and decoding.
- **DatasetProfile + ColumnProfile**: All metadata about the real dataset is captured in these
  dataclasses before generation and passed down through `PromptBuilder`.
- **Remote vLLM uses `/v1/chat/completions`**: The remote backend sends requests via the
  OpenAI-compatible chat completions endpoint. This is important for reasoning models (e.g.,
  Qwen3.5-122B-A10B) where `/v1/completions` ignores `enable_thinking=False` and produces
  `<think>` blocks instead of tabular output.
- **Thinking mode is supported but optional**: When `--enable-thinking` is set on the
  remote-vllm backend, the server returns reasoning in `message.reasoning_content` and the
  final answer (CSV rows) in `message.content`. Key implementation details:
  - The code reads `message.reasoning_content` (with fallback to `message.reasoning`) and
    extracts the tabular output from `message.content`.
  - If `content` is `None` or empty (the model spent all tokens reasoning), the code logs a
    warning and treats the response as zero decoded rows for that batch. This typically
    happens when `finish_reason == "length"` — the token budget was exhausted by thinking.
  - To mitigate budget exhaustion, the remote generator enforces a floor of **65 536** max
    tokens when thinking is enabled (`THINKING_TOKEN_FLOOR` in `generator_remote_vllm.py`).
  - Thinking mode is **off by default** and recommended off for throughput-oriented runs.
    Enable it only when reasoning quality matters more than speed/cost.
- **Encoding format defaults to JSON**: The `--encoding-format` flag defaults to `json`,
  which uses structured JSON objects for few-shot examples and model output. This enables
  guided JSON decoding on vLLM backends, producing significantly higher decode success rates
  compared to the older `predictive` (GReaT-style) format. To use the legacy format:
  - `--encoding-format predictive` — GReaT-style `column is value` format
- **Experiment directories**: All outputs go under `experiments/{experiment_name}/`:
  - `datasets/synthetic/` — generated CSV files
  - `datasets/real/` — representative example pool
  - `evaluation_reports/` — ML evaluation CSV results
  - `config_k{N}.json` — generation config + Langfuse trace ID per k-shot
  - `logs/` — generation log files
  - `figures/` — plots

---

## ML Evaluation (`ml_eval.py`)

Evaluates synthetic datasets against the real dataset using multiple ML classifiers/regressors.
Experiment names may contain `/` (e.g., `qwen3.5-9b/adult-regen`) — they are sanitised to
underscores for filename usage but used as-is for directory lookup.

### Usage

```bash
python ml_eval.py \
  --dataset real_data/adult/adult.csv \
  --experiments "qwen3.5-9b/adult-regen" \
  --synthetic-datasets real synthetic_llm_120_shots synthetic_llm_140_shots synthetic_llm_160_shots \
  --target-column income \
  --task-type classification \
  --train-size 1000 \
  --test-size 500
```

### How it works

1. Copies synthetic CSVs from `experiments/{experiment}/datasets/synthetic/` to
   `ML_EVALUATIONS/datasets/{dataset_name}/`.
2. Splits the real dataset into train/test and saves to the same directory.
3. Auto-generates `metadata.json` (SDV-format column types) if it does not exist.
4. Trains multiple models (SVM, DecisionTree, RandomForest, LogisticRegression, MLP)
   on each synthetic dataset and evaluates on the held-out real test set.
5. Copies results back to `experiments/{experiment}/evaluation_reports/`.
6. If Langfuse is configured, pushes per-model scores back to the generation trace
   using the `last_trace_id` from `config_k{N}.json`.

### Langfuse score push-back

When a `config_k{N}.json` contains a `last_trace_id`, ml_eval automatically pushes
evaluation scores (e.g., `ml_eval/RandomForestClassifier`, `ml_eval/Average`) to the
corresponding Langfuse trace. This links generation quality to downstream ML performance.

Disable with `--disable-langfuse` (or `--disable-lang-fuse`).

---

## Langfuse Observability

All Langfuse integration is centralised in `flash_tabgen_tensorrt/core/langfuse_utils.py`
via the `LangfuseManager` class. It is **fail-safe**: if the `langfuse` package is missing,
credentials are not set, or any API call fails, generation continues uninterrupted.

### What is traced

- **Generation**: Each k-shot run creates a root trace (`tabgen-generate`) with child
  spans for profiling and generation observations for each LLM batch call.
- **Prompts**: The generation prompt template is versioned in Langfuse as
  `tabgen-generation-prompt`.
- **Datasets**: The real dataset profile is registered as a Langfuse dataset.
- **Sessions**: All traces for an experiment share a `session_id` matching the
  `--experiment` name.
- **Scores**: ML evaluation metrics are pushed back to the corresponding generation
  trace after `ml_eval.py` runs.

### Proxy configuration

`LangfuseManager.init()` automatically adds the Langfuse host domain (extracted from
`LANGFUSE_BASE_URL`) to `NO_PROXY`/`no_proxy` to avoid corporate proxy interference.
This is also set in `.env` for belt-and-suspenders reliability.

---

## Notes for Agents

- There are no Cursor rules (`.cursorrules`) or GitHub Copilot instructions
  (`.github/copilot-instructions.md`) in this repository.
- The `synthetic_data_evaluator` package is maintained in a separate repository
  (<https://github.com/sordi-ai/Synthetic-Data-Generation-Evaluator>). Install it
  with `pip install -e` from a local checkout. Changes to its API may require
  updating both its tests and the top-level `ml_eval.py` / `ml_eval.ipynb`.
- GPU-dependent code (TensorRT, vLLM) should not be executed in CI without a GPU; keep
  such imports lazy so unit tests can run on CPU-only machines.
