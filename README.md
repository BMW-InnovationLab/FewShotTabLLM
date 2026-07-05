# FewShotTabLLM — Few-Shot LLMs as Synthetic Tabular Data Generators

Official implementation of the paper **"Few-Shot LLMs as Synthetic Tabular Data
Generators"** (IEEE COMPSAC 2026) — see [Citation](#citation).

FewShotTabLLM is a few-shot, **training-free** framework for generating
high-quality synthetic tabular data with Large Language Models. The library
profiles your real dataset, builds schema-enriched few-shot prompts from
representative samples, and uses an LLM to generate new rows that preserve the
statistical properties of the original data — no dataset-specific training,
fine-tuning, or preprocessing required.

Three inference backends are supported -- pick the one that fits your setup:

| Backend | Flag | GPU required on client? | Notes |
|---|---|---|---|
| **TensorRT-LLM** | `--backend tensorrt` | Yes | Highest throughput. Builds an engine on first run. |
| **vLLM (local)** | `--backend vllm` | Yes | No engine build. Works with any HuggingFace model. |
| **Remote vLLM** | `--backend remote-vllm` | No | HTTP calls to a deployed vLLM server. |

## Quick Start

```bash
# Install (vLLM backend -- easiest to get started)
python3 -m venv venv && source venv/bin/activate
pip install uv
uv pip install -r requirements_vllm.txt
pip install vllm
uv pip install flash-attn --no-build-isolation
pip install -e ".[dev]"

# Generate 5000 synthetic rows
python tensorrt_sampling_optimized.py \
  --backend vllm \
  --dataset adult.csv --experiment my_experiment \
  --target-column income --k-shots 40 60 80 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --n-rows 5000 --batch-size 25
```

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Generating Synthetic Data](#generating-synthetic-data)
  - [vLLM (local)](#vllm-local)
  - [Remote vLLM](#remote-vllm)
  - [TensorRT-LLM](#tensorrt-llm)
- [CLI Parameter Reference](#cli-parameter-reference)
- [How the Prompt Works](#how-the-prompt-works)
- [Evaluating Synthetic Data](#evaluating-synthetic-data)
- [Plotting Results](#plotting-results)
- [Supported Models](#supported-models)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [License](#license)
- [Citation](#citation)

---

## Environment Setup

**Python 3.12** recommended (3.9--3.11 also supported).

Create a `.env` file (see `example.env`):
```dotenv
# GPU device
CUDA_VISIBLE_DEVICES=0

# Remote vLLM backend (optional)
VLLM_SERVER_URL=http://localhost:8000
VLLM_API_KEY=

# Langfuse observability (optional -- leave blank to disable)
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=http://your-langfuse-host:3000
```

### vLLM Backend

```bash
python3 -m venv venv && source venv/bin/activate
pip install uv
uv pip install -r requirements_vllm.txt
pip install vllm
uv pip install flash-attn --no-build-isolation
pip install -e ".[dev]"
```

### Remote vLLM Backend

The remote backend only needs `httpx` on the client (no GPU, no torch).
The heavy dependencies are on the server side.

```bash
python3 -m venv venv && source venv/bin/activate
pip install uv
uv pip install -r requirements_vllm.txt
pip install -e ".[dev]"
```

### TensorRT-LLM Backend

Requires NVIDIA CUDA drivers, TensorRT-LLM, and a supported GPU.

```bash
pip install uv
uv pip install -r requirements_tensorrt.txt
uv pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130
apt-get update && apt-get -y install libopenmpi-dev libzmq3-dev
pip install --upgrade pip setuptools && pip3 install tensorrt_llm
uv pip install flash-attn --no-build-isolation
pip install -e ".[dev]"
```

---

## Generating Synthetic Data

All backends use the same CLI entry point: **`tensorrt_sampling_optimized.py`**.
Outputs are saved to `experiments/{experiment_name}/datasets/synthetic/`.

### vLLM (local)

```bash
python tensorrt_sampling_optimized.py \
  --backend vllm \
  --dataset adult.csv --experiment experiment_C2 \
  --target-column income --k-shots 40 60 80 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --n-rows 5000 --batch-size 25 \
  --top-p 0.8 --top-k 20 --temperature 0.7 \
  --use-correlation-matrix
```

**Multi-GPU (tensor parallelism):**
```bash
CUDA_VISIBLE_DEVICES=0,1 python tensorrt_sampling_optimized.py \
  --backend vllm \
  --dataset adult.csv --experiment experiment_C2 \
  --target-column income --k-shots 40 60 80 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --n-rows 5000 --batch-size 25 --tensor-parallel 2
```

### Remote vLLM

Send prompts to a deployed vLLM server over HTTP. No local GPU needed.

```bash
python tensorrt_sampling_optimized.py \
  --backend remote-vllm \
  --server-url http://my-gpu-server:8000 \
  --model Qwen/Qwen3.5-122B-A10B \
  --dataset adult.csv --experiment experiment_C2 \
  --target-column income --k-shots 40 60 80 \
  --n-rows 5000 --batch-size 25 \
  --concurrent-requests 300 --request-timeout 600
```

**With API key (via env vars):**
```bash
export VLLM_SERVER_URL=http://my-gpu-server:8000
export VLLM_API_KEY=my-secret-token

python tensorrt_sampling_optimized.py \
  --backend remote-vllm \
  --model Qwen/Qwen3.5-122B-A10B \
  --dataset adult.csv --experiment experiment_C2 \
  --target-column income --k-shots 40 60 80 \
  --n-rows 5000 --batch-size 25
```

**Reasoning models (thinking mode):**

Thinking mode is fully supported on the remote-vllm backend but disabled by
default. When enabled, the model reasons internally before producing tabular
output -- reasoning is returned in `message.reasoning_content` and the CSV
rows in `message.content`. The code enforces a minimum of 65 536 max tokens
when thinking is on to prevent the reasoning phase from exhausting the token
budget. If the budget is still exceeded (`finish_reason == "length"`), the
batch is treated as zero decoded rows and a warning is logged.

Keep `--enable-thinking` off for throughput-oriented runs. Enable it when
reasoning quality matters more than speed.

```bash
# Default (thinking disabled -- best throughput)
python tensorrt_sampling_optimized.py \
  --backend remote-vllm \
  --server-url http://my-server:8000 \
  --model Qwen/Qwen3.5-122B-A10B \
  --dataset adult.csv --experiment experiment_C2 \
  --target-column income --k-shots 40 --n-rows 100

# Enable thinking (supported -- higher quality, lower throughput)
python tensorrt_sampling_optimized.py \
  --backend remote-vllm --enable-thinking \
  --server-url http://my-server:8000 \
  --model Qwen/Qwen3.5-122B-A10B \
  --dataset adult.csv --experiment experiment_C2 \
  --target-column income --k-shots 40 --n-rows 100
```

### TensorRT-LLM

Builds an optimized inference engine on first run (cached for subsequent runs).

```bash
python tensorrt_sampling_optimized.py \
  --backend tensorrt \
  --dataset adult.csv --experiment experiment_C2 \
  --target-column income --k-shots 40 60 80 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --n-rows 5000 --batch-size 25 \
  --max-input-len 16384 --max-output-len 16384
```

**With date columns:**
```bash
python tensorrt_sampling_optimized.py \
  --backend vllm \
  --dataset orders.csv --experiment orders_experiment \
  --target-column product_category \
  --k-shots 120 --model Qwen/Qwen2.5-14B-Instruct \
  --n-rows 5000 --batch-size 25 \
  --date-columns order_date \
  --type-overrides product_code=text \
  --max-context-len 32768 --verbose
```

---

## CLI Parameter Reference

### Common Parameters (all backends)

| Parameter | Default | Description |
|---|---|---|
| `--backend` | `tensorrt` | `tensorrt`, `vllm`, or `remote-vllm` |
| `--dataset` | `adult.csv` | Path to training CSV |
| `--experiment` | `experiment_C2` | Experiment name (output directory) |
| `--target-column` | `income` | Target column for predictive mode |
| `--k-shots` | `40 60 80 ...` | K-shot values (space-separated; runs one experiment per value) |
| `--model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Model name |
| `--n-rows` | `5000` | Synthetic rows to generate |
| `--batch-size` | `25` | Rows per prompt |
| `--device` | `0` | GPU device ID (ignored when `--tensor-parallel > 1`) |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-p` | `0.8` | Nucleus sampling |
| `--top-k` | `20` | Top-k sampling |
| `--min-p` | `0.0` | Minimum probability |
| `--presence-penalty` | `1.5` | Penalise tokens already in prompt/output. Higher values encourage diversity |
| `--repetition-penalty` | `1.0` | Multiplicative penalty for repeated tokens. Values > 1.0 discourage repetition |
| `--float-precision` | `4` | Decimal places for floats |
| `--encoding-format` | `json` | `json` (structured output, guided decoding) or `predictive` (legacy GReaT-style) |
| `--date-columns` | `[]` | Columns to treat as datetime |
| `--type-overrides` | `[]` | Override inferred column types as `col=type` pairs. Valid types: `categorical`, `integer`, `float`, `boolean`, `datetime`, `text`, `id` |
| `--conditionals` | `[]` | Conditional constraints for generation (e.g., `--conditionals "age=30" "income=>50K"`) |
| `--use-correlation-matrix` | `True` | Include correlations in prompt (disable with `--no-correlation-matrix`) |
| `--permute` | `False` | Permute column order in prompts |
| `--verbose` | `False` | Detailed logging |

### vLLM-Specific Parameters

| Parameter | Default | Description |
|---|---|---|
| `--tensor-parallel` | `1` | GPUs for tensor parallelism |
| `--max-context-len` | auto | Context window cap (input + output) |

### TensorRT-Specific Parameters

| Parameter | Default | Description |
|---|---|---|
| `--max-input-len` | `16384` | Engine input length limit |
| `--max-output-len` | `16384` | Engine output length limit |

### Remote vLLM Parameters

| Parameter | Default | Description |
|---|---|---|
| `--server-url` | `$VLLM_SERVER_URL` | Remote vLLM server URL |
| `--api-key` | `$VLLM_API_KEY` | Bearer token for auth |
| `--concurrent-requests` | `300` | Max in-flight HTTP requests |
| `--request-timeout` | `600` | Per-request timeout (seconds) |
| `--enable-thinking` | `False` | Enable reasoning mode (remote-vllm only). Reasoning in `message.reasoning_content`, output in `message.content`. Enforces 65 536 token floor. |
| `--disable-lang-fuse` | `False` | Disable Langfuse tracing even when env vars are set (alias: `--disable-langfuse`) |

---

## Observability with Langfuse

Both the **remote-vllm** and **local vllm** backends have optional
[Langfuse](https://langfuse.com) integration for full-pipeline observability.
When enabled, each experiment run creates traces with the following hierarchy:

```
Session: {experiment_name}
└─ profiling (span)                 -- schema inference & statistics
└─ tabgen-generate (span)           -- one per generate() call
     ├─ batch-1 (span)              -- one per batch iteration
     │   ├─ llm-generation (generation)  -- one per LLM request
     │   ├─ llm-generation (generation)
     │   ├─ decoding (span)         -- decode results & success rate
     │   └─ ...
     ├─ batch-2 (span)
     │   └─ ...
     └─ ...
```

**What is traced:**

| Stage | Data captured |
|-------|---------------|
| **Profiling** | Dataset shape, detected column types, profiling duration |
| **Generation root** | All sampling params, model config, dataset metadata |
| **Batch** | Prompt count, decode rate, timing |
| **LLM generation** | Input prompt, output text, token usage, latency, finish reason |
| **Decoding** | Rows decoded, empty texts, errors, decode success rate |

**Additional features:**

- **Session grouping** — all traces from one experiment are grouped by `session_id`
  (defaults to the experiment name) for easy comparison in the Langfuse UI.
- **Prompt management** — the generation prompt template is registered in Langfuse
  and versioned automatically.  Changes across experiments are tracked.
- **Dataset registration** — the real dataset schema and sample rows are registered
  in Langfuse for reference.
- **Score attachment** — evaluation metrics can be pushed back to generation traces:
  - **Programmatically** via `tabgen.push_evaluation_scores({"f1": 0.87})`.
  - **Automatically from `ml_eval.py`** — when a generation config file with a
    `last_trace_id` exists (saved by `tensorrt_sampling_optimized.py`), running
    `ml_eval.py` will push per-model ML scores (e.g. `ml_eval/XGBoost`,
    `ml_eval/Average`) to the corresponding Langfuse trace.
- **Trace URL logging** — the Langfuse UI URL for each trace is printed after
  generation for quick inspection.

### Setup

1. Install Langfuse (included in `requirements_vllm.txt`, or install separately):
   ```bash
   pip install "langfuse>=4.0.0"
   # or via extras:
   pip install -e ".[observability]"
   ```

2. Set environment variables in `.env`:
   ```dotenv
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_BASE_URL=http://your-langfuse-host:3000
   ```

3. Run as usual -- tracing is automatic:
   ```bash
   python tensorrt_sampling_optimized.py \
     --backend remote-vllm \
     --server-url http://my-server:8000 \
     --model Qwen/Qwen3.5-9B \
     --dataset data.csv --experiment my_exp \
     --target-column target --k-shots 40 --n-rows 100
   ```

### Disabling Langfuse

Langfuse is **enabled by default** when the environment variables are set.
To disable it without removing env vars:

```bash
# During generation:
python tensorrt_sampling_optimized.py \
  --backend remote-vllm --disable-lang-fuse \
  ...

# During evaluation:
python ml_eval.py --disable-lang-fuse \
  --dataset data.csv --experiments my_exp \
  --synthetic-datasets real synthetic_llm_40_shots \
  --target-column target
```

### Fail-safe Behaviour

The integration is fully fail-safe:

- **`langfuse` not installed** -- tracing is silently disabled; no import error.
- **Environment variables missing** -- tracing is silently disabled.
- **Langfuse server unreachable** -- a warning is logged; generation continues normally.
- **Any tracing call fails mid-generation** -- the error is caught and logged at
  DEBUG level; generation is never interrupted.

---

## How the Prompt Works

Each prompt sent to the model contains:

1. **Schema & Statistics** -- column types, min/max/mean/std, quantiles
2. **Correlation Matrix** -- numerical feature correlations (if `--use-correlation-matrix`)
3. **K-shot examples** -- `k` stratified-sampled real rows, encoded as JSON objects (default)
   or GReaT-style `column is value` rows (`--encoding-format predictive`)
4. **Generation instructions** -- format-specific output instructions

With the default JSON format, the model generates structured JSON objects that are
validated via guided decoding on vLLM backends, resulting in higher decode success
rates. The legacy `predictive` format uses numbered `col is val` lines.

The model generates `batch_size` new rows, which are parsed back into a DataFrame
by the decoder.

---

## Evaluating Synthetic Data

### ML Evaluation

```bash
python ml_eval.py \
  --dataset adult.csv \
  --experiments experiment_A2 experiment_B2 \
  --synthetic-datasets real ctgan tvae synthetic_llm_40_shots synthetic_llm_80_shots \
  --target-column income --task-type classification \
  --train-size 5000 --test-size 1000
```

Results are saved to `experiments/{experiment}/evaluation_reports/`.

### Standalone Evaluator

The [`synthetic_data_evaluator`](https://github.com/sordi-ai/Synthetic-Data-Generation-Evaluator)
package — developed alongside this project and distributed separately — provides
a CLI and Python API for computing quality metrics (pMSE, TabSynDex, Hellinger,
KS, DCR, disclosure protection, ML efficacy). It is also required by the
optional MCP server below.

```bash
git clone https://github.com/sordi-ai/Synthetic-Data-Generation-Evaluator.git
pip install -e Synthetic-Data-Generation-Evaluator/synthetic_data_evaluator/
synth-eval evaluate --real real.csv --synth synthetic.csv --report Similarity
```

### MCP Server

An optional MCP server exposes evaluation tools to AI assistants. See
[mcp_server/README.md](mcp_server/README.md).

---

## Plotting Results

```bash
python plot_evaluation_graphs.py \
  --dataset adult.csv \
  --experiments experiment_C2 \
  --synthetic-datasets ctgan tvae synthetic_llm_40_shots synthetic_llm_80_shots
```

Figures are saved to `experiments/{experiment}/figures/`.

---

## Supported Models

### Tested and Recommended

| Model | Backends |
|---|---|
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | tensorrt, vllm, remote-vllm |
| `Qwen/Qwen3.5-122B-A10B` | remote-vllm |
| `Qwen/Qwen3-4B-Instruct-2507` | tensorrt, vllm |
| `Qwen/Qwen2.5-14B-Instruct` | vllm |
| `Qwen/Qwen2.5-32B-Instruct` | vllm |
| `Qwen/Qwen3.5-9B-Instruct` | vllm |
| `Qwen/Qwen3.5-30B-A3B-FP8` | vllm |
| `Qwen/Qwen3-Coder-30B-A3B-FP8` | vllm |

### Also Supported (vLLM)

Any HuggingFace model works with the vLLM and remote-vllm backends.
Additionally tested:
- `google/gemma-3-27b-it`, `google/gemma-2-27b-it`, `google/gemma-2-9b-it`
- `meta-llama/Meta-Llama-3.1-70B-Instruct`, `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

### TensorRT-LLM Supported Models

TensorRT requires model-specific engine builds. Tested:
- `Qwen/Qwen3-30B-A3B-Instruct-2507`
- `nvidia/Qwen3-30B-A3B-FP4`
- `google/gemma-3-27b-it`, `google/gemma-2-27b-it`
- `meta-llama/Meta-Llama-3.1-70B-Instruct`, `meta-llama/Meta-Llama-3.1-8B-Instruct`

---

## Datasets

The paper evaluates FewShotTabLLM on five public benchmark datasets (not
bundled with this repository — download them and pass the CSV via `--dataset`):

| Dataset | Task | Target column | Source |
|---|---|---|---|
| Adult | classification | `income` | [UCI ML Repository](https://doi.org/10.24432/C5XW20) |
| Buddy | classification | `pet_category` | [Kaggle](https://www.kaggle.com/datasets/akash14/adopt-a-buddy) |
| Child | classification | `Sick` | CHILD Bayesian network benchmark (Spiegelhalter et al., 1993) |
| Diabetes | classification | `Outcome` | [Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set) |
| Housing | regression | `median_house_value` | [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices) |

Baselines compared in the paper: CTGAN, TVAE, TabDDPM, and Be-GReaT.

---

## Project Structure

```
tensorrt_sampling_optimized.py       # Main CLI entry-point (all backends)
ml_eval.py                           # ML evaluation script
plot_evaluation_graphs.py            # Plotting script

flash_tabgen_tensorrt/               # Core library package
  core/
    data_profiler.py                 # Dataset profiling & schema inference
    encoding.py                      # Row encoding (json/predictive)
    prompts.py                       # Few-shot prompt construction
    decoder.py                       # LLM output -> DataFrame parsing
    generator_vllm.py                # Local vLLM inference engine
    generator_tensorrt.py            # TensorRT-LLM inference engine
    generator_remote_vllm.py         # Remote HTTP inference (async/concurrent)
    tabgen_vllm.py                   # High-level TabGen class (local vLLM)
    tabgen_tensorrt.py               # High-level TabGen class (TensorRT)
    tabgen_remote_vllm.py            # High-level TabGen class (remote vLLM)
    langfuse_utils.py                # Langfuse observability integration

services/                            # Shared utilities
  parsers.py                         # CLI argument parsers
  sampling.py                        # Representative sample selection
  logger_config.py                   # Logging setup
  data_handler.py                    # Data loading helpers
  plot_data_statistics.py            # Data statistics visualizations
  plot_evaluation_scores.py          # Evaluation score plotting

mcp_server/                          # MCP server for AI assistant integration
  mcp_server.py                      # FastMCP tool definitions
  core/                              # Server logic, schemas, validation
  synthesizer/                       # CTGAN train/generate
  tests/                             # MCP server tests
```

---

## Development

```bash
# Format
black .

# Lint
ruff check .
ruff check --fix .

# Type check
mypy flash_tabgen_tensorrt/ services/

# Test
pytest
pytest --cov=flash_tabgen_tensorrt --cov=services --cov-report=term-missing
```

---

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{koubeissy2026fewshottabllm,
  title     = {Few-Shot LLMs as Synthetic Tabular Data Generators},
  author    = {Koubeissy, Hadi and El Khoury, Michel and Kamradt, Marc and Makhoul, Abdallah},
  booktitle = {IEEE Annual Computers, Software, and Applications Conference (COMPSAC)},
  year      = {2026}
}
```
