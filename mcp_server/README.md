# Synthetic Data Evaluator — MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server that exposes the Synthetic Data Evaluator as a set of tools for LLM agents and AI assistants.

Built with [FastMCP](https://github.com/jlowin/fastmcp). Runs over HTTP on `http://0.0.0.0:8001` by default.

---
### Run the server

Requires the [`synthetic_data_evaluator`](https://github.com/sordi-ai/Synthetic-Data-Generation-Evaluator)
package, which is maintained in a separate repository. From the repository root:

```bash
python3 -m venv .evaluate_venv
source .evaluate_venv/bin/activate
pip install uv
uv pip install -r requirements_vllm.txt

# Install the evaluator package
git clone https://github.com/sordi-ai/Synthetic-Data-Generation-Evaluator.git
pip install -e Synthetic-Data-Generation-Evaluator/synthetic_data_evaluator/

python3 -m mcp_server.mcp_server
```

The server starts on `http://0.0.0.0:8001` and exposes 11 tools that any MCP-compatible client can call.

### Connecting to an AI assistant

Add the server to your MCP client configuration. For example, in VS Code / Copilot:

```json
{
  "servers": {
    "synthetic-data-evaluator": {
      "type": "http",
      "url": "http://127.0.0.1:8001"
    }
  }
}
```

---

## Available Tools

### Listing

| Tool | Description |
|---|---|
| `list_evaluation_metrics` | Returns all available evaluation metric names |
| `list_evaluation_reports` | Returns all report categories and the metrics within each |
| `list_ml_evaluation_models` | Returns available ML models for classification and regression |

### Parameter Introspection

| Tool | Description |
|---|---|
| `get_evaluation_metric_parameters` | Returns the parameter schema for a specific metric (required vs optional) |
| `get_report_parameters` | Returns the parameter schema for a specific report category |
| `get_ml_evaluation_parameters` | Returns the parameter schema for ML evaluation |

### Data Inspection

| Tool | Description |
|---|---|
| `inspect_data` | Profiles a CSV file — returns overview stats, per-column details, notable correlations, and optional target analysis to help choose evaluation parameters |
| `get_dataset_metadata` | Auto-generates column metadata from one or more CSV files. Optional — evaluation tools generate metadata automatically if none is provided. |

### Evaluation

| Tool | Description |
|---|---|
| `evaluate_using_single_metric` | Runs a single evaluation metric on real vs synthetic data |
| `evaluate_using_report` | Runs a full report category (collection of metrics) on real vs synthetic data |

### ML Evaluation

| Tool | Description |
|---|---|
| `ml_eval` | Trains ML models on real or synthetic data and evaluates them on held-out real data |

---

## Tool Reference

### `inspect_data`

Profiles a dataset before evaluation. Useful for discovering column names, types, missing values, and choosing a target column.

```
real_data_path   (str, required)    — Absolute path to real data CSV
target           (str, optional)    — Column to analyse as a prediction target
```

**Returns:**
- `overview` — row/column counts, memory usage, duplicate rows, total missing values
- `columns` — per-column `pandas_dtype`, `sdtype`, unique count, missing stats, and distribution/stats depending on type
- `type_summary` — count of numerical / categorical / boolean / datetime columns
- `notable_correlations` — pairs of numerical columns with |r| > 0.5
- `target_analysis` *(when `target` is given)* — suggested task type, class distribution (classification) or descriptive stats (regression), imbalance warnings

### `evaluate_using_single_metric`

```
metric           (str, required)    — Metric name (e.g. "Boundary_Adherence")
real_data_path   (str, required)    — Absolute path to real data CSV
synth_data_path  (str, optional)    — Absolute path to synthetic data CSV
params           (dict, required)   — Evaluation parameters (can be {} for defaults)
metadata         (dict, optional)   — Dataset metadata; auto-generated if omitted
columns          (list, optional)   — Subset of columns to evaluate
table_names      (list, optional)   — Table names for multi-table evaluation
primary_key      (str, optional)    — Primary key column (multi-table)
foreign_key      (str, optional)    — Foreign key column (multi-table)
```

### `evaluate_using_report`

Same signature as `evaluate_using_single_metric` but with `report` instead of `metric` (e.g. `"Similarity"`, `"Distance"`, `"Privacy"`).

### `ml_eval`

```
real_data_path   (str, required)               — Path to real data CSV
synth_data_path  (str, optional)               — Path to synthetic data CSV (required when train_source="synthetic")
target           (str, required)               — Column to predict
task_type        (str, default "classification") — "classification" or "regression"
train_source     (str, default "real")         — "synthetic" (train on synth, test on real) or "real" (train on real, test on real)
train_size       (int, optional)               — Max rows for training
test_size        (int, optional)               — Max real rows for testing
test_split       (float, default 0.2)          — Fraction of real data held out for testing
id_column        (str, optional)               — ID column to drop before training
date_columns     (list, optional)              — Date columns for cyclical feature engineering
random_state     (int, default 42)             — Random seed for reproducibility
learning_rate    (float, default 0.01)         — Learning rate for models that support it
n_estimators     (int, default 100)            — Estimators for ensemble models
max_depth        (int, default 10)             — Max tree depth for tree-based models
C                (float, default 1.0)          — Regularization for SVM / LogisticRegression
kernel           (str, default "rbf")          — Kernel type for SVM
max_iter         (int, default 200)            — Max iterations for optimization solvers
```

---

## Recommended Workflow

The tool descriptions guide LLM agents through a safe workflow:

1. **Inspect** *(optional)* — Call `inspect_data` on the user's CSV to understand column names, types, and distributions. Use this to help pick a target column or evaluation parameters.
2. **List** — Call `list_evaluation_metrics` or `list_evaluation_reports` to discover what's available.
3. **Inspect params** — Call `get_evaluation_metric_parameters` / `get_report_parameters` to see required and optional parameters. If any parameters are required, **ask the user** before proceeding.
4. **Run** — Call `evaluate_using_single_metric` or `evaluate_using_report` with the collected parameters.

This prevents the agent from guessing column names for privacy fields, targets, etc.

---

## Architecture

```
project_root/
└── mcp_server/
    ├── mcp_server.py           ← Entry point: python3 -m mcp_server.mcp_server
    └── core/
        ├── server.py           ← Tool definitions (@mcp.tool decorators) — 11 tools
        ├── tool_descriptions.py← Human-readable descriptions for each tool (fed to the LLM)
        ├── services.py         ← Business logic (no FastMCP types)
        ├── loaders.py          ← Data loading wrappers (CSV paths → DataFrames + metadata)
        ├── validation.py       ← Required-parameter registries + validators
        └── schemas.py          ← success() / failure() response wrappers
```

### Layer responsibilities

- **server.py** — Defines the 11 MCP tools with type hints and descriptions. Each tool is a thin wrapper that delegates to `services.py` and catches exceptions.
- **services.py** — Pure business logic. Calls into `synthetic_data_evaluator` for evaluation, ML eval, metadata generation, and parameter introspection. Also implements `inspect_data` (dataset profiling). Returns plain dicts.
- **validation.py** — `METRIC_REQUIRED_PARAMS` (auto-generated from `param_registry`) and `REPORT_REQUIRED_PARAMS` registries. Validators ensure required parameters are present before running an evaluation, returning clear error messages so the agent knows exactly what to ask the user for.
- **loaders.py** — Wraps `load_csv_data()`, `prepare_data()`, and `generate_metadata()` from the evaluator package so the service layer works with file paths directly.
- **schemas.py** — `success(data)` and `failure(message)` helpers that wrap responses in a consistent envelope.

### Response format

All tools return a dict:

```json
// Success
{"success": true, "data": { ... }, "error": null}

// Failure
{"success": false, "data": null, "error": "Something went wrong"}
```

---

## Configuration

The server binds to `0.0.0.0:8001` by default. To change the host or port, edit the entry point:

```python
# mcp_server.py
mcp.run(transport="http", host="0.0.0.0", port=9000)
```

FastMCP also supports `stdio` transport for local pipe-based integrations:

```python
mcp.run(transport="stdio")
```

---

## Validation & Required Parameters

Some metrics require specific parameters. The validation layer checks these before running an evaluation and returns a descriptive error if anything is missing:

| Metric / Report Group | Required Parameters |
|---|---|
| CAP metrics (`Categorical_CAP`, `Categorical_Zero_CAP`, `Categorical_Generalized_CAP`) | `key_fields`, `sensitive_fields` |
| Privacy inference (`Numerical_MLP/LR/SVR/RadiusNearestNeighbor`, `Categorical_KNN/NB/RF`) | `key_fields`, `sensitive_fields` |
| `Disclosure_Protection`, `Disclosure_Protection_Estimate` | `key_fields`, `sensitive_fields` |
| `DCR_Baseline_Protection`, `DCR_Overfitting_Protection` | `key_fields`, `sensitive_fields` |
| Privacy report | `key_fields`, `sensitive_fields` |
| ML Efficacy metrics (`Binary_*`, `MultiClass_*`, `Linear_Regression`, `MLP_Regressor`, `LSTM_Classifier`) | `target` |
| Machine_Learning_Efficacy report | `targets` (list) |
| Multi-table (`Cardinality_*`, `Referential_Integrity`) | `table_names`, `primary_key`, `foreign_key` |
| `Key_Uniqueness` | `primary_key` |
| Sequential (`LSTM_Detection`, `LSTM_Classifier`) | `sequence_key` |
| `ml_eval` | `target`, `task_type`, `train_source` |

The `get_evaluation_metric_parameters` and `get_report_parameters` tools surface these requirements so the agent knows exactly what to collect from the user before calling an evaluation tool.
