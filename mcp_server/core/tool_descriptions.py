"""
DEscription for MCP Tools
"""

LIST_METRICS_DESC = (
    "Return the list of available synthetic data evaluation metrics. "
    "Use this tool by default whenever the user asks to run an evaluation "
    "and the name is not clearly one of the 6 report names "
    "(Similarity, Privacy, Nearest_Neighbor, Machine_Learning_Efficacy, "
    "Likelihood_Fitness, Distance, Correlation). "
    "Most evaluation requests (e.g. 'run disclosure protection', "
    "'run range coverage') are metrics, not reports."
)

LIST_REPORTS_DESC = (
    "Return the list of available synthetic data evaluation reports. "
    "There are exactly 6 reports: Similarity, Privacy, Nearest_Neighbor, "
    "Machine_Learning_Efficacy, Likelihood_Fitness, Distance, Correlation. "
    "Only call this tool when the user explicitly asks about reports or "
    "mentions one of these 6 names. Do NOT call this tool to look up metrics."
)

LIST_ML_MODELS_DESC = (
    "Return the available ML models for classification and regression."
)

GET_METRIC_PARAMS_DESC = (
    "Return the parameter schema for a specific evaluation metric. "
    "Use this before calling evaluate_metric. "
    "If the response includes required_parameters, ask the user for "
    "those values before running the metric. "
    "Additionally, present the available optional parameters to the user "
    "and ask if they would like to customize any of them, explaining "
    "what each one does. Do not skip optional parameters silently."
)

GET_REPORT_PARAMS_DESC = (
    "Return the parameter schema for a specific evaluation report. "
    "Report names are case-sensitive. "
    "Use this before calling evaluate_report. "
    "If required_parameters are present, collect them from the user first. "
    "Additionally, present the available optional parameters to the user "
    "and ask if they would like to customize any of them, explaining "
    "what each one does. Do not skip optional parameters silently."
)

GET_ML_EVAL_PARAMS_DESC = (
    "Return the parameter schema for ML evaluation. "
    "'target', 'task_type', and 'train_source' are all required. "
    "Ask the user which column to predict, whether the task is "
    "'classification' or 'regression', and whether to train on "
    "'real' or 'synthetic' data before calling ml_eval. "
    "Do NOT assume or default train_source to 'synthetic'. "
    "Additionally, present the available optional parameters "
    "(e.g. learning_rate, n_estimators, max_depth, test_split, etc.) "
    "to the user and ask if they would like to customize any of them. "
    "Do not skip optional parameters silently."
    "Only call this tool when the user explicitly asks for machine learning evaluation, nothing else."
)

GET_METADATA_DESC = (
    "Generate metadata for one or more CSV datasets. "
    "This tool is optional and intended for inspection or debugging. "
    "Evaluation tools automatically generate metadata if none is provided, "
    "You never need this tool for evaluation."
)

EVALUATE_METRIC_DESC = (
    "Run a single synthetic data evaluation metric. "
    "Before calling this tool, call get_evaluation_metric_parameters "
    "to determine required parameters. "
    "Do not call this tool until all required parameters have been "
    "collected from the user. "
    "Do not infer or guess required values."
)

EVALUATE_REPORT_DESC = (
    "Run a synthetic data evaluation report (a collection of metrics). "
    "Before calling this tool, call get_report_parameters to determine "
    "required parameters. "
    "Collect all required values from the user before execution. "
    "Do not infer missing parameters."
)

ML_EVAL_DESC = (
    "Train ML models on either synthetic data or real data and test them on real data. "
    "Required inputs: "
    "- target: the column to predict "
    "- task_type: 'classification' or 'regression' "
    "- train_source: 'real' or 'synthetic' (wether the user wants to train the models on real or synthetic data)"
    "Ask the user to specify these before calling this tool. "
    "Do not guess the target column or task type."
    "Only use this tool when the user explicitly asks for machine learning evaluation"
    "Always ask the user if they want train on real or synthetic data (task_type = 'real'|'synthetic'). If "
    "If they want to train on synthetic data, make sure to ask them to provide the synthetic data if they havent otherwise"
)

INSPECT_DATA_DESC = (
    "Inspect a real dataset to understand its structure, columns, types, "
    "missing values, and class distributions. "
    "Use this tool when: "
    "- The user needs help choosing a target column, task type, or any other evaluation parameter "
    "- The user asks what columns are in the data or wants to explore the dataset "
    "- You need context about the data before running an evaluation or ML task "
    "- The user asks about data quality, missing values, or column statistics. "
    "This tool only requires the real data path. Optionally pass a target column "
    "to get its class distribution."
    "After using this tool for introspection about choosing parameters, share your newly gained context with the user before finalizing parameters"
)

# ── Synthesizer tools ─────────────────────────────────────────────────────

GET_TRAIN_PARAMS_DESC = (
    "Return the parameter schema for training a CTGAN synthesizer. "
    "Always Use this before calling train_synthesizer to understand which "
    "hyper-parameters are available (epochs, batch_size, learning rates, etc.). "
    "Only real_data_path and model_name are required; "
    "all other hyper-parameters have sensible defaults. "
    "After collecting the required values, present the full list of "
    "optional hyper-parameters (epochs, batch_size, learning rates, "
    "architecture dimensions, etc.) to the user and ask if they would "
    "like to customize any of them. Do not skip optional parameters silently."
)

GET_GENERATE_PARAMS_DESC = (
    "Return the parameter schema for generating synthetic data. "
    "Always Use this before calling generate_synthetic_data to understand "
    "the required and optional parameters. "
    "After collecting the required values (model_name, num_rows), "
    "present the available optional parameters (output_dir, "
    "output_filename, models_dir) to the user and ask if they would "
    "like to customize any of them. Do not skip optional parameters silently."
)

TRAIN_SYNTHESIZER_DESC = (
    "Train a CTGAN synthesizer model on a real CSV dataset. "
    "Required inputs: "
    "- real_data_path: absolute path to the training CSV "
    "- model_name: a name for the model (saved as <name>.pkl) "
    "All CTGAN hyper-parameters (epochs, batch_size, learning rates, "
    "architecture dimensions, etc.) are optional with sensible defaults. "
    "The trained model is saved to disk and can be used later with "
    "generate_synthetic_data. "
    "Ask the user for real_data_path and model_name before calling this tool. "
    "Always ask the user if they would like to customize optional train parameters before training"
    "Do not guess the data path."
)

GENERATE_SYNTHETIC_DATA_DESC = (
    "Generate synthetic data from a previously trained CTGAN model. "
    "Required inputs: "
    "- model_name: name of the trained model "
    "- num_rows: number of synthetic rows to generate "
    "The generated CSV is saved to disk. "
    "Ask the user for both values before calling this tool. "
    "If the user hasn't trained a model yet, suggest using "
    "train_synthesizer first."
)

LIST_TRAINED_MODELS_DESC = (
    "List all trained synthesizer models available on disk. "
    "Use this when the user wants to know which models they can "
    "generate data from, or to check if a model exists."
)

# ── Visualization tools ───────────────────────────────────────────────────

PLOT_DISTRIBUTIONS_DESC = (
    "Generate per-column distribution comparison plots (real vs synthetic). "
    "Numerical columns are shown as overlaid density histograms; "
    "categorical columns are shown as grouped bar charts. "
    "Returns saved image paths and optionally base64-encoded PNGs. "
    "Use this to visually compare how well synthetic data matches "
    "the real data distributions."
)