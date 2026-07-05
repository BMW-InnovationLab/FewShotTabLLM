import json
import os
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, BayesianRidge, LinearRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


def generate_metadata(df, dataset_name):
    """
    Generate SDV-style metadata.json from a DataFrame.

    Infers column types as 'numerical' (int/float) or 'categorical' (object/bool)
    and returns a dict compatible with the ``metadata["tables"][name]["columns"]``
    structure used throughout ``data_handler``.

    Args:
        df: The DataFrame to infer metadata from.
        dataset_name: Name used as the table key in the metadata dict.

    Returns:
        A metadata dict in SDV single-table format.
    """
    columns = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            columns[col] = {"sdtype": "numerical"}
        else:
            columns[col] = {"sdtype": "categorical"}

    return {
        "tables": {
            dataset_name: {
                "columns": columns,
            }
        }
    }


def ensure_metadata(dataset_dir, dataset_name, df):
    """
    Ensure ``metadata.json`` exists in *dataset_dir*.

    If the file is missing it is auto-generated from *df* using
    :func:`generate_metadata` and written to disk.

    Args:
        dataset_dir: Directory where ``metadata.json`` should reside.
        dataset_name: Table name key for the metadata dict.
        df: DataFrame to infer column types from.

    Returns:
        The loaded (or generated) metadata dict.
    """
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            return json.load(f)

    metadata = generate_metadata(df, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Auto-generated {metadata_path}")
    return metadata


def load_dataset(dataset_name, synthesizer_name, part):
    dataset_path = f"{part}/datasets/{dataset_name}"
    real = pd.read_csv(os.path.join(dataset_path, f"{synthesizer_name}.csv"))
    test = pd.read_csv(os.path.join(dataset_path, f"{dataset_name}_test.csv"))
    metadata = json.load(open(os.path.join(dataset_path, f"metadata.json"), "r"))
    train_size = real.shape[0]
    return real, test, metadata, train_size, dataset_path


def process_data_for_ml(data, test, dataset_name, metadata, target_col, date_cols, id_col=None):
    """
    Complete pipeline to clean IDs, engineer date features, handle missing values,
    and OneHotEncode categoricals for both train and test data.

    Args:
        data: Training DataFrame
        test: Test DataFrame
        dataset_name: Name of the dataset
        metadata: Metadata dictionary with column information
        target_col: Target column name
        id_col: ID column to drop (optional)

    Returns:
        data, test, y, y_test: Processed DataFrames
    """
    columns = metadata["tables"][dataset_name]["columns"]

    # Extract target before processing
    y = pd.DataFrame(data[target_col], columns=[target_col])
    y_test = pd.DataFrame(test[target_col], columns=[target_col])

    data = data.drop(columns=[target_col]).copy(deep=True)
    test = test.drop(columns=[target_col]).copy(deep=True)

    # --- A. DROP ID COLUMN ---
    if id_col and id_col in data.columns:
        print(f"Dropping ID column: {id_col}")
        data = data.drop(columns=[id_col])
        if id_col in test.columns:
            test = test.drop(columns=[id_col])

    if date_cols:
        print(f"Processing datetime columns: {date_cols}")

        # Convert to datetime
        for col in date_cols:
            data[col] = pd.to_datetime(data[col], errors="coerce")
            test[col] = pd.to_datetime(test[col], errors="coerce")

        # If we have 2+ date columns, calculate duration between first two
        if len(date_cols) >= 2:
            data["duration_days"] = (data[date_cols[1]] - data[date_cols[0]]).dt.total_seconds() / (
                3600 * 24
            )
            test["duration_days"] = (test[date_cols[1]] - test[date_cols[0]]).dt.total_seconds() / (
                3600 * 24
            )

            # Fill NaN durations with 0
            data["duration_days"] = data["duration_days"].fillna(0)
            test["duration_days"] = test["duration_days"].fillna(0)

        # Extract cyclical features from the first date column
        primary_date_train = data[date_cols[0]]
        primary_date_test = test[date_cols[0]]

        # Month (1-12) - cyclical encoding
        data["date_month_sin"] = np.sin(2 * np.pi * primary_date_train.dt.month.fillna(1) / 12)
        data["date_month_cos"] = np.cos(2 * np.pi * primary_date_train.dt.month.fillna(1) / 12)
        test["date_month_sin"] = np.sin(2 * np.pi * primary_date_test.dt.month.fillna(1) / 12)
        test["date_month_cos"] = np.cos(2 * np.pi * primary_date_test.dt.month.fillna(1) / 12)

        # Hour (0-23) - cyclical encoding
        data["date_hour_sin"] = np.sin(2 * np.pi * primary_date_train.dt.hour.fillna(0) / 24)
        data["date_hour_cos"] = np.cos(2 * np.pi * primary_date_train.dt.hour.fillna(0) / 24)
        test["date_hour_sin"] = np.sin(2 * np.pi * primary_date_test.dt.hour.fillna(0) / 24)
        test["date_hour_cos"] = np.cos(2 * np.pi * primary_date_test.dt.hour.fillna(0) / 24)

        # Drop original datetime columns
        data = data.drop(columns=date_cols)
        test = test.drop(columns=date_cols)

    # --- D. HANDLE MISSING VALUES ---
    # Numerical columns
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        try:
            imputer = SimpleImputer(strategy="mean")
            data[num_cols] = imputer.fit_transform(data[num_cols])
            test[num_cols] = imputer.transform(test[num_cols])
        except:
            data[num_cols] = data[num_cols].fillna(0)
            test[num_cols] = test[num_cols].fillna(0)

    # Categorical columns - fill with 'none'
    cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        data[cat_cols] = data[cat_cols].fillna("none")
        test[cat_cols] = test[cat_cols].fillna("none")

        # Convert boolean values to strings for categorical columns
        for col in cat_cols:
            if data[col].dtype == "bool":
                data[col] = data[col].astype(str)
            if test[col].dtype == "bool":
                test[col] = test[col].astype(str)

    return data, test, y, y_test


def split_real_dataset(data, dataset_name, train_size, test_size, random_state=42):

    total_needed = train_size + test_size

    # If dataset is larger than needed, sample first
    if len(data) > total_needed:
        data = data.sample(n=total_needed, random_state=random_state).reset_index(drop=True)
        test_proportion = test_size / total_needed
    elif len(data) < total_needed:
        print(
            f"Warning: Dataset has {len(data)} rows but {total_needed} requested. Using all available data."
        )
        total_needed = len(data)
        test_proportion = 0.2
    else:
        test_proportion = test_size / total_needed

    train_df, test_df = train_test_split(data, test_size=test_proportion, random_state=random_state)

    train_df = train_df.head(train_size).reset_index(drop=True)
    test_df = test_df.head(test_size).reset_index(drop=True)

    dataset_dir = f"ML_EVALUATIONS/datasets/{dataset_name}"
    os.makedirs(dataset_dir, exist_ok=True)

    train_path = f"{dataset_dir}/{dataset_name}_real.csv"
    test_path = f"{dataset_dir}/{dataset_name}_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Also save as "real.csv" so load_dataset(dataset_name, "real", ...) works
    real_path = f"{dataset_dir}/real.csv"
    train_df.to_csv(real_path, index=False)

    # Auto-generate metadata.json if it does not exist
    ensure_metadata(dataset_dir, dataset_name, data)

    return train_df, test_df


def concat_df(data, test):
    results_df = pd.concat([data, test], axis=0)
    return results_df


def encode_datasets(results_df, y_column):
    cat_cols = results_df.select_dtypes(include=["object"]).columns.tolist()
    if y_column in cat_cols:
        cat_cols.remove(y_column)

    # Convert any boolean columns to strings to ensure uniform input for encoder
    for col in cat_cols:
        if results_df[col].dtype == "bool":
            results_df[col] = results_df[col].astype(str)

    results_df.reset_index(inplace=True, drop=True)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    one_hot_encoded = encoder.fit_transform(results_df[cat_cols])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(cat_cols))
    results_df = pd.concat([results_df, one_hot_df], axis=1)
    results_df = results_df.drop(cat_cols, axis=1)

    return results_df


def split_datasets(results_df, y, y_test, y_column, train_size):
    data = results_df[:train_size]
    test = results_df[train_size:]

    y = y[y_column]
    y_test = y_test[y_column]

    return data, test, y, y_test


def encode_y(y, y_test, dataset_name, metadata, y_column):
    columns = metadata["tables"][dataset_name]["columns"]
    if columns[y_column]["sdtype"] == "categorical":
        le = LabelEncoder()
        y = le.fit_transform(y)
        y_test = le.fit_transform(y_test)
    return y, y_test


def convert_to_np(data, test, y, y_test):
    X_train = data.to_numpy()
    X_test = test.to_numpy()
    y_train = y
    y_test = y_test

    return X_train, X_test, y_train, y_test


def init_models(task_type):
    models = (
        {
            "SVM": SVC(),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(),
            "MultinomialLogisticRegression": LogisticRegression(solver="lbfgs"),
            "MLPClassifier": MLPClassifier(),
        }
        if task_type == "classification"
        else {
            "XGBRegressor": XGBRegressor(),
            "Ridge": Ridge(),
            "BayesianRidge": BayesianRidge(),
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(),
        }
    )
    return models


def train_predict(
    X_train,
    X_test,
    y_train,
    y_test,
    y_column,
    models,
    dataset_name,
    synthesizer_name,
    task_type,
    part,
    metadata,
):
    columns = metadata["tables"][dataset_name]["columns"]
    model_results = []
    model_names = []

    for name, model in models.items():
        a = model.fit(X_train, y_train)
        predicted = a.predict(X_test)
        if task_type == "regression" and columns[y_column]["sdtype"] == "categorical":
            predicted_int = []
            for pred in predicted:
                predicted_int.append(int(pred))
        score = (
            f1_score(y_test, predicted, average="macro")
            if task_type == "classification"
            else r2_score(y_test, predicted)
        )
        model_results.append(score)
        model_names.append(name)

    avg = sum(model_results) / len(model_results)
    model_results.append(avg)
    model_names.append("Average")

    results = pd.DataFrame([model_names, model_results])
    results = results.transpose()
    results = results.rename(columns={0: "Model", 1: "Score"})

    save_path = os.path.join(part, "results", dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save with synthesizer_name only (not prefixed with dataset_name)
    filename = f"{synthesizer_name}.csv"
    print(f"Saving to {save_path}/{filename}")
    results.to_csv(os.path.join(save_path, filename))
