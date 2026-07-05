"""
Service layer for synthesizer MCP tools.

Handles data loading, metadata detection, parameter conversion,
and delegates to CTGANTrainer / CTGANGenerator.
"""

import os
import glob
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sdv.metadata import SingleTableMetadata

from ..synthesizer.train import CTGANTrainer
from ..synthesizer.generate import CTGANGenerator


# ── Defaults ──────────────────────────────────────────────────────────────

_DEFAULT_MODELS_DIR = os.path.join(os.getcwd(), "models")
_DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "synthetic_output")


# ── CTGAN parameter info (for introspection tools) ───────────────────────

CTGAN_PARAM_SCHEMA: Dict[str, Dict[str, Any]] = {
    "enforce_min_max_values": {
        "type": "bool",
        "default": True,
        "description": "Clip reverse-transformed numerical values to the min/max seen during fit.",
    },
    "enforce_rounding": {
        "type": "bool",
        "default": True,
        "description": "Round numerical columns to match original data precision.",
    },
    "locales": {
        "type": "list[str] | str",
        "default": '["en_US"]',
        "description": "Locale(s) for AnonymizedFaker transformers.",
    },
    "embedding_dim": {
        "type": "int",
        "default": 128,
        "description": "Size of the noise vector fed to the Generator.",
    },
    "generator_dim": {
        "type": "list[int]",
        "default": [256, 256],
        "description": "Residual layer sizes in the Generator (passed as a list, e.g. [256, 256]).",
    },
    "discriminator_dim": {
        "type": "list[int]",
        "default": [256, 256],
        "description": "Linear layer sizes in the Discriminator (passed as a list, e.g. [256, 256]).",
    },
    "generator_lr": {
        "type": "float",
        "default": 2e-4,
        "description": "Generator learning rate.",
    },
    "generator_decay": {
        "type": "float",
        "default": 1e-6,
        "description": "Generator weight decay (Adam).",
    },
    "discriminator_lr": {
        "type": "float",
        "default": 2e-4,
        "description": "Discriminator learning rate.",
    },
    "discriminator_decay": {
        "type": "float",
        "default": 1e-6,
        "description": "Discriminator weight decay (Adam).",
    },
    "batch_size": {
        "type": "int",
        "default": 500,
        "description": "Samples per training step.",
    },
    "discriminator_steps": {
        "type": "int",
        "default": 1,
        "description": "Discriminator updates per Generator update.",
    },
    "log_frequency": {
        "type": "bool",
        "default": True,
        "description": "Use log-frequency for categorical conditional sampling.",
    },
    "verbose": {
        "type": "bool",
        "default": False,
        "description": "Print training progress.",
    },
    "epochs": {
        "type": "int",
        "default": 300,
        "description": "Number of training epochs.",
    },
    "pac": {
        "type": "int",
        "default": 10,
        "description": "PacGAN grouping size for the Discriminator.",
    },
    "cuda": {
        "type": "bool | str",
        "default": True,
        "description": "Use GPU (true), specific device string, or CPU (false).",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────

def _build_model_path(model_name: str, models_dir: Optional[str] = None) -> str:
    base = models_dir or _DEFAULT_MODELS_DIR
    if not model_name.endswith(".pkl"):
        model_name += ".pkl"
    return os.path.join(base, model_name)


# ── Public service functions ──────────────────────────────────────────────

def get_train_parameters() -> Dict[str, Any]:
    """Return the full parameter schema for train_synthesizer."""
    return {
        "required_parameters": {
            "real_data_path": "Absolute path to the real CSV dataset to train on.",
            "model_name": "Name for the trained model (e.g. 'adult_ctgan'). Saved as <name>.pkl.",
        },
        "optional_parameters": CTGAN_PARAM_SCHEMA,
        "note": (
            "All CTGAN hyper-parameters are optional and have sensible defaults. "
            "Only real_data_path and model_name are required."
        ),
    }


def get_generate_parameters() -> Dict[str, Any]:
    """Return the full parameter schema for generate_synthetic_data."""
    return {
        "required_parameters": {
            "model_name": "Name of a previously trained model (e.g. 'adult_ctgan').",
            "num_rows": "Number of synthetic rows to generate.",
        },
        "optional_parameters": {
            "output_filename": {
                "type": "str",
                "default": "<model_name>_synthetic.csv",
                "description": "Filename for the generated CSV.",
            },
            "output_dir": {
                "type": "str",
                "default": "./synthetic_output",
                "description": "Directory to save the generated CSV.",
            },
        },
    }


def list_trained_models(models_dir: Optional[str] = None) -> Dict[str, Any]:
    """List all .pkl model files in the models directory."""
    base = models_dir or _DEFAULT_MODELS_DIR
    if not os.path.isdir(base):
        return {"models": [], "models_dir": base, "note": "Models directory does not exist yet. Train a model first."}

    files = sorted(glob.glob(os.path.join(base, "*.pkl")))
    models = []
    for f in files:
        stat = os.stat(f)
        models.append({
            "name": os.path.splitext(os.path.basename(f))[0],
            "path": os.path.abspath(f),
            "size_mb": round(stat.st_size / (1024 ** 2), 2),
        })
    return {"models": models, "models_dir": os.path.abspath(base), "total": len(models)}


def train_model(
    real_data_path: str,
    model_name: str,
    models_dir: Optional[str] = None,
    # ── CTGAN params (all optional with defaults) ─────────
    enforce_min_max_values: bool = True,
    enforce_rounding: bool = True,
    locales: Optional[Union[List[str], str]] = None,
    embedding_dim: int = 128,
    generator_dim: Optional[List[int]] = None,
    discriminator_dim: Optional[List[int]] = None,
    generator_lr: float = 2e-4,
    generator_decay: float = 1e-6,
    discriminator_lr: float = 2e-4,
    discriminator_decay: float = 1e-6,
    batch_size: int = 500,
    discriminator_steps: int = 1,
    log_frequency: bool = True,
    verbose: bool = False,
    epochs: int = 300,
    pac: int = 10,
    cuda: Union[bool, str] = True,
) -> Dict[str, Any]:
    """Load data, detect metadata, train CTGAN, and persist the model."""

    # ── load & validate ───────────────────────────────────
    if not os.path.isfile(real_data_path):
        raise FileNotFoundError(f"Real data file not found: {real_data_path}")

    data = pd.read_csv(real_data_path)
    n_rows, n_cols = data.shape

    # ── validate batch_size vs pac ────────────────────────
    adjusted = False
    original_batch_size = batch_size
    if batch_size % pac != 0:
        batch_size = (batch_size // pac) * pac
        if batch_size == 0:
            batch_size = pac
        adjusted = True

    # ── auto-detect metadata ──────────────────────────────
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    # ── paths ─────────────────────────────────────────────
    model_path = _build_model_path(model_name, models_dir)

    # ── train ─────────────────────────────────────────────
    trainer = CTGANTrainer(model_path=model_path, output_path="")
    trainer.init_synthesizer(
        metadata,
        enforce_min_max_values=enforce_min_max_values,
        enforce_rounding=enforce_rounding,
        locales=locales,
        embedding_dim=embedding_dim,
        generator_dim=generator_dim or [256, 256],
        discriminator_dim=discriminator_dim or [256, 256],
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
    trainer.train(data)

    result = {
        "model_name": model_name,
        "model_path": os.path.abspath(model_path),
        "training_rows": n_rows,
        "training_columns": n_cols,
        "column_names": list(data.columns),
        "epochs": epochs,
        "batch_size": batch_size,
        "status": "Training complete. Model saved.",
    }
    if adjusted:
        result["warning"] = (
            f"batch_size was adjusted from {original_batch_size} to {batch_size} "
            f"because batch_size must be divisible by pac ({pac})."
        )
    return result


def generate_data(
    model_name: str,
    num_rows: int,
    output_filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a trained model and sample *num_rows* synthetic rows."""

    model_path = _build_model_path(model_name, models_dir)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            f"Train a model first with train_synthesizer."
        )

    out_dir = output_dir or _DEFAULT_OUTPUT_DIR
    filename = output_filename or f"{model_name}_synthetic.csv"

    generator = CTGANGenerator(model_path=model_path, output_path=out_dir)
    generator.load_model()
    synthetic_data = generator.generate(num_rows=num_rows, filename=filename)

    csv_path = os.path.join(out_dir, filename)

    return {
        "model_name": model_name,
        "num_rows": num_rows,
        "num_columns": len(synthetic_data.columns),
        "column_names": list(synthetic_data.columns),
        "output_path": os.path.abspath(csv_path),
        "status": "Synthetic data generated and saved.",
    }
