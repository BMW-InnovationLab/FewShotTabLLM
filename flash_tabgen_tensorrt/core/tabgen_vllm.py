"""
vLLM TabGen: High-performance synthetic tabular data generation using vLLM

Simplified alternative to TensorRT-LLM with easier setup and good performance.

Includes full Langfuse observability (same as the remote-vLLM backend):
- **Session-level grouping** via ``session_id``.
- **Profiling span** — tracks schema inference, column types, and profiling duration.
- **Generation root span** — encapsulates the entire ``generate()`` call.
- **Batch spans** — one per batch iteration with prompt count, timing.
- **Decoding** — per-batch decode with success/failure counts.
- **Dataset / prompt registration** and **score attachment**.

All tracing is fail-safe: missing env vars, missing package, or server errors
never interrupt generation.
"""

import json
import math
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Any
import pandas as pd
import numpy as np

from flash_tabgen_tensorrt.core.data_profiler import DataProfiler, DatasetProfile
from flash_tabgen_tensorrt.core.encoding import get_encoder, BaseEncoder
from flash_tabgen_tensorrt.core.prompts import PromptBuilder
from flash_tabgen_tensorrt.core.decoder import Decoder
from flash_tabgen_tensorrt.core.generator_vllm import GeneratorVLLM
from flash_tabgen_tensorrt.core.langfuse_utils import LangfuseManager

logger = logging.getLogger(__name__)


class TabGenVLLM:
    """
    High-performance Tabular Data Generator using vLLM

    Simpler alternative to TensorRT-LLM with:
    - No engine building required
    - Easier setup and deployment
    - Good performance with multi-GPU support
    - Text-only mode for models like Qwen 3.5

    Args:
        model: HuggingFace model name.
        mode: Generation mode ('fast', 'flexible', 'predictive').
        float_precision: Decimal places for floats.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        max_model_len: Maximum context length.
        gpu_memory_utilization: GPU memory utilization (0-1).
        max_batch_size: Maximum batch size.
        langfuse_enabled: When True (default), Langfuse tracing is auto-enabled if
                          env vars are set.
        session_id: Optional Langfuse session ID (e.g. experiment name).
        k_shots: Number of examples in few-shot prompts.
        verbose: Enable verbose logging.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3.5-35B-A3B",
        mode: str = "flexible",
        float_precision: int = 3,
        tensor_parallel_size: int = 2,
        max_model_len: int = 131072,
        gpu_memory_utilization: float = 0.90,
        max_batch_size: int = 64,
        langfuse_enabled: bool = True,
        session_id: Optional[str] = None,
        k_shots: int = 10,
        verbose: bool = False,
    ):
        self.model_name = model
        self.mode = mode
        self.float_precision = float_precision
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_batch_size = max_batch_size
        self.langfuse_enabled = langfuse_enabled
        self.session_id = session_id

        # Components (initialized after profiling)
        self.profiler = DataProfiler()
        self.dataset_profile: Optional[DatasetProfile] = None
        self.encoder: Optional[BaseEncoder] = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self.generator: Optional[GeneratorVLLM] = None
        self.decoder: Optional[Decoder] = None

        # Data
        self.train_data: Optional[pd.DataFrame] = None
        self.k_shots = k_shots
        self.verbose = verbose

        # Langfuse manager
        self._lf = LangfuseManager(
            enabled=langfuse_enabled,
            session_id=session_id,
            tags=["tabgen", "vllm-local"],
            metadata={"model": model},
        )
        self._lf_initialised = False
        self._last_trace_id: Optional[str] = None
        self._last_root_span: Optional[Any] = None

        if verbose:
            logger.info("=" * 80)
            logger.info("vLLM TABULAR DATA GENERATOR")
            logger.info("=" * 80)
            logger.info(f"Model: {model}")
            logger.info(f"Mode: {mode}")
            logger.info(f"Tensor parallel size: {tensor_parallel_size}")
            logger.info(f"Max model length: {max_model_len}")
            logger.info(f"Max batch size: {max_batch_size}")
            logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
            logger.info(f"Langfuse: {'enabled' if langfuse_enabled else 'disabled'}")
            logger.info("=" * 80)

    def profile(
        self,
        data: Union[str, pd.DataFrame],
        target_column: Optional[str] = None,
        type_overrides: Optional[Dict[str, str]] = None,
    ) -> DatasetProfile:
        """
        Profile dataset and infer schema.

        When Langfuse is active, a ``profiling`` span is logged.

        Args:
            data: Path to CSV/Parquet or DataFrame.
            target_column: Name of target variable (for predictive mode).
            type_overrides: Manual type specifications.

        Returns:
            DatasetProfile
        """
        # Ensure Langfuse is initialised
        if not self._lf_initialised:
            self._lf.init()
            self._lf_initialised = True

        profile_span = self._lf.start_span(
            name="profiling",
            input={
                "data_source": data if isinstance(data, str) else "(DataFrame)",
                "target_column": target_column,
                "type_overrides": type_overrides,
            },
        )
        profile_start = time.time()

        # Load data
        if isinstance(data, str):
            if data.endswith(".parquet"):
                self.train_data = pd.read_parquet(data)
            else:
                self.train_data = pd.read_csv(data)
        else:
            self.train_data = data.copy()

        # Profile
        self.dataset_profile = self.profiler.profile(
            self.train_data,
            target_column=target_column,
            type_overrides=type_overrides,
        )

        # Initialize encoder
        self.encoder = get_encoder(
            mode=self.mode,
            profile=self.dataset_profile,
            float_precision=self.float_precision,
        )

        # Initialize decoder — JSON mode when encoding is JSON, otherwise flexible
        decoder_mode = "json" if self.mode == "json" else "flexible"
        self.decoder = Decoder(self.dataset_profile, mode=decoder_mode)

        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(
            profile=self.dataset_profile,
            encoder=self.encoder,
            k_shots=self.k_shots,
            train_data=self.train_data,
        )

        profile_duration = time.time() - profile_start
        column_types = {col: prof.dtype for col, prof in self.dataset_profile.columns.items()}

        self._lf.end_span(
            profile_span,
            output={
                "n_rows": self.dataset_profile.n_rows,
                "n_cols": self.dataset_profile.n_cols,
                "column_types": column_types,
                "target_column": target_column,
                "duration_s": round(profile_duration, 3),
            },
        )

        if self.verbose:
            logger.info(
                f"Profiled {self.dataset_profile.n_rows} rows × "
                f"{self.dataset_profile.n_cols} columns"
            )
            logger.info(
                f"Column types: "
                f"{[f'{col}: {prof.dtype}' for col, prof in self.dataset_profile.columns.items()]}"
            )

        return self.dataset_profile

    def generate(
        self,
        n_rows: int = 100,
        conditional: Optional[Dict[str, Any]] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int = 2048,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        batch_size: int = 10,
        prompts_per_batch: int = 8,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic data using vLLM.

        Args:
            n_rows: Number of rows to generate.
            conditional: Conditional constraints {column: value}.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling.
            min_p: Minimum probability threshold.
            max_tokens: Max tokens per generation.
            presence_penalty: Penalty for token presence.
            repetition_penalty: Penalty for repeating tokens.
            batch_size: Number of rows to request per prompt.
            prompts_per_batch: Number of prompts per vLLM batch.
            seed: Random seed.

        Returns:
            DataFrame with synthetic data.
        """
        if self.dataset_profile is None:
            raise ValueError("Must call profile() first")

        # Initialize vLLM generator (lazy loading)
        if self.generator is None:
            if self.verbose:
                logger.info(f"Initializing vLLM in-process for: {self.model_name}")
                logger.info(f"Tensor parallel size: {self.tensor_parallel_size}")
                logger.info(f"GPU memory utilization: {self.gpu_memory_utilization}")

            # Build guided_json schema when in JSON encoding mode
            guided_json = None
            if self.mode == "json" and self.encoder is not None:
                if hasattr(self.encoder, "build_json_schema"):
                    guided_json = self.encoder.build_json_schema()
                    if self.verbose:
                        logger.info("[JSON mode] Guided JSON schema enabled for vLLM generator")

            self.generator = GeneratorVLLM(
                model_name=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                guided_json=guided_json,
                verbose=self.verbose,
            )

        # Lazy init Langfuse
        if not self._lf_initialised:
            self._lf.init()
            self._lf_initialised = True

        # Langfuse root span
        lf_root_span = self._lf.start_span(
            name="tabgen-generate",
            input={
                "n_rows": n_rows,
                "batch_size": batch_size,
                "prompts_per_batch": prompts_per_batch,
                "model": self.model_name,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "presence_penalty": presence_penalty,
                "repetition_penalty": repetition_penalty,
                "seed": seed,
            },
            metadata={
                "tensor_parallel_size": self.tensor_parallel_size,
                "max_model_len": self.max_model_len,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "dataset_rows": self.dataset_profile.n_rows,
                "dataset_cols": self.dataset_profile.n_cols,
            },
        )

        self._last_root_span = lf_root_span
        self._last_trace_id = self._lf.get_trace_id(lf_root_span)

        all_rows: List[pd.DataFrame] = []
        try:
            all_rows = self._generate_batch_optimized(
                n_rows=n_rows,
                rows_per_prompt=batch_size,
                prompts_per_batch=prompts_per_batch,
                conditional=conditional,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                langfuse_root=lf_root_span,
            )
        finally:
            total_generated = sum(len(df) for df in all_rows) if all_rows else 0
            self._lf.end_span(
                lf_root_span,
                output={"total_rows_generated": total_generated},
            )
            self._lf.flush()

        # Combine batches
        if all_rows:
            synthetic_df = pd.concat(all_rows, ignore_index=True)
            return synthetic_df.head(n_rows)
        else:
            return pd.DataFrame(columns=self.dataset_profile.column_order)

    def _generate_batch_optimized(
        self,
        n_rows: int,
        rows_per_prompt: int,
        prompts_per_batch: int,
        conditional: Optional[Dict[str, Any]] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int = 2048,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        langfuse_root: Optional[Any] = None,
    ) -> List[pd.DataFrame]:
        """
        Generate using parallel batch processing for maximum GPU utilization.

        Sends multiple prompts to vLLM at once instead of one at a time.
        """
        from tqdm import tqdm

        all_rows: List[pd.DataFrame] = []
        batch_times: List[float] = []
        total_rows_generated = 0
        batch_idx = 0

        # Calculate total prompts needed
        num_prompts_total = math.ceil(n_rows / rows_per_prompt)
        num_batches = math.ceil(num_prompts_total / prompts_per_batch)

        if self.verbose:
            logger.info(f"Parallel batch generation:")
            logger.info(f"  Rows per prompt: {rows_per_prompt}")
            logger.info(f"  Prompts per batch: {prompts_per_batch}")
            logger.info(f"  Total prompts: {num_prompts_total}")
            logger.info(f"  Total batches: {num_batches}")

        with tqdm(total=n_rows, desc="Generating rows", unit="row", ncols=100) as pbar:
            while total_rows_generated < n_rows:
                # Calculate how many prompts for this batch
                rows_remaining = n_rows - total_rows_generated
                prompts_needed = math.ceil(rows_remaining / rows_per_prompt)
                num_prompts_this_batch = min(prompts_per_batch, prompts_needed)

                if num_prompts_this_batch <= 0:
                    break

                # Build all prompts for this batch
                prompts_batch: List[str] = []
                for _ in range(num_prompts_this_batch):
                    prompt = self.prompt_builder.build_generation_prompt(
                        demo_data=self.train_data,
                        n_samples=rows_per_prompt,
                        conditional=conditional,
                        mode=self.mode,
                    )
                    prompts_batch.append(prompt)

                # Langfuse batch span
                lf_batch_span = self._lf.start_span(
                    name=f"batch-{batch_idx + 1}",
                    parent=langfuse_root,
                    input={
                        "num_prompts": num_prompts_this_batch,
                        "rows_remaining": rows_remaining,
                    },
                    metadata={"batch_idx": batch_idx},
                )

                # Generate ALL prompts in parallel
                batch_start = time.time()
                generated_texts = self.generator.generate_batch(
                    prompts=prompts_batch,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                )
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                # Decode all outputs (with Langfuse span)
                batch_rows = 0
                decode_errors = 0
                for text in generated_texts:
                    if text:
                        try:
                            batch_df = self.decoder.decode_batch([text])
                            if not batch_df.empty:
                                if len(batch_df) > rows_per_prompt:
                                    batch_df = batch_df.head(rows_per_prompt)
                                all_rows.append(batch_df)
                                batch_rows += len(batch_df)
                                total_rows_generated += len(batch_df)
                        except Exception as e:
                            decode_errors += 1
                            if self.verbose:
                                logger.warning(f"Decode error: {e}")
                            continue

                # Close batch span
                self._lf.end_span(
                    lf_batch_span,
                    output={
                        "rows_decoded": batch_rows,
                        "decode_errors": decode_errors,
                        "batch_time_s": round(batch_time, 2),
                    },
                )

                # Update progress
                rows_per_sec = batch_rows / batch_time if batch_time > 0 else 0
                pbar.set_postfix(
                    {
                        "batch": f"{batch_idx + 1}/{num_batches}",
                        "speed": f"{rows_per_sec:.0f} rows/s",
                    }
                )
                pbar.update(min(batch_rows, n_rows - pbar.n))

                if self.verbose:
                    logger.info(
                        f"Batch {batch_idx + 1}: {batch_rows} rows in "
                        f"{batch_time:.2f}s ({rows_per_sec:.1f} rows/s)"
                    )

                batch_idx += 1

        # Summary
        if batch_times:
            total_time = sum(batch_times)
            throughput = total_rows_generated / total_time if total_time > 0 else 0.0
            logger.info(
                f"\nGeneration complete: {len(batch_times)} batches | "
                f"{total_rows_generated} rows | {throughput:.1f} rows/s"
            )

        return all_rows

    # ------------------------------------------------------------------
    # Langfuse: dataset & prompt registration, score attachment
    # ------------------------------------------------------------------

    def register_dataset(
        self,
        dataset_name: str,
        description: Optional[str] = None,
        sample_items: int = 5,
    ) -> Optional[Any]:
        """Register the profiled real dataset in Langfuse."""
        if self.dataset_profile is None or self.train_data is None:
            logger.warning("Cannot register dataset — call profile() first")
            return None

        import json as _json

        ds_metadata = {
            "n_rows": str(self.dataset_profile.n_rows),
            "n_cols": str(self.dataset_profile.n_cols),
            "target_column": self.dataset_profile.target_column or "",
            "column_types": _json.dumps(
                {col: prof.dtype for col, prof in self.dataset_profile.columns.items()}
            ),
        }

        ds = self._lf.create_dataset(
            name=dataset_name,
            description=description or f"Real dataset: {dataset_name}",
            metadata=ds_metadata,
        )

        if ds is not None and sample_items > 0:
            sample = self.train_data.head(sample_items)
            for idx, row in sample.iterrows():
                row_dict = {
                    col: (val.item() if hasattr(val, "item") else val) for col, val in row.items()
                }
                self._lf.create_dataset_item(
                    dataset_name=dataset_name,
                    input=row_dict,
                    metadata={"row_index": str(idx)},
                )
        return ds

    def register_prompt_template(self) -> Optional[Any]:
        """Register the current prompt template in Langfuse for versioning."""
        if self.prompt_builder is None:
            logger.warning("Cannot register prompt — call profile() first")
            return None

        template = (
            "## Task: Generate Synthetic Tabular Data\n\n"
            "Generate realistic, diverse, and UNIQUE synthetic tabular data.\n\n"
            "## Schema and Statistics:\n{{schema_stats}}\n\n"
            "## Examples ({{k_shots}} rows):\n{{examples}}\n\n"
            "## Instructions:\n"
            "Generate **only** {{n_samples}} NEW and UNIQUE synthetic row(s).\n\n"
            "## Generated Data:"
        )

        return self._lf.create_prompt(
            name="tabgen-generation-prompt",
            prompt=template,
            prompt_type="text",
            labels=["production"],
            config={
                "model": self.model_name,
                "mode": self.mode,
                "k_shots": self.k_shots,
                "float_precision": self.float_precision,
            },
        )

    def push_evaluation_scores(
        self,
        scores: Dict[str, float],
        comment: Optional[str] = None,
    ) -> None:
        """Push evaluation metric scores to the last generation trace."""
        if self._last_root_span is None:
            logger.warning("[Langfuse] No generation trace found — call generate() first")
            return
        for metric_name, value in scores.items():
            self._lf.score_trace(
                self._last_root_span,
                name=metric_name,
                value=value,
                comment=comment,
            )
        self._lf.flush()

    @property
    def last_trace_id(self) -> Optional[str]:
        """Return the Langfuse trace ID from the most recent generate() call."""
        return self._last_trace_id

    @property
    def last_trace_url(self) -> Optional[str]:
        """Return the Langfuse UI URL for the most recent generate() trace."""
        return self._lf.get_trace_url(self._last_root_span)

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def save_config(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON."""
        config: Dict[str, Any] = {
            "model": self.model_name,
            "mode": self.mode,
            "float_precision": self.float_precision,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_batch_size": self.max_batch_size,
            "langfuse_enabled": self.langfuse_enabled,
            "k_shots": self.k_shots,
        }
        if self._last_trace_id:
            config["last_trace_id"] = self._last_trace_id
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        if self.verbose:
            logger.info(f"Saved config to {path}")

    @classmethod
    def from_config(cls, path: Union[str, Path], verbose: bool = False) -> "TabGenVLLM":
        """Load from configuration JSON."""
        with open(path, "r") as f:
            config = json.load(f)
        config.pop("last_trace_id", None)
        return cls(**config, verbose=verbose)
