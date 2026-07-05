"""
TensorRT-LLM TabGen: High-performance synthetic tabular data generation

Optimized for B200 and other high-end GPUs using TensorRT-LLM.
Provides significantly better performance than transformers-based implementation.
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
from flash_tabgen_tensorrt.core.generator_tensorrt import GeneratorTensorRT

logger = logging.getLogger(__name__)


class TabGenTensorRT:
    """
    High-performance Tabular Data Generator using TensorRT-LLM

    Provides significantly better performance than transformers-based implementation
    on B200 and other high-end GPUs with optimized memory usage and inference speed.

    Based on NVIDIA's TensorRT-LLM: https://nvidia.github.io/TensorRT-LLM/
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        mode: str = "flexible",
        float_precision: int = 3,
        engine_dir: Optional[str] = None,
        max_batch_size: int = 64,
        max_input_len: int = 16384,
        max_output_len: int = 2048,
        dtype: str = "bfloat16",
        use_paged_attention: bool = True,
        use_inflight_batching: bool = True,
        max_concurrent_prompts: Optional[int] = None,
        k_shots: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize TensorRT-LLM TabGen

        Args:
            model: HuggingFace model name
            mode: Generation mode ('fast', 'flexible', 'predictive')
            float_precision: Decimal places for floats
            engine_dir: Directory containing pre-built TensorRT engine
            max_batch_size: Maximum batch size for TensorRT-LLM
            max_input_len: Maximum input sequence length
            max_output_len: Maximum output sequence length
            dtype: Data type for inference (bfloat16, float16, float32)
            use_paged_attention: Enable paged attention for memory efficiency
            use_inflight_batching: Enable in-flight batching for better throughput
            max_concurrent_prompts: Max prompts dispatched per TensorRT call
        """
        self.model_name = model
        self.mode = mode
        self.float_precision = float_precision
        self.engine_dir = engine_dir
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.dtype = dtype
        self.use_paged_attention = use_paged_attention
        self.use_inflight_batching = use_inflight_batching
        self.max_concurrent_prompts = max(
            1,
            min(
                max_batch_size,
                max_concurrent_prompts if max_concurrent_prompts is not None else max_batch_size,
            ),
        )

        # Components (initialized after profiling)
        self.profiler = DataProfiler()
        self.dataset_profile: Optional[DatasetProfile] = None
        self.encoder: Optional[BaseEncoder] = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self.generator: Optional[GeneratorTensorRT] = None
        self.decoder: Optional[Decoder] = None

        # Data
        self.train_data: Optional[pd.DataFrame] = None
        self.k_shots = k_shots
        self.verbose = verbose

        if verbose:
            logger.info("=" * 80)
            logger.info("TENSORRT-LLM TABULAR DATA GENERATOR")
            logger.info("=" * 80)
            logger.info(f"Model: {model}")
            logger.info(f"Mode: {mode}")
            logger.info(f"Max batch size: {max_batch_size}")
            logger.info(f"Max concurrent prompts: {self.max_concurrent_prompts}")
            logger.info(f"Max input/output length: {max_input_len}/{max_output_len}")
            logger.info(f"Data type: {dtype}")
            logger.info(f"Paged attention: {use_paged_attention}")
            logger.info(f"In-flight batching: {use_inflight_batching}")
            logger.info("=" * 80)

    def profile(
        self,
        data: Union[str, pd.DataFrame],
        target_column: Optional[str] = None,
        type_overrides: Optional[Dict[str, str]] = None,
    ) -> DatasetProfile:
        """
        Profile dataset and infer schema

        Args:
            data: Path to CSV/Parquet or DataFrame
            target_column: Name of target variable (for predictive mode)
            type_overrides: Manual type specifications

        Returns:
            DatasetProfile
        """
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

        if self.verbose:
            logger.info(
                f"Profiled {self.dataset_profile.n_rows} rows × {self.dataset_profile.n_cols} columns"
            )
            logger.info(
                f"Column types: {[f'{col}: {prof.dtype}' for col, prof in self.dataset_profile.columns.items()]}"
            )

        return self.dataset_profile

    def generate(
        self,
        n_rows: int = 100,
        conditional: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        batch_size: int = 10,
        seed: Optional[int] = None,
        use_batch_generation: bool = True,
        permute: bool = False,
        use_correlation_matrix: bool = True,
    ) -> pd.DataFrame:
        """
        Generate synthetic data using TensorRT-LLM

        Args:
            n_rows: Number of rows to generate
            conditional: Conditional constraints {column: value}
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            min_p: Minimum cumulative probability for nucleus sampling
            top_k: Top-k sampling parameter
            max_new_tokens: Max tokens per generation
            batch_size: Number of rows to request per model call
            seed: Random seed
            use_batch_generation: Use TensorRT-LLM batch generation for better performance

        Returns:
            DataFrame with synthetic data
        """
        if self.dataset_profile is None:
            raise ValueError("Must call profile() first")

        # if seed is not None:
        #     np.random.seed(seed)

        # Initialize TensorRT-LLM generator (lazy loading)
        if self.generator is None:
            if self.verbose:
                logger.info(f"Initializing TensorRT-LLM generator for: {self.model_name}")
            self.generator = GeneratorTensorRT(
                model_name=self.model_name,
                profile=self.dataset_profile,
                engine_dir=self.engine_dir,
                max_batch_size=self.max_batch_size,
                max_input_len=self.max_input_len,
                max_output_len=self.max_output_len,
                dtype=self.dtype,
                use_paged_attention=self.use_paged_attention,
                use_inflight_batching=self.use_inflight_batching,
                verbose=self.verbose,
            )

        # Generate in batches
        all_rows = []
        rows_per_call = max(1, int(batch_size))

        if self.verbose:
            logger.info(f"Using TensorRT-LLM batch size: {rows_per_call} rows per call")
            logger.info(f"Total rows to generate: {n_rows}")

        if use_batch_generation and rows_per_call > 1:
            # Use TensorRT-LLM batch generation for better performance
            all_rows = self._generate_batch_optimized(
                n_rows=n_rows,
                rows_per_call=rows_per_call,
                conditional=conditional,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_new_tokens=max_new_tokens,
                permute=permute,
                use_correlation_matrix=use_correlation_matrix,
            )
        else:
            # Use sequential generation
            all_rows = self._generate_sequential(
                n_rows=n_rows,
                rows_per_call=rows_per_call,
                conditional=conditional,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_new_tokens=max_new_tokens,
                use_correlation_matrix=use_correlation_matrix,
            )

        # Combine batches
        if all_rows:
            synthetic_df = pd.concat(all_rows, ignore_index=True)
            return synthetic_df.head(n_rows)
        else:
            # Return empty dataframe with correct columns if nothing generated
            return pd.DataFrame(columns=self.dataset_profile.column_order)

    def _generate_batch_optimized(
        self,
        n_rows: int,
        rows_per_call: int,
        conditional: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        permute: bool = False,
        use_correlation_matrix: bool = True,
    ) -> List[pd.DataFrame]:
        """
        Generate using TensorRT-LLM batch optimization with true parallel prompt processing

        Key optimization: Sends multiple prompts to TensorRT in parallel for maximum GPU utilization
        """
        if n_rows <= 0:
            return []

        if self.generator is None:
            raise RuntimeError("TensorRT generator must be initialized before batch generation.")

        # Calculate batching strategy
        max_prompts_per_batch = min(self.max_concurrent_prompts, self.generator.max_batch_size)

        # Calculate dynamic max_new_tokens based on actual prompt if not provided
        if max_new_tokens is None:
            num_columns = self.dataset_profile.n_cols
            max_new_tokens = self.generator.estimate_output_tokens(rows_per_call, num_columns)

        all_rows: List[pd.DataFrame] = []
        batch_times: List[float] = []
        total_rows_generated = 0
        batch_idx = 0

        # Progress bar
        from tqdm import tqdm

        with tqdm(total=n_rows, desc="Generating rows", unit="row", ncols=100) as pbar:
            while total_rows_generated < n_rows:
                # Calculate how many prompts needed for remaining rows
                rows_remaining = n_rows - total_rows_generated
                prompts_needed = math.ceil(rows_remaining / rows_per_call)

                # Ensure we don't exceed max concurrent prompts
                num_prompts_this_batch = min(max_prompts_per_batch, prompts_needed)

                if num_prompts_this_batch <= 0:
                    break

                # Build all prompts for this batch
                prompts_batch: List[str] = []
                for _ in range(num_prompts_this_batch):
                    prompt = self.prompt_builder.build_generation_prompt(
                        demo_data=self.train_data,
                        n_samples=rows_per_call,
                        conditional=conditional,
                        mode=self.mode,
                        permute=permute,
                        use_correlation_matrix=use_correlation_matrix,
                    )
                    prompts_batch.append(prompt)
                    # Log each prompt in the batch
                    logger.debug("================ BATCH PROMPT ================")
                    logger.debug(prompt)
                    logger.debug("===============================================")

                # Generate all prompts in parallel
                batch_start = time.perf_counter()
                generated_texts = self.generator.generate_batch(
                    prompts=prompts_batch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                    top_k=top_k,
                )
                batch_time = time.perf_counter() - batch_start
                batch_times.append(batch_time)

                # Log raw generated text
                for i, gen_text_list in enumerate(generated_texts):
                    for j, gen_text in enumerate(gen_text_list):
                        logger.debug(
                            f"--- Raw Generated Text (Batch {batch_idx}, Prompt {i}, Seq {j}) ---"
                        )
                        logger.debug(gen_text)
                        logger.debug(
                            "------------------------------------------------------------------"
                        )

                # Decode all generated texts
                batch_rows = 0
                for generated_text_list in generated_texts:
                    if generated_text_list:
                        batch_df = self.decoder.decode_batch(generated_text_list)
                        if len(batch_df) > 0:
                            # Log decoded dataframe
                            logger.debug(f"--- Decoded Data Batch (size: {len(batch_df)}) ---")
                            logger.debug(f"\n{batch_df.to_string()}")
                            logger.debug("-----------------------------------------------")

                            # Limit to requested rows_per_call (e.g., if asked for 25, only take first 25)
                            if len(batch_df) > rows_per_call:
                                batch_df = batch_df.head(rows_per_call)

                            all_rows.append(batch_df)
                            batch_rows += len(batch_df)
                            total_rows_generated += len(batch_df)
                        else:
                            logger.debug("--- No valid rows decoded from this text sequence ---")

                # Update progress bar
                rows_per_sec = batch_rows / batch_time if batch_time > 0 else 0
                pbar.set_postfix(
                    {"batch": f"{batch_idx + 1}", "speed": f"{rows_per_sec:.0f} rows/s"}
                )
                pbar.update(min(batch_rows, n_rows - pbar.n))  # Don't exceed total

                logger.debug(
                    f"Batch {batch_idx + 1} completed. Rows generated: {batch_rows}. Speed: {rows_per_sec:.1f} rows/s"
                )

                batch_idx += 1

        # Print summary statistics
        if batch_times:
            total_time = sum(batch_times)
            throughput = total_rows_generated / total_time if total_time > 0 else 0.0
            logger.info(
                f"\nGeneration complete: {len(batch_times)} batches | "
                f"{total_rows_generated} rows | {throughput:.1f} rows/s"
            )

        return all_rows

    def _generate_sequential(
        self,
        n_rows: int,
        rows_per_call: int,
        conditional: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        use_correlation_matrix: bool = True,
    ) -> List[pd.DataFrame]:
        """Generate using sequential calls"""
        all_rows = []
        remaining = n_rows

        while remaining > 0:
            current_request = min(rows_per_call, remaining)

            # Build prompt
            prompt = self.prompt_builder.build_generation_prompt(
                demo_data=self.train_data,
                n_samples=current_request,
                conditional=conditional,
                mode=self.mode,
                use_correlation_matrix=use_correlation_matrix,
            )
            # if self.verbose:
            logger.info("================ PROMPT ================")
            logger.info("========================================")
            logger.info(prompt)
            logger.info("========================================")
            logger.info("========================================")

            # Generate using TensorRT-LLM
            generated_texts = self.generator.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens or (current_request * 200),
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                num_return_sequences=1,
            )

            # Log raw generated text
            for i, gen_text in enumerate(generated_texts):
                logger.debug(f"--- Raw Generated Text (Seq {i}) ---")
                logger.debug(gen_text)
                logger.debug("------------------------------------")

            # Decode
            batch_df = self.decoder.decode_batch(generated_texts)

            if len(batch_df) > 0:
                # Log decoded dataframe
                logger.debug(f"--- Decoded Data Batch (size: {len(batch_df)}) ---")
                logger.debug(f"\n{batch_df.to_string()}")
                logger.debug("-----------------------------------------------")

                # Limit to requested number of rows
                if len(batch_df) > current_request:
                    batch_df = batch_df.head(current_request)

                all_rows.append(batch_df)
                actual_generated = len(batch_df)
                remaining -= actual_generated
                logger.info(f"Generated {actual_generated} rows")
            else:
                logger.warning(f"Warning: No valid rows decoded from batch")
                remaining -= current_request  # Avoid infinite loop

        return all_rows

    def evaluate(
        self,
        real_test: Union[str, pd.DataFrame],
        synthetic: pd.DataFrame,
        real_train: Optional[Union[str, pd.DataFrame]] = None,
        run_tstr: bool = True,
        run_privacy: bool = True,
    ) -> Dict:
        """
        Comprehensive evaluation of synthetic data quality

        Uses the same evaluation logic as the transformers implementation
        """
        from flash_tabgen.core.evaluators import (
            tstr_evaluation,
            correlation_distance,
            statistical_similarity,
        )
        from flash_tabgen.core.evaluators.statistical import dcr_privacy

        # Load real test data
        if isinstance(real_test, str):
            if real_test.endswith(".parquet"):
                real_df = pd.read_parquet(real_test)
            else:
                real_df = pd.read_csv(real_test)
        else:
            real_df = real_test

        # Basic statistics
        report = {
            "n_synthetic": len(synthetic),
            "n_real": len(real_df),
            "column_stats": {},
        }

        # Column-wise statistics
        for col in synthetic.columns:
            if col in real_df.columns:
                real_col = real_df[col]
                synth_col = synthetic[col]

                col_prof = self.dataset_profile.columns[col]

                if col_prof.dtype in ["integer", "float"]:
                    report["column_stats"][col] = {
                        "real_mean": float(real_col.mean()),
                        "synth_mean": float(synth_col.mean()),
                        "real_std": float(real_col.std()),
                        "synth_std": float(synth_col.std()),
                    }
                elif col_prof.dtype == "categorical":
                    real_dist = real_col.value_counts(normalize=True).to_dict()
                    synth_dist = synth_col.value_counts(normalize=True).to_dict()
                    report["column_stats"][col] = {
                        "real_distribution": real_dist,
                        "synth_distribution": synth_dist,
                    }

        # Statistical similarity
        try:
            report["statistical_similarity"] = statistical_similarity(real_df, synthetic)
            report["correlation_distance"] = correlation_distance(real_df, synthetic)
        except Exception as e:
            report["statistical_similarity"] = {"error": str(e)}

        # TSTR evaluation
        if run_tstr and self.dataset_profile.target_column:
            try:
                report["tstr"] = tstr_evaluation(
                    synthetic_train=synthetic,
                    real_test=real_df,
                    target_column=self.dataset_profile.target_column,
                )
            except Exception as e:
                report["tstr"] = {"error": str(e)}

        # Privacy evaluation (DCR)
        if run_privacy and real_train is not None:
            try:
                # Load real train data
                if isinstance(real_train, str):
                    if real_train.endswith(".parquet"):
                        train_df = pd.read_parquet(real_train)
                    else:
                        train_df = pd.read_csv(real_train)
                else:
                    train_df = real_train

                report["privacy"] = dcr_privacy(
                    real_train=train_df, real_test=real_df, synthetic=synthetic
                )
            except Exception as e:
                report["privacy"] = {"error": str(e)}

        return report

    def save_engine(self, engine_dir: str):
        """
        Save the TensorRT engine for faster loading in future runs

        Args:
            engine_dir: Directory to save the engine
        """
        if self.generator is not None:
            self.generator.save_engine(engine_dir)
        else:
            logger.warning("No generator loaded to save engine")

    def cleanup(self):
        """
        Clean up TensorRT-LLM resources to prevent memory leaks and hanging

        Call this when done with TabGenTensorRT, especially for sequential experiments.
        """
        logger.info("Cleaning up TensorRT resources...")

        # Clean up generator and LLM
        if self.generator is not None:
            try:
                if hasattr(self.generator, "cleanup"):
                    self.generator.cleanup()
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Warning: Generator cleanup error: {e}")
            try:
                del self.generator
            except:
                pass
            self.generator = None

        # Clean up all references
        self.train_data = None
        self.dataset_profile = None
        self.encoder = None
        self.prompt_builder = None
        self.decoder = None

        # Aggressive garbage collection
        import gc

        gc.collect()
        gc.collect()  # Run twice for cyclic references

        # Clear CUDA cache and synchronize
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()  # Clean up IPC handles
        except Exception as e:
            if self.verbose:
                logger.warning(f"Warning: CUDA cleanup error: {e}")

        logger.info("✓ Cleanup complete")

    # def __del__(self):
    #     """Destructor to ensure cleanup on object deletion"""
    #     try:
    #         self.cleanup()
    #     except:
    #         pass

    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance information about the TensorRT-LLM setup"""
        info = {
            "model_name": self.model_name,
            "mode": self.mode,
            "max_batch_size": self.max_batch_size,
            "max_input_len": self.max_input_len,
            "max_output_len": self.max_output_len,
            "dtype": self.dtype,
            "use_paged_attention": self.use_paged_attention,
            "use_inflight_batching": self.use_inflight_batching,
            "engine_dir": self.engine_dir,
        }

        if self.generator is not None:
            info.update(self.generator.get_model_info())

        return info

    def save(self, directory: str):
        """Save model configuration and profile"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        config = {
            "model_name": self.model_name,
            "mode": self.mode,
            "float_precision": self.float_precision,
            "engine_dir": self.engine_dir,
            "max_batch_size": self.max_batch_size,
            "max_input_len": self.max_input_len,
            "max_output_len": self.max_output_len,
            "dtype": self.dtype,
            "use_paged_attention": self.use_paged_attention,
            "use_inflight_batching": self.use_inflight_batching,
        }

        with open(dir_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        if self.dataset_profile:
            with open(dir_path / "profile.json", "w") as f:
                json.dump(self.dataset_profile.to_dict(), f, indent=2)

        if self.verbose:
            logger.info(f"Saved TensorRT-LLM configuration to {directory}")

    @classmethod
    def load_from_dir(cls, directory: str) -> "TabGenTensorRT":
        """Load model from directory"""
        dir_path = Path(directory)

        with open(dir_path / "config.json", "r") as f:
            config = json.load(f)

        tabgen = cls(**config)

        # Load profile if exists
        profile_path = dir_path / "profile.json"
        if profile_path.exists():
            with open(profile_path, "r") as f:
                profile_dict = json.load(f)
            # TODO: Deserialize profile from dict
            print("Profile loading from dict not fully implemented")

        return tabgen
