"""
Remote vLLM TabGen: High-performance synthetic tabular data generation via a remote vLLM server.

Replaces the local in-process vLLM engine (GeneratorVLLM) with HTTP calls to a deployed
vLLM server that exposes the OpenAI-compatible /v1/chat/completions endpoint.

All data profiling, prompt construction, and output decoding are identical to TabGenVLLM;
only the generation step changes — prompts are sent concurrently over HTTP.

Full Langfuse observability is built in:
- **Session-level grouping** via ``session_id`` (typically the experiment name).
- **Profiling span** — tracks schema inference, column types, and profiling duration.
- **Generation root span** — encapsulates the entire ``generate()`` call with all
  sampling parameters, model config, and dataset metadata.
- **Batch spans** — one per batch iteration with prompt count, decode rate, timing.
- **Generation observations** — one per LLM HTTP request with token usage, latency.
- **Decoding span** — per-batch decode with success/failure counts.
- **Prompt management** — optionally fetch prompt templates from Langfuse.
- **Dataset registration** — register real datasets and link to traces.
- **Score attachment** — push evaluation metrics back to traces.

All tracing is fail-safe: missing env vars, missing package, or server errors
never interrupt generation.

Typical usage::

    tabgen = TabGenRemoteVLLM(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        base_url="http://my-gpu-server:8000",
        k_shots=80,
        concurrent_requests=32,
    )
    tabgen.profile("adult.csv", target_column="income")
    df = tabgen.generate(n_rows=5000, temperature=0.7, batch_size=25)
"""

import json
import math
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Any
import pandas as pd

from flash_tabgen_tensorrt.core.data_profiler import DataProfiler, DatasetProfile
from flash_tabgen_tensorrt.core.encoding import get_encoder, BaseEncoder
from flash_tabgen_tensorrt.core.prompts import PromptBuilder
from flash_tabgen_tensorrt.core.decoder import Decoder
from flash_tabgen_tensorrt.core.generator_remote_vllm import GeneratorRemoteVLLM
from flash_tabgen_tensorrt.core.langfuse_utils import LangfuseManager

logger = logging.getLogger(__name__)


class TabGenRemoteVLLM:
    """
    Synthetic tabular data generator that offloads inference to a remote vLLM server.

    Compared to TabGenVLLM:
    - No GPU required on the client machine.
    - No model weights are loaded locally.
    - Prompts are sent concurrently over HTTP for high throughput.
    - ``concurrent_requests`` controls how many HTTP requests are in-flight at once;
      tune this to match your server's capacity.

    Args:
        model: Model name as registered on the remote server (must match exactly what
               the server was started with).
        base_url: URL of the remote vLLM server, e.g. "http://my-server:8000".
                   The /v1/chat/completions path is appended automatically.
        api_key: Optional Bearer token if the server requires authentication.
        mode: Generation mode passed to the encoder ('fast', 'flexible', 'predictive').
        float_precision: Decimal places used when encoding float values in prompts.
        concurrent_requests: Maximum number of HTTP requests in-flight at once.
                             Higher values increase throughput but require the server to
                             have enough capacity.  Default: 32.
        request_timeout: Per-request HTTP timeout in seconds.  Increase for very large
                         batch sizes or slow networks.  Default: 600.
        langfuse_enabled: When True (default), Langfuse tracing is auto-enabled if
                          LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set in the
                          environment.  Set to False to disable tracing entirely
                          (equivalent to ``--disable-lang-fuse`` on the CLI).
        session_id: Optional Langfuse session ID to group all traces from one
                    experiment run.  Typically set to the experiment name.
        k_shots: Number of real-data examples to include in each few-shot prompt.
        verbose: Emit detailed logs including per-batch timing and progress.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        mode: str = "predictive",
        float_precision: int = 4,
        concurrent_requests: int = 300,
        request_timeout: float = 600.0,
        enable_thinking: bool = False,
        langfuse_enabled: bool = True,
        session_id: Optional[str] = None,
        k_shots: int = 40,
        verbose: bool = False,
    ):
        self.model_name = model
        self.base_url = base_url
        self.api_key = api_key
        self.mode = mode
        self.float_precision = float_precision
        self.concurrent_requests = concurrent_requests
        self.request_timeout = request_timeout
        self.enable_thinking = enable_thinking
        self.langfuse_enabled = langfuse_enabled
        self.session_id = session_id
        self.k_shots = k_shots
        self.verbose = verbose

        # Components — initialised after profile() is called
        self.profiler = DataProfiler()
        self.dataset_profile: Optional[DatasetProfile] = None
        self.encoder: Optional[BaseEncoder] = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self.decoder: Optional[Decoder] = None
        self.train_data: Optional[pd.DataFrame] = None

        # Generator — created lazily on first generate() call so __init__ is cheap
        self.generator: Optional[GeneratorRemoteVLLM] = None

        # Langfuse manager — initialised lazily on first generate() call
        self._lf = LangfuseManager(
            enabled=langfuse_enabled,
            session_id=session_id,
            tags=["tabgen", "remote-vllm"],
            metadata={"model": model, "base_url": base_url},
        )
        self._lf_initialised = False

        # Track the last generation's trace ID and root span for score attachment
        self._last_trace_id: Optional[str] = None
        self._last_root_span: Optional[Any] = None

        if verbose:
            logger.info("=" * 80)
            logger.info("REMOTE vLLM TABULAR DATA GENERATOR")
            logger.info("=" * 80)
            logger.info(f"  Server          : {base_url}")
            logger.info(f"  Model           : {model}")
            logger.info(f"  Mode            : {mode}")
            logger.info(f"  Concurrent reqs : {concurrent_requests}")
            logger.info(f"  Request timeout : {request_timeout}s")
            logger.info(f"  Enable thinking : {enable_thinking}")
            logger.info(f"  Langfuse        : {'enabled' if langfuse_enabled else 'disabled'}")
            logger.info(f"  Session ID      : {session_id or '(auto)'}")
            logger.info(f"  k-shots         : {k_shots}")
            logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def profile(
        self,
        data: Union[str, pd.DataFrame],
        target_column: Optional[str] = None,
        type_overrides: Optional[Dict[str, str]] = None,
    ) -> DatasetProfile:
        """
        Profile the dataset to infer schema and statistics.

        Must be called before generate().

        When Langfuse is active, a ``profiling`` span is logged with the
        detected column types, dataset shape, and profiling duration.

        Args:
            data: Path to a CSV / Parquet file, or an already-loaded DataFrame.
            target_column: Name of the target variable (used in predictive mode).
            type_overrides: Dict of {column: dtype_string} to override auto-detection.

        Returns:
            DatasetProfile with column types, statistics, and encoding info.
        """
        # Ensure Langfuse is initialised so profiling can be traced
        if not self._lf_initialised:
            self._lf.init()
            self._lf_initialised = True

        # Start profiling span
        profile_span = self._lf.start_span(
            name="profiling",
            input={
                "data_source": data if isinstance(data, str) else "(DataFrame)",
                "target_column": target_column,
                "type_overrides": type_overrides,
            },
        )
        profile_start = time.time()

        if isinstance(data, str):
            if data.endswith(".parquet"):
                self.train_data = pd.read_parquet(data)
            else:
                self.train_data = pd.read_csv(data)
        else:
            self.train_data = data.copy()

        self.dataset_profile = self.profiler.profile(
            self.train_data,
            target_column=target_column,
            type_overrides=type_overrides,
        )

        self.encoder = get_encoder(
            mode=self.mode,
            profile=self.dataset_profile,
            float_precision=self.float_precision,
        )

        # Decoder always uses 'flexible' mode to parse the GReaT "col is val" format
        # except in JSON mode where it parses JSON objects.
        decoder_mode = "json" if self.mode == "json" else "flexible"
        self.decoder = Decoder(self.dataset_profile, mode=decoder_mode)

        self.prompt_builder = PromptBuilder(
            profile=self.dataset_profile,
            encoder=self.encoder,
            k_shots=self.k_shots,
            train_data=self.train_data,
        )

        profile_duration = time.time() - profile_start

        # Build column type summary for Langfuse
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
                "Column types: "
                + str([f"{c}: {p.dtype}" for c, p in self.dataset_profile.columns.items()])
            )

        return self.dataset_profile

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

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
        batch_size: int = 25,
        prompts_per_batch: int = 8,
    ) -> pd.DataFrame:
        """
        Generate synthetic data by sending prompts to the remote vLLM server.

        Args:
            n_rows: Total number of synthetic rows to produce.
            conditional: Optional column constraints, e.g. {"income": ">50K"}.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling.
            min_p: Minimum probability threshold.
            max_tokens: Maximum tokens to generate per prompt (i.e. per batch of rows).
            presence_penalty: Penalises tokens that have appeared in the prompt.
            repetition_penalty: Penalises repeated tokens in the output.
            batch_size: Number of rows requested from the model in a single prompt.
                        Each prompt asks for ``batch_size`` rows.
            prompts_per_batch: Number of prompts dispatched concurrently per HTTP round-trip.
                               Total concurrent HTTP requests = prompts_per_batch (capped by
                               ``concurrent_requests`` set at construction time).
                               Higher values saturate the server faster.

        Returns:
            DataFrame with exactly ``n_rows`` synthetic rows (or fewer if the model
            consistently produces short outputs).
        """
        if self.dataset_profile is None:
            raise ValueError("Must call profile() before generate().")

        # Lazy init of the HTTP generator (cheap — just stores config)
        if self.generator is None:
            if self.verbose:
                logger.info("Initialising remote vLLM generator ...")

            # Build guided_json schema when in JSON encoding mode
            guided_json = None
            if self.mode == "json" and self.encoder is not None:
                if hasattr(self.encoder, "build_json_schema"):
                    guided_json = self.encoder.build_json_schema()
                    if self.verbose:
                        logger.info("[JSON mode] Guided JSON schema enabled for remote generator")

            self.generator = GeneratorRemoteVLLM(
                base_url=self.base_url,
                model=self.model_name,
                api_key=self.api_key,
                concurrent_requests=self.concurrent_requests,
                request_timeout=self.request_timeout,
                enable_thinking=self.enable_thinking,
                guided_json=guided_json,
                verbose=self.verbose,
            )

        # Lazy init Langfuse (once per TabGen instance)
        if not self._lf_initialised:
            self._lf.init()
            self._lf_initialised = True

        # ------------------------------------------------------------------
        # Langfuse: create a root span for the entire generate() call
        # ------------------------------------------------------------------
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
                "enable_thinking": self.enable_thinking,
                "k_shots": self.k_shots,
                "mode": self.mode,
                "conditional": conditional,
            },
            metadata={
                "base_url": self.base_url,
                "concurrent_requests": self.concurrent_requests,
                "dataset_rows": self.dataset_profile.n_rows,
                "dataset_cols": self.dataset_profile.n_cols,
                "target_column": self.dataset_profile.target_column,
                "session_id": self.session_id or "",
            },
        )

        # Store for external access (e.g. score attachment from ml_eval)
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
            # Always close the root span and flush, even on error
            total_generated = sum(len(df) for df in all_rows) if all_rows else 0
            self._lf.end_span(
                lf_root_span,
                output={"total_rows_generated": total_generated},
            )
            self._lf.flush()

        if all_rows:
            synthetic_df = pd.concat(all_rows, ignore_index=True)
            return synthetic_df.head(n_rows)

        return pd.DataFrame(columns=self.dataset_profile.column_order)

    # ------------------------------------------------------------------
    # Internal: batched concurrent generation loop
    # ------------------------------------------------------------------

    def _decode_texts(
        self,
        generated_texts: List[str],
        rows_per_prompt: int,
        langfuse_parent: Optional[Any] = None,
    ) -> List["pd.DataFrame"]:
        """
        Decode a list of raw completion texts into DataFrames.

        When a Langfuse parent is provided, a ``decoding`` span is logged with
        success/failure counts and decode rate.

        Returns a list of non-empty DataFrames (one per successfully decoded text).
        """
        decode_span = self._lf.start_span(
            name="decoding",
            parent=langfuse_parent,
            input={
                "num_texts": len(generated_texts),
                "rows_per_prompt": rows_per_prompt,
            },
        )
        decode_start = time.time()

        decoded: List[pd.DataFrame] = []
        decode_errors = 0
        empty_texts = 0
        total_rows_decoded = 0

        for text in generated_texts:
            if not text:
                empty_texts += 1
                continue
            try:
                batch_df = self.decoder.decode_batch([text])
                if not batch_df.empty:
                    if len(batch_df) > rows_per_prompt:
                        batch_df = batch_df.head(rows_per_prompt)
                    decoded.append(batch_df)
                    total_rows_decoded += len(batch_df)
            except Exception as exc:
                decode_errors += 1
                if self.verbose:
                    logger.warning(f"Decode error: {exc}")

        decode_duration = time.time() - decode_start
        non_empty = len(generated_texts) - empty_texts
        decode_rate = total_rows_decoded / (non_empty * rows_per_prompt) if non_empty > 0 else 0.0

        self._lf.end_span(
            decode_span,
            output={
                "rows_decoded": total_rows_decoded,
                "texts_received": len(generated_texts),
                "empty_texts": empty_texts,
                "decode_errors": decode_errors,
                "decode_rate": round(decode_rate, 4),
                "duration_s": round(decode_duration, 3),
            },
        )

        return decoded

    def _generate_batch_optimized(
        self,
        n_rows: int,
        rows_per_prompt: int,
        prompts_per_batch: int,
        conditional: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        max_tokens: int,
        presence_penalty: float,
        repetition_penalty: float,
        langfuse_root: Optional[Any] = None,
    ) -> List[pd.DataFrame]:
        """
        Core generation loop.

        Each iteration:
          1. Build prompts (up to ``prompts_per_batch``) and fire them concurrently.
          2. Decode each response text into a DataFrame using the Decoder.
          3. Accumulate rows until we reach ``n_rows``.

        The number of prompts per batch accounts for the *observed decode success
        rate* — i.e. the fraction of requested rows that actually decode.  This
        prevents the pathological ramp-down where the loop sends fewer and fewer
        prompts each round because it naively assumes 100% decode success.

        On the very first batch we assume a conservative 30% decode rate so the
        server is saturated from the start.  Subsequent batches use the actual
        observed rate.  A 1.5× overshoot factor is applied on top so we finish
        in fewer rounds; any excess rows are trimmed by `generate()`.
        """
        from tqdm import tqdm

        # ------------------------------------------------------------------
        # Tunables
        # ------------------------------------------------------------------
        _INITIAL_DECODE_RATE = 0.30  # conservative first-batch assumption
        _OVERSHOOT_FACTOR = 1.5  # generate 50% more prompts than the
        # minimum to finish faster
        _MAX_CONSECUTIVE_EMPTY = 5  # abort if this many consecutive batches
        # produce zero rows (avoids infinite loop)

        all_rows: List[pd.DataFrame] = []
        batch_times: List[float] = []
        total_rows_generated = 0
        total_prompts_sent = 0
        batch_idx = 0
        consecutive_empty = 0

        # Estimate batches for the progress display (will be adjusted)
        est_prompts_total = math.ceil(
            n_rows / (rows_per_prompt * _INITIAL_DECODE_RATE) * _OVERSHOOT_FACTOR
        )
        est_batches = max(1, math.ceil(est_prompts_total / prompts_per_batch))

        if self.verbose:
            logger.info("Remote concurrent batch generation:")
            logger.info(f"  Target rows       : {n_rows}")
            logger.info(f"  Rows per prompt   : {rows_per_prompt}")
            logger.info(f"  Prompts per batch : {prompts_per_batch}")
            logger.info(f"  Max concurrency   : {self.concurrent_requests}")
            logger.info(
                f"  Estimated batches : ~{est_batches} "
                f"(assuming {_INITIAL_DECODE_RATE:.0%} decode rate)"
            )

        with tqdm(total=n_rows, desc="Generating rows", unit="row", ncols=100) as pbar:
            while total_rows_generated < n_rows:
                rows_remaining = n_rows - total_rows_generated

                # -- Estimate prompts needed, compensating for decode loss --
                if total_prompts_sent > 0 and total_rows_generated > 0:
                    # Use observed average rows per prompt
                    avg_rows_per_prompt = total_rows_generated / total_prompts_sent
                else:
                    # First batch: assume conservative decode rate
                    avg_rows_per_prompt = rows_per_prompt * _INITIAL_DECODE_RATE

                if avg_rows_per_prompt > 0:
                    prompts_needed = math.ceil(
                        rows_remaining / avg_rows_per_prompt * _OVERSHOOT_FACTOR
                    )
                else:
                    # Fallback: just saturate the server
                    prompts_needed = prompts_per_batch

                # Clamp: at least 1, at most prompts_per_batch
                num_prompts_this_batch = max(1, min(prompts_per_batch, prompts_needed))

                # ---- Build all prompts for this batch ----
                prompts_batch: List[str] = []
                for _ in range(num_prompts_this_batch):
                    prompt = self.prompt_builder.build_generation_prompt(
                        demo_data=self.train_data,
                        n_samples=rows_per_prompt,
                        conditional=conditional,
                        mode=self.mode,
                    )
                    prompts_batch.append(prompt)

                # ---- Fire all prompts concurrently via HTTP ----
                # Create a Langfuse span for this batch if tracing is active
                lf_batch_span = self._lf.start_span(
                    name=f"batch-{batch_idx + 1}",
                    parent=langfuse_root,
                    input={
                        "num_prompts": num_prompts_this_batch,
                        "rows_remaining": rows_remaining,
                        "avg_rows_per_prompt": round(avg_rows_per_prompt, 2),
                    },
                    metadata={"batch_idx": batch_idx},
                )

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
                    langfuse_parent=lf_batch_span,
                )
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                total_prompts_sent += num_prompts_this_batch

                # ---- Decode responses (with Langfuse span) ----
                decoded_frames = self._decode_texts(
                    generated_texts, rows_per_prompt, langfuse_parent=lf_batch_span
                )

                # ---- Auto-retry if entire batch decoded to 0 rows ----
                if not decoded_frames:
                    retry_max_tokens = min(max_tokens * 2, 131072)
                    logger.warning(
                        f"[RemoteVLLM] Batch {batch_idx + 1}: 0 rows decoded with "
                        f"max_tokens={max_tokens}. Retrying with max_tokens={retry_max_tokens}."
                    )
                    retry_texts = self.generator.generate_batch(
                        prompts=prompts_batch,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        max_tokens=retry_max_tokens,
                        presence_penalty=presence_penalty,
                        repetition_penalty=repetition_penalty,
                        langfuse_parent=lf_batch_span,
                    )
                    decoded_frames = self._decode_texts(
                        retry_texts, rows_per_prompt, langfuse_parent=lf_batch_span
                    )
                    if not decoded_frames:
                        logger.warning(
                            f"[RemoteVLLM] Batch {batch_idx + 1}: retry also produced 0 rows. "
                            f"Skipping batch."
                        )

                batch_rows = sum(len(df) for df in decoded_frames)

                # Close the batch span with results
                self._lf.end_span(
                    lf_batch_span,
                    output={
                        "rows_decoded": batch_rows,
                        "batch_time_s": round(batch_time, 2),
                    },
                )

                for df in decoded_frames:
                    all_rows.append(df)
                    total_rows_generated += len(df)

                # ---- Guard against infinite loop on persistent decode failure ----
                if batch_rows == 0:
                    consecutive_empty += 1
                    if consecutive_empty >= _MAX_CONSECUTIVE_EMPTY:
                        logger.error(
                            f"[RemoteVLLM] {_MAX_CONSECUTIVE_EMPTY} consecutive batches "
                            f"produced 0 rows. Aborting generation. "
                            f"Got {total_rows_generated}/{n_rows} rows so far."
                        )
                        break
                else:
                    consecutive_empty = 0

                # ---- Progress update ----
                rows_per_sec = batch_rows / batch_time if batch_time > 0 else 0.0
                avg_rpp = (
                    total_rows_generated / total_prompts_sent if total_prompts_sent > 0 else 0.0
                )
                pbar.set_postfix(
                    {
                        "batch": f"{batch_idx + 1}",
                        "speed": f"{rows_per_sec:.0f} rows/s",
                        "prompts": num_prompts_this_batch,
                        "rows/prompt": f"{avg_rpp:.1f}",
                    }
                )
                pbar.update(min(batch_rows, n_rows - pbar.n))

                if self.verbose:
                    logger.info(
                        f"Batch {batch_idx + 1}: {batch_rows} rows decoded | "
                        f"{num_prompts_this_batch} prompts sent | "
                        f"{batch_time:.2f}s | {rows_per_sec:.1f} rows/s | "
                        f"avg {avg_rpp:.1f} rows/prompt"
                    )

                batch_idx += 1

        # ---- Summary ----
        if batch_times:
            total_time = sum(batch_times)
            throughput = total_rows_generated / total_time if total_time > 0 else 0.0
            logger.info(
                f"\nGeneration complete: {len(batch_times)} batches | "
                f"{total_prompts_sent} prompts sent | "
                f"{total_rows_generated} rows | {throughput:.1f} rows/s"
            )

        return all_rows

    # ------------------------------------------------------------------
    # Langfuse: prompt management helpers
    # ------------------------------------------------------------------

    def register_prompt_template(self) -> Optional[Any]:
        """
        Register the current prompt template in Langfuse for versioning.

        Captures the structured prompt format used by PromptBuilder so that
        prompt changes can be tracked across experiments.

        Returns the created prompt object, or None.
        """
        if self.prompt_builder is None:
            logger.warning("Cannot register prompt — call profile() first")
            return None

        # Build a representative prompt to capture the template
        template = (
            "## Task: Generate Synthetic Tabular Data\n\n"
            "Generate realistic, diverse, and UNIQUE synthetic tabular data that "
            "matches the schema and statistical patterns below.\n"
            "IMPORTANT: Generate rows that are different from the examples to "
            "avoid privacy leakage.\n\n"
            "## Schema and Statistics:\n{{schema_stats}}\n\n"
            "## Examples ({{k_shots}} rows):\n{{examples}}\n\n"
            "## Instructions:\n"
            "Generate **only** {{n_samples}} NEW and UNIQUE synthetic row(s).\n\n"
            "**Output Format (CRITICAL)**:\n"
            "Output {{n_samples}} rows, one per line, numbered 1 to {{n_samples}}.\n"
            "Format: NUMBER. column1 is value1, column2 is value2, ..., "
            "columnN is valueN\n\n"
            "**Requirements**:\n"
            "- Generate UNIQUE rows (different from examples)\n"
            "- Follow the EXACT format: NUMBER. col1 is val1, col2 is val2, ...\n"
            "- Respect column types and statistical ranges\n"
            "- Maintain realistic correlations between features\n"
            "- Ensure diversity in generated samples\n"
            "- Output exactly {{n_samples}} rows\n\n"
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

    # ------------------------------------------------------------------
    # Langfuse: dataset registration
    # ------------------------------------------------------------------

    def register_dataset(
        self,
        dataset_name: str,
        description: Optional[str] = None,
        sample_items: int = 5,
    ) -> Optional[Any]:
        """
        Register the profiled real dataset in Langfuse.

        Creates a Langfuse dataset and optionally adds sample items from the
        real data for reference.

        Args:
            dataset_name: Name for the dataset in Langfuse.
            description: Human-readable description.
            sample_items: Number of sample rows to register as dataset items.

        Returns:
            The Langfuse dataset object, or None.
        """
        if self.dataset_profile is None or self.train_data is None:
            logger.warning("Cannot register dataset — call profile() first")
            return None

        ds_metadata = {
            "n_rows": str(self.dataset_profile.n_rows),
            "n_cols": str(self.dataset_profile.n_cols),
            "target_column": self.dataset_profile.target_column or "",
            "column_types": json.dumps(
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

    # ------------------------------------------------------------------
    # Langfuse: score attachment
    # ------------------------------------------------------------------

    def push_evaluation_scores(
        self,
        scores: Dict[str, float],
        comment: Optional[str] = None,
    ) -> None:
        """
        Push evaluation metric scores to the last generation trace.

        Typically called after running ml_eval.py or the evaluator CLI.

        Args:
            scores: Dict of {metric_name: value}, e.g.
                    ``{"xgboost_f1": 0.87, "column_shape": 0.92}``.
            comment: Optional free-text comment.
        """
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
        """
        Save configuration to a JSON file.

        Args:
            path: File path to write.
        """
        config: Dict[str, Any] = {
            "model": self.model_name,
            "base_url": self.base_url,
            "mode": self.mode,
            "float_precision": self.float_precision,
            "concurrent_requests": self.concurrent_requests,
            "request_timeout": self.request_timeout,
            "enable_thinking": self.enable_thinking,
            "langfuse_enabled": self.langfuse_enabled,
            "k_shots": self.k_shots,
        }
        # Include trace ID if available, so evaluation can link back
        if self._last_trace_id:
            config["last_trace_id"] = self._last_trace_id
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        if self.verbose:
            logger.info(f"Config saved to {path}")

    @classmethod
    def from_config(
        cls,
        path: Union[str, Path],
        api_key: Optional[str] = None,
        verbose: bool = False,
    ) -> "TabGenRemoteVLLM":
        """
        Load from a JSON configuration file.

        Args:
            path: Path to the JSON config previously saved with save_config().
            api_key: Optional API key (not stored in config for security).
            verbose: Enable verbose logging.

        Returns:
            Configured TabGenRemoteVLLM instance.
        """
        with open(path) as f:
            config = json.load(f)
        # Remove non-constructor keys
        config.pop("last_trace_id", None)
        return cls(**config, api_key=api_key, verbose=verbose)
