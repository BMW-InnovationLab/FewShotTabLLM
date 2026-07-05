"""
Langfuse observability utilities for flash-tabgen.

Centralises all Langfuse operations so that every TabGen backend (remote-vLLM,
local vLLM, TensorRT) can share the same fail-safe tracing, prompt management,
dataset registration, and score-pushing logic.

All public functions are **fail-safe**: if the ``langfuse`` package is missing,
environment variables are not configured, or any Langfuse API call fails, the
functions log a warning/debug message and return ``None`` (or silently skip the
operation).  Generation is **never** interrupted by tracing failures.

Typical usage inside a TabGen class::

    from flash_tabgen_tensorrt.core.langfuse_utils import LangfuseManager

    lf = LangfuseManager(enabled=True, session_id="my-experiment")
    lf.init()  # lazy — safe to call even if langfuse is missing

    root = lf.start_span("tabgen-generate", input={...})
    ...
    lf.end_span(root, output={...})
    lf.score_trace(root, name="xgboost_f1", value=0.87)
    lf.flush()
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class LangfuseManager:
    """
    Fail-safe wrapper around the Langfuse Python SDK.

    Provides helpers for:
    - Client initialisation (from env vars)
    - Span / generation lifecycle management
    - Prompt management (create, fetch, compile)
    - Dataset / dataset-item registration
    - Score attachment to traces

    Args:
        enabled: Master switch.  When ``False``, all operations are no-ops.
        session_id: Optional session ID to group traces (e.g. experiment name).
        tags: Optional list of tags applied to every trace.
        metadata: Optional metadata dict applied to every trace.
    """

    def __init__(
        self,
        enabled: bool = True,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.enabled = enabled
        self.session_id = session_id
        self.tags = tags or []
        self.metadata = metadata or {}

        self._client: Optional[Any] = None
        self._initialised = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init(self) -> bool:
        """
        Lazily initialise the Langfuse client.

        Returns True if the client is ready, False otherwise.
        Safe to call multiple times — will only initialise once.
        """
        if self._initialised:
            return self._client is not None

        self._initialised = True

        if not self.enabled:
            logger.info("[Langfuse] Tracing disabled via enabled=False")
            return False

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        if not public_key or not secret_key:
            logger.debug(
                "[Langfuse] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — tracing disabled"
            )
            return False

        try:
            from langfuse import Langfuse

            # Ensure the Langfuse host is excluded from HTTP proxy to avoid
            # corporate proxy interference with self-hosted instances.
            self._ensure_no_proxy()

            self._client = Langfuse()
            base_url = os.environ.get("LANGFUSE_BASE_URL", "cloud")
            logger.info(f"[Langfuse] Tracing enabled (base_url={base_url})")
            return True
        except Exception as exc:
            logger.warning(f"[Langfuse] Failed to initialise client: {exc} — tracing disabled")
            self._client = None
            return False

    @property
    def active(self) -> bool:
        """Return True if the client was successfully initialised."""
        return self._client is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_no_proxy() -> None:
        """Ensure ``LANGFUSE_BASE_URL`` host is in ``NO_PROXY`` / ``no_proxy``.

        Corporate environments often route traffic through an HTTP proxy.
        Self-hosted Langfuse instances (e.g. ``http://langfuse.internal:3000``)
        must bypass the proxy to avoid connection failures or 407 errors.

        This method extracts the hostname from ``LANGFUSE_BASE_URL`` and,
        if it is not already covered by the existing ``NO_PROXY`` value,
        appends a wildcard domain suffix to both ``NO_PROXY`` and
        ``no_proxy`` environment variables.
        """
        base_url = os.environ.get("LANGFUSE_BASE_URL", "")
        if not base_url:
            return

        try:
            from urllib.parse import urlparse

            hostname = urlparse(base_url).hostname or ""
        except Exception:
            return

        if not hostname:
            return

        # Build a domain suffix like ".internal" from "langfuse.internal"
        parts = hostname.rsplit(".", 1)
        suffix = f".{parts[-1]}" if len(parts) > 1 else hostname

        for var in ("NO_PROXY", "no_proxy"):
            current = os.environ.get(var, "")
            if suffix in current or hostname in current:
                continue
            os.environ[var] = f"{current},{suffix}" if current else suffix
            logger.debug(f"[Langfuse] Added '{suffix}' to {var}")

    @property
    def client(self) -> Optional[Any]:
        """Return the underlying Langfuse client (or None)."""
        return self._client

    # ------------------------------------------------------------------
    # Span / observation lifecycle
    # ------------------------------------------------------------------

    def start_span(
        self,
        name: str,
        input: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent: Optional[Any] = None,
    ) -> Optional[Any]:
        """
        Start a new span observation.

        If ``parent`` is provided the span is nested under it; otherwise it
        becomes a root-level span (new trace).  Root-level spans automatically
        get ``session_id`` and ``tags`` from the manager's configuration via
        ``propagate_attributes``.

        Returns the span object, or ``None`` on failure.
        """
        if not self.active:
            return None
        try:
            target = parent if parent is not None else self._client

            # Merge instance-level metadata with call-level metadata
            merged_meta = {**self.metadata}
            if metadata:
                merged_meta.update(metadata)

            kwargs: Dict[str, Any] = {
                "name": name,
                "as_type": "span",
            }
            if input is not None:
                kwargs["input"] = input
            if merged_meta:
                kwargs["metadata"] = merged_meta

            # Root-level spans: set session_id and tags via propagate_attributes
            if parent is None and (self.session_id or self.tags):
                try:
                    from langfuse import propagate_attributes

                    prop_kwargs: Dict[str, Any] = {}
                    if self.session_id:
                        prop_kwargs["session_id"] = self.session_id
                    if self.tags:
                        prop_kwargs["tags"] = self.tags
                    with propagate_attributes(**prop_kwargs):
                        span = target.start_observation(**kwargs)
                except ImportError:
                    span = target.start_observation(**kwargs)
            else:
                span = target.start_observation(**kwargs)

            return span
        except Exception as exc:
            logger.debug(f"[Langfuse] Failed to start span '{name}': {exc}")
            return None

    def end_span(
        self,
        span: Optional[Any],
        output: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End a span, optionally setting its output."""
        if span is None:
            return
        try:
            if output is not None:
                span.update(output=output)
            span.end()
        except Exception as exc:
            logger.debug(f"[Langfuse] Failed to end span: {exc}")

    def start_generation(
        self,
        name: str,
        parent: Optional[Any] = None,
        model: Optional[str] = None,
        input: Optional[Any] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Start a generation observation (LLM call)."""
        if not self.active:
            return None
        try:
            target = parent if parent is not None else self._client
            kwargs: Dict[str, Any] = {
                "name": name,
                "as_type": "generation",
            }
            if model:
                kwargs["model"] = model
            if input is not None:
                kwargs["input"] = input
            if model_parameters:
                kwargs["model_parameters"] = model_parameters
            if metadata:
                kwargs["metadata"] = metadata
            return target.start_observation(**kwargs)
        except Exception as exc:
            logger.debug(f"[Langfuse] Failed to start generation '{name}': {exc}")
            return None

    def end_generation(
        self,
        gen: Optional[Any],
        output: Optional[Any] = None,
        usage_details: Optional[Dict[str, int]] = None,
        level: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End a generation observation with output and token usage."""
        if gen is None:
            return
        try:
            update_kwargs: Dict[str, Any] = {}
            if output is not None:
                update_kwargs["output"] = output
            if usage_details:
                update_kwargs["usage_details"] = usage_details
            if level:
                update_kwargs["level"] = level
            if metadata:
                update_kwargs["metadata"] = metadata
            if update_kwargs:
                gen.update(**update_kwargs)
            gen.end()
        except Exception as exc:
            logger.debug(f"[Langfuse] Failed to end generation: {exc}")

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    def score_trace(
        self,
        span: Optional[Any],
        name: str,
        value: Union[float, int, str, bool],
        comment: Optional[str] = None,
        data_type: Optional[str] = None,
    ) -> None:
        """
        Attach a score to the trace that owns ``span``.

        Args:
            span: Any observation belonging to the target trace.
            name: Score name (e.g. ``"xgboost_f1"``).
            value: Numeric, categorical, or boolean score value.
            comment: Optional free-text explanation.
            data_type: ``"NUMERIC"``, ``"CATEGORICAL"``, or ``"BOOLEAN"``.
                       Auto-detected from ``value`` type if omitted.
        """
        if span is None or not self.active:
            return
        try:
            kwargs: Dict[str, Any] = {"name": name, "value": value}
            if comment:
                kwargs["comment"] = comment
            if data_type:
                kwargs["data_type"] = data_type
            span.score_trace(**kwargs)
        except Exception as exc:
            logger.debug(f"[Langfuse] Failed to score trace with '{name}': {exc}")

    def score_trace_by_id(
        self,
        trace_id: str,
        name: str,
        value: Union[float, int, str, bool],
        comment: Optional[str] = None,
        data_type: Optional[str] = None,
    ) -> None:
        """
        Attach a score to a trace by its ID (no live span reference needed).

        This is useful for post-hoc evaluation workflows where only a
        previously-saved trace ID is available (e.g. from a config file).

        Args:
            trace_id: The Langfuse trace ID string.
            name: Score name (e.g. ``"xgboost_f1"``).
            value: Numeric, categorical, or boolean score value.
            comment: Optional free-text explanation.
            data_type: ``"NUMERIC"``, ``"CATEGORICAL"``, or ``"BOOLEAN"``.
                       Auto-detected from ``value`` type if omitted.
        """
        if not self.active or not trace_id:
            return
        try:
            kwargs: Dict[str, Any] = {
                "trace_id": trace_id,
                "name": name,
                "value": value,
            }
            if comment:
                kwargs["comment"] = comment
            if data_type:
                kwargs["data_type"] = data_type
            self._client.create_score(**kwargs)
        except Exception as exc:
            logger.debug(f"[Langfuse] Failed to score trace {trace_id} with '{name}': {exc}")

    def score_observation(
        self,
        observation: Optional[Any],
        name: str,
        value: Union[float, int, str, bool],
        comment: Optional[str] = None,
    ) -> None:
        """Attach a score directly to an observation (span or generation)."""
        if observation is None or not self.active:
            return
        try:
            kwargs: Dict[str, Any] = {"name": name, "value": value}
            if comment:
                kwargs["comment"] = comment
            observation.score(**kwargs)
        except Exception as exc:
            logger.debug(f"[Langfuse] Failed to score observation with '{name}': {exc}")

    # ------------------------------------------------------------------
    # Prompt management
    # ------------------------------------------------------------------

    def get_prompt(
        self,
        name: str,
        prompt_type: str = "text",
        label: str = "production",
        fallback: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Fetch a managed prompt from Langfuse.

        Args:
            name: Prompt name in Langfuse.
            prompt_type: ``"text"`` or ``"chat"``.
            label: Label to fetch (default ``"production"``).
            fallback: If the prompt is not found, return None and log
                      a debug message (caller should fall back to hardcoded).

        Returns:
            Langfuse prompt object, or ``None`` if not found / error.
        """
        if not self.active:
            return None
        try:
            prompt = self._client.get_prompt(name, type=prompt_type, label=label)
            logger.debug(f"[Langfuse] Fetched prompt '{name}' (label={label})")
            return prompt
        except Exception as exc:
            logger.debug(f"[Langfuse] Prompt '{name}' not found or error: {exc} — using fallback")
            return None

    def create_prompt(
        self,
        name: str,
        prompt: Union[str, List[Dict[str, str]]],
        prompt_type: str = "text",
        labels: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Create or update a managed prompt in Langfuse.

        Creating with the same ``name`` auto-increments the version.

        Args:
            name: Prompt name.
            prompt: Prompt text (for ``"text"``) or list of messages (for ``"chat"``).
            prompt_type: ``"text"`` or ``"chat"``.
            labels: Labels to apply (e.g. ``["production"]``).
            config: Arbitrary config dict versioned with the prompt.

        Returns:
            Created prompt object, or ``None`` on failure.
        """
        if not self.active:
            return None
        try:
            kwargs: Dict[str, Any] = {
                "name": name,
                "type": prompt_type,
                "prompt": prompt,
            }
            if labels:
                kwargs["labels"] = labels
            if config:
                kwargs["config"] = config
            result = self._client.create_prompt(**kwargs)
            logger.info(f"[Langfuse] Created/updated prompt '{name}'")
            return result
        except Exception as exc:
            logger.debug(f"[Langfuse] Failed to create prompt '{name}': {exc}")
            return None

    # ------------------------------------------------------------------
    # Datasets & experiment items
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Create a dataset in Langfuse (idempotent — re-creating with same name
        returns the existing dataset).

        Args:
            name: Dataset name.
            description: Human-readable description.
            metadata: Arbitrary metadata dict.

        Returns:
            Dataset object, or ``None`` on failure.
        """
        if not self.active:
            return None
        try:
            kwargs: Dict[str, Any] = {"name": name}
            if description:
                kwargs["description"] = description
            if metadata:
                kwargs["metadata"] = metadata
            ds = self._client.create_dataset(**kwargs)
            logger.info(f"[Langfuse] Created/fetched dataset '{name}'")
            return ds
        except Exception as exc:
            logger.debug(f"[Langfuse] Failed to create dataset '{name}': {exc}")
            return None

    def create_dataset_item(
        self,
        dataset_name: str,
        input: Optional[Dict[str, Any]] = None,
        expected_output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_trace_id: Optional[str] = None,
        source_observation_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Add an item to a Langfuse dataset.

        Args:
            dataset_name: Name of the target dataset.
            input: Input data for this item.
            expected_output: Expected output / ground truth.
            metadata: Arbitrary metadata.
            source_trace_id: Link to a production trace.
            source_observation_id: Link to a specific observation.

        Returns:
            Dataset item object, or ``None`` on failure.
        """
        if not self.active:
            return None
        try:
            kwargs: Dict[str, Any] = {"dataset_name": dataset_name}
            if input is not None:
                kwargs["input"] = input
            if expected_output is not None:
                kwargs["expected_output"] = expected_output
            if metadata:
                kwargs["metadata"] = metadata
            if source_trace_id:
                kwargs["source_trace_id"] = source_trace_id
            if source_observation_id:
                kwargs["source_observation_id"] = source_observation_id
            item = self._client.create_dataset_item(**kwargs)
            return item
        except Exception as exc:
            logger.debug(f"[Langfuse] Failed to create dataset item: {exc}")
            return None

    def get_dataset(self, name: str) -> Optional[Any]:
        """Fetch a dataset by name."""
        if not self.active:
            return None
        try:
            return self._client.get_dataset(name)
        except Exception as exc:
            logger.debug(f"[Langfuse] Failed to fetch dataset '{name}': {exc}")
            return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush all buffered Langfuse data to the server."""
        if not self.active:
            return
        try:
            self._client.flush()
        except Exception as exc:
            logger.debug(f"[Langfuse] Flush failed: {exc}")

    def shutdown(self) -> None:
        """Flush and shut down the Langfuse client."""
        if not self.active:
            return
        try:
            self._client.shutdown()
        except Exception as exc:
            logger.debug(f"[Langfuse] Shutdown failed: {exc}")

    def get_trace_id(self, span: Optional[Any]) -> Optional[str]:
        """
        Extract the trace ID from a span/observation.

        Returns ``None`` if the span is ``None`` or the ID cannot be read.
        """
        if span is None:
            return None
        try:
            # Langfuse v4 observations expose trace_id
            return getattr(span, "trace_id", None)
        except Exception:
            return None

    def get_trace_url(self, span: Optional[Any]) -> Optional[str]:
        """
        Build the Langfuse UI URL for the trace that owns ``span``.

        Returns ``None`` if the trace ID cannot be determined or the base URL
        is not configured.
        """
        trace_id = self.get_trace_id(span)
        if trace_id is None:
            return None
        base_url = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        return f"{base_url}/trace/{trace_id}"
