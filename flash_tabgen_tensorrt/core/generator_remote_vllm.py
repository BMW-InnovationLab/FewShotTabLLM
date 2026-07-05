"""
Remote vLLM Generator Backend for TabGen

Sends generation requests to a remote vLLM server that exposes the OpenAI-compatible
/v1/chat/completions endpoint.  Fully replaces the local vLLM in-process engine with
HTTP calls, eliminating the need for a GPU on the client machine.

Key design decisions:
- Uses /v1/chat/completions (NOT /v1/completions) so that ``chat_template_kwargs``
  is honoured by the server.  This is *critical* for reasoning models (Qwen3-thinking,
  DeepSeek-R1): the ``enable_thinking=False`` flag only takes effect on the chat
  endpoint.  With /v1/completions the flag was silently ignored.
- When thinking is enabled, the server returns the reasoning in ``message.reasoning``
  (a separate field) and the actual answer in ``message.content``.  The ``max_tokens``
  budget covers both thinking *and* the answer — if the reasoning phase exhausts the
  budget ``message.content`` comes back as ``None`` and ``finish_reason`` is
  ``"length"``.  The generator detects this and returns an empty string so the
  caller's retry / overshoot logic can compensate.
- A ``_strip_thinking_tokens`` regex is retained as a safety-net for servers that
  inline ``<think>…</think>`` tags inside ``message.content`` instead of using the
  separate ``message.reasoning`` field.
- Async HTTP (httpx.AsyncClient) for maximum concurrency: all prompts in a batch are
  fired simultaneously and awaited together via asyncio.gather.
- Semaphore-based concurrency cap: --concurrent-requests (default 300) prevents
  overwhelming the server.
- Exponential backoff retry on 429 / 503 / connection errors.
- Drop-in replacement: exposes the same generate() / generate_batch() / cleanup()
  interface as GeneratorVLLM.
- Optional Langfuse observability: when a ``langfuse_parent`` is passed into
  ``generate_batch()``, each LLM call is logged as a *generation* observation
  with full input/output, token usage, and model parameters.
"""

import asyncio
import logging
import re
import time
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Compiled once — strips <think>...</think> blocks produced by reasoning models
# (e.g. Qwen3-thinking, DeepSeek-R1).  The block may span multiple lines and
# the model may emit it with or without surrounding whitespace.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# ---------------------------------------------------------------------------
# Retry / backoff config
# ---------------------------------------------------------------------------
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds; doubles each retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RemoteVLLMError(RuntimeError):
    """Raised when the remote vLLM server returns an unrecoverable error."""


class GeneratorRemoteVLLM:
    """
    Remote vLLM text generator that sends requests to an OpenAI-compatible server.

    Uses the /v1/chat/completions endpoint so that ``chat_template_kwargs``
    (including ``enable_thinking``) is properly honoured by the server.

    Provides the same interface as GeneratorVLLM (generate / generate_batch / cleanup)
    so it can be swapped in transparently by TabGenRemoteVLLM.

    Args:
        base_url: Base URL of the remote vLLM server, e.g. "http://my-server:8000".
                  The /v1/chat/completions path is appended automatically.
        model: Model name as known by the server (must match what the server loaded).
        api_key: Optional Bearer token if the server requires auth.
        concurrent_requests: Maximum number of in-flight HTTP requests at a time.
                             Tune this to match your server's capacity.
        request_timeout: Per-request timeout in seconds.  Long generation calls need
                         a generous value (default: 600 s).
        enable_thinking: When False (default), sends ``enable_thinking=False`` via
                         ``chat_template_kwargs`` to suppress reasoning blocks.
        verbose: Emit debug-level logs for every request/response.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        concurrent_requests: int = 300,
        request_timeout: float = 600.0,
        enable_thinking: bool = False,
        guided_json: Optional[dict] = None,
        verbose: bool = False,
    ):
        # Normalise the URL so callers can pass with or without a trailing slash
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.model = model
        self.api_key = api_key
        self.concurrent_requests = concurrent_requests
        self.request_timeout = request_timeout
        self.enable_thinking = enable_thinking
        self.guided_json = guided_json
        self.verbose = verbose

        # Build default headers
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        if self.verbose:
            logger.info("Remote vLLM generator initialised")
            logger.info(f"  Endpoint : {self.chat_url}")
            logger.info(f"  Model    : {self.model}")
            logger.info(f"  Max concurrent requests: {self.concurrent_requests}")
            logger.info(f"  Request timeout: {self.request_timeout}s")
            logger.info(f"  Enable thinking: {self.enable_thinking}")

    # ------------------------------------------------------------------
    # Public interface (mirrors GeneratorVLLM)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int = 2048,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """
        Generate text for a single prompt (synchronous convenience wrapper).

        Delegates to generate_batch([prompt]) and returns the single result.

        Args:
            prompt: Input prompt text.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling.
            min_p: Minimum probability threshold.
            max_tokens: Maximum tokens to generate.
            presence_penalty: Token presence penalty.
            repetition_penalty: Token repetition penalty.
            stop: Optional stop sequences.
            enable_thinking: Override instance-level enable_thinking for this call.

        Returns:
            Generated text as a plain string (empty string on failure).
        """
        results = self.generate_batch(
            prompts=[prompt],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            stop=stop,
            enable_thinking=enable_thinking,
        )
        return results[0] if results else ""

    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int = 2048,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
        enable_thinking: Optional[bool] = None,
        langfuse_parent: Optional[Any] = None,
    ) -> List[str]:
        """
        Generate text for multiple prompts concurrently.

        All prompts are dispatched in parallel up to ``concurrent_requests`` at a time,
        using asyncio + httpx.AsyncClient.  The method is synchronous from the caller's
        perspective (it calls asyncio.run internally).

        Args:
            prompts: List of input prompt texts.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling.
            min_p: Minimum probability threshold.
            max_tokens: Maximum tokens to generate per prompt.
            presence_penalty: Token presence penalty.
            repetition_penalty: Token repetition penalty.
            stop: Optional stop sequences.
            enable_thinking: Override instance-level enable_thinking for this batch.
            langfuse_parent: Optional Langfuse trace or span to attach generation
                             observations to.  When ``None``, no tracing is performed.

        Returns:
            List of generated texts (same order as prompts).  Empty string for any
            prompt that failed after all retries.
        """
        if not prompts:
            return []

        # Resolve enable_thinking: per-call override > instance default
        _enable_thinking = self.enable_thinking if enable_thinking is None else enable_thinking

        sampling_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            stop=stop,
            enable_thinking=_enable_thinking,
        )

        return asyncio.run(
            self._async_generate_batch(prompts, langfuse_parent=langfuse_parent, **sampling_kwargs)
        )

    def cleanup(self) -> None:
        """
        No-op.  Provided for interface parity with GeneratorVLLM.

        The httpx client is created per-batch inside the async context manager
        and closed automatically; there is nothing persistent to clean up.
        """
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_thinking_tokens(text: str) -> str:
        """
        Remove inline ``<think>...</think>`` reasoning blocks from completion text.

        Most vLLM servers return reasoning in a separate ``message.reasoning``
        field (handled in ``_async_generate_single``).  This method is a
        safety-net for servers that instead embed ``<think>`` tags directly
        inside ``message.content``.

        Args:
            text: Raw completion text from the server.

        Returns:
            Text with all ``<think>…</think>`` spans removed and
            leading/trailing whitespace stripped.
        """
        return _THINK_RE.sub("", text).strip()

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _async_generate_batch(
        self,
        prompts: List[str],
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        max_tokens: int,
        presence_penalty: float,
        repetition_penalty: float,
        stop: Optional[List[str]],
        enable_thinking: bool,
        langfuse_parent: Optional[Any] = None,
    ) -> List[str]:
        """
        Dispatch all prompts concurrently, bounded by a semaphore.

        Returns results in the same order as `prompts`.
        """
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for the remote-vllm backend. Install it with: pip install httpx"
            ) from exc

        semaphore = asyncio.Semaphore(self.concurrent_requests)

        timeout = httpx.Timeout(
            connect=30.0,
            read=self.request_timeout,
            write=60.0,
            pool=self.request_timeout,  # pool timeout must cover the full queue wait
        )

        # Connection pool sized to match max concurrency so we never queue at the
        # transport layer.  Default httpx limit is 100 which would bottleneck 300+
        # concurrent requests.
        limits = httpx.Limits(
            max_connections=self.concurrent_requests + 10,
            max_keepalive_connections=self.concurrent_requests,
            keepalive_expiry=30.0,
        )

        # A single shared client for all requests in this batch (connection pooling).
        # trust_env=False prevents httpx from picking up HTTP_PROXY / HTTPS_PROXY
        # environment variables, which can redirect internal hostnames through a
        # corporate proxy and cause DNS resolution failures.
        async with httpx.AsyncClient(
            headers=self._headers,
            timeout=timeout,
            limits=limits,
            http2=False,  # keep HTTP/1.1 for maximum vLLM server compat
            trust_env=False,
        ) as client:
            tasks = [
                self._async_generate_single(
                    client=client,
                    semaphore=semaphore,
                    prompt=prompt,
                    prompt_index=i,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                    stop=stop,
                    enable_thinking=enable_thinking,
                    langfuse_parent=langfuse_parent,
                )
                for i, prompt in enumerate(prompts)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Normalise: convert exceptions to empty strings and log them
        texts: List[str] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"[RemoteVLLM] Prompt {i} failed: {result}")
                texts.append("")
            else:
                texts.append(result)

        return texts

    async def _async_generate_single(
        self,
        client,  # httpx.AsyncClient
        semaphore: asyncio.Semaphore,
        prompt: str,
        prompt_index: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        max_tokens: int,
        presence_penalty: float,
        repetition_penalty: float,
        stop: Optional[List[str]],
        enable_thinking: bool,
        langfuse_parent: Optional[Any] = None,
    ) -> str:
        """
        Send one chat completion request, with exponential-backoff retry.

        Acquires the semaphore before making the HTTP call so total concurrency
        is capped at ``self.concurrent_requests``.

        Uses /v1/chat/completions with the prompt wrapped as a user message.
        This is required so that ``chat_template_kwargs`` (including
        ``enable_thinking``) is honoured by the server.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        # vLLM guided JSON decoding: forces the model to produce a valid JSON
        # object matching the provided schema.  This eliminates parsing failures.
        if self.guided_json is not None:
            payload["guided_json"] = self.guided_json

        # Setting enable_thinking=False via chat_template_kwargs prevents reasoning
        # models from emitting long <think>…</think> blocks that consume the token
        # budget without contributing decodable tabular rows.
        # This ONLY works on /v1/chat/completions — the /v1/completions endpoint
        # silently ignores it.
        if not enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        async with semaphore:
            for attempt in range(_MAX_RETRIES):
                try:
                    t0 = time.monotonic()
                    response = await client.post(self.chat_url, json=payload)
                    elapsed = time.monotonic() - t0

                    if response.status_code == 200:
                        data = response.json()
                        choice = data["choices"][0]
                        message = choice["message"]
                        finish_reason = choice.get("finish_reason", "")
                        usage = data.get("usage", {})

                        # -------------------------------------------------
                        # Extract content — handle thinking-model responses
                        # -------------------------------------------------
                        # vLLM returns thinking in ``message.reasoning``
                        # (separate field) and the actual answer in
                        # ``message.content``.  When the reasoning phase
                        # exhausts the ``max_tokens`` budget the content
                        # field is ``None`` and finish_reason is "length".
                        content = message.get("content") or ""
                        reasoning = message.get("reasoning") or ""

                        # Safety-net: strip inline <think> tags if the
                        # server embeds them inside content instead of
                        # using the separate reasoning field.
                        text = self._strip_thinking_tokens(content)

                        # Detect thinking-truncation: reasoning exists but
                        # content is empty because the model ran out of
                        # tokens during (or right after) the thinking phase.
                        if not text and reasoning and finish_reason == "length":
                            logger.warning(
                                f"[RemoteVLLM] Thinking exhausted the token "
                                f"budget (max_tokens={max_tokens}, "
                                f"completion_tokens="
                                f"{usage.get('completion_tokens', '?')}). "
                                f"Reasoning was {len(reasoning)} chars but "
                                f"content is empty. Consider increasing "
                                f"max_tokens or disabling thinking."
                            )

                        if self.verbose:
                            reasoning_info = (
                                f" | reasoning={len(reasoning)} chars" if reasoning else ""
                            )
                            logger.info(
                                f"[RemoteVLLM] Prompt {prompt_index}: "
                                f"status=200 | {elapsed:.2f}s | "
                                f"finish={finish_reason} | "
                                f"tokens={usage.get('completion_tokens', '?')}"
                                f"{reasoning_info}"
                            )
                            # Log a preview of the first response per batch
                            # for debugging decode issues (first 300 chars)
                            if text:
                                preview = text[:300].replace("\n", "\\n")
                                logger.debug(f"[RemoteVLLM] Response preview: {preview!r}")

                        # -------------------------------------------------
                        # Langfuse v4: log generation observation
                        # -------------------------------------------------
                        if langfuse_parent is not None:
                            try:
                                lf_model_params = {
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "top_k": top_k,
                                    "min_p": min_p,
                                    "max_tokens": max_tokens,
                                    "presence_penalty": presence_penalty,
                                    "repetition_penalty": repetition_penalty,
                                    "enable_thinking": enable_thinking,
                                }
                                lf_level = (
                                    "WARNING"
                                    if (not text and finish_reason == "length")
                                    else "DEFAULT"
                                )
                                gen = langfuse_parent.start_observation(
                                    name="llm-generation",
                                    as_type="generation",
                                    model=self.model,
                                    input=[{"role": "user", "content": prompt}],
                                    model_parameters=lf_model_params,
                                    level=lf_level,
                                    metadata={
                                        "prompt_index": prompt_index,
                                        "finish_reason": finish_reason,
                                        "latency_s": round(elapsed, 2),
                                        "reasoning_chars": len(reasoning),
                                        "output_chars": len(text),
                                    },
                                )
                                gen.update(
                                    output=text,
                                    usage_details={
                                        "input": usage.get("prompt_tokens", 0),
                                        "output": usage.get("completion_tokens", 0),
                                        "total": usage.get("total_tokens", 0),
                                    },
                                )
                                gen.end()
                            except Exception as lf_exc:
                                logger.debug(f"[Langfuse] Failed to log generation: {lf_exc}")

                        return text

                    # Retryable server errors
                    if response.status_code in _RETRYABLE_STATUS_CODES:
                        delay = _RETRY_BASE_DELAY * (2**attempt)
                        logger.warning(
                            f"[RemoteVLLM] HTTP {response.status_code} on attempt "
                            f"{attempt + 1}/{_MAX_RETRIES}, retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                        continue

                    # Non-retryable error — fail fast
                    raise RemoteVLLMError(
                        f"Server returned HTTP {response.status_code}: {response.text[:300]}"
                    )

                except RemoteVLLMError:
                    raise  # propagate immediately
                except Exception as exc:
                    # Network errors, timeouts, JSON decode failures
                    delay = _RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        f"[RemoteVLLM] Network error on attempt "
                        f"{attempt + 1}/{_MAX_RETRIES}: "
                        f"{exc!r} — retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

            logger.error(
                f"[RemoteVLLM] All {_MAX_RETRIES} retries exhausted for prompt "
                f"(first 80 chars): {prompt[:80]!r}"
            )
            return ""
