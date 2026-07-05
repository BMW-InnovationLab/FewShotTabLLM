"""
vLLM Generator Backend for TabGen

Direct vLLM engine wrapper for in-process generation.
Provides a clean interface similar to GeneratorTensorRT for consistency.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class GeneratorVLLM:
    """
    vLLM-based text generator for tabular data synthesis.

    Wraps vLLM's LLM class and provides a simplified interface
    that returns plain strings instead of raw vLLM output objects.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        enable_chunked_prefill: bool = False,
        enforce_eager: bool = True,
        guided_json: Optional[dict] = None,
        verbose: bool = False,
    ):
        """
        Initialize vLLM engine.

        Args:
            model_name: HuggingFace model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum context length
            gpu_memory_utilization: GPU memory utilization (0-1)
            trust_remote_code: Whether to trust remote code in model
            enable_chunked_prefill: Enable chunked prefill optimization
            enforce_eager: Enforce eager mode (disable CUDA graphs)
            guided_json: Optional JSON schema dict for guided decoding.
                         When set, vLLM constrains output to valid JSON
                         matching this schema.
            verbose: Enable verbose logging
        """
        from vllm import LLM

        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.guided_json = guided_json
        self.verbose = verbose

        if self.verbose:
            logger.info(f"Initializing vLLM engine for: {model_name}")
            logger.info(f"Tensor parallel size: {tensor_parallel_size}")
            logger.info(f"Max model length: {max_model_len}")
            logger.info(f"GPU memory utilization: {gpu_memory_utilization}")

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            enable_chunked_prefill=enable_chunked_prefill,
            enforce_eager=enforce_eager,
        )

        if self.verbose:
            logger.info("vLLM engine initialized successfully")

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
    ) -> str:
        """
        Generate text for a single prompt.

        Delegates to :meth:`generate_batch` with a single-element list.

        Args:
            prompt: Input prompt text
            temperature: Sampling temperature (Qwen3.5 recommended: 1.0)
            top_p: Nucleus sampling parameter (Qwen3.5 recommended: 0.95)
            top_k: Top-k sampling (Qwen3.5 recommended: 20)
            min_p: Minimum probability threshold
            max_tokens: Maximum tokens to generate
            presence_penalty: Penalty for token presence (Qwen3.5 recommended: 1.5)
            repetition_penalty: Penalty for repeating tokens
            stop: Optional list of stop sequences

        Returns:
            Generated text as a plain string
        """
        results = self.generate_batch(
            [prompt],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            stop=stop,
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
    ) -> List[str]:
        """
        Generate text for multiple prompts in a batch.

        Args:
            prompts: List of input prompt texts
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate per prompt
            stop: Optional list of stop sequences

        Returns:
            List of generated texts as plain strings
        """
        if not prompts:
            return []

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            stop=stop,
        )

        # Apply guided JSON decoding if configured
        if self.guided_json is not None:
            try:
                from vllm.sampling_params import GuidedDecodingParams

                sampling_params.guided_decoding = GuidedDecodingParams(json=self.guided_json)
            except ImportError:
                logger.warning(
                    "GuidedDecodingParams not available in this vLLM version; "
                    "falling back to unconstrained generation"
                )

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text)
            else:
                results.append("")

        return results

    def cleanup(self):
        """
        Clean up resources.

        Note: vLLM handles cleanup automatically, but this method
        is provided for interface consistency with other generators.
        """
        pass
