"""
TensorRT-LLM Generator: High-performance text generation

Optimized for B200 and other high-end GPUs using TensorRT-LLM.
Based on NVIDIA's TensorRT-LLM documentation: https://nvidia.github.io/TensorRT-LLM/
"""

from typing import List, Dict, Optional, Any
import torch
from pathlib import Path

# TensorRT-LLM imports
try:
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    TENSORRT_AVAILABLE = True
    TENSORRT_IMPORT_ERROR = None
except ImportError as import_error:
    print(
        "Warning: TensorRT-LLM import failed. "
        "Install with: pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com"
    )
    print(f"TensorRT-LLM detailed import error: {import_error}")
    TENSORRT_AVAILABLE = False
    TENSORRT_IMPORT_ERROR = import_error
    # Create dummy classes to prevent import errors when type checking
    class LLM:
        pass
    class SamplingParams:
        pass
    class KvCacheConfig:
        pass

from flash_tabgen_tensorrt.core.data_profiler import DatasetProfile


class GeneratorTensorRT:
    """
    High-performance LLM generator using TensorRT-LLM
    
    Provides significantly better performance than transformers on B200 GPUs
    with optimized memory usage and inference speed.
    """
    
    def __init__(
        self,
        model_name: str,
        profile: DatasetProfile,
        engine_dir: Optional[str] = None,
        max_batch_size: int = 64,
        max_input_len: int = 16384,
        max_output_len: int = 2048,
        dtype: str = "bfloat16",
        use_paged_attention: bool = True,
        use_inflight_batching: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize TensorRT-LLM generator
        
        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen3-30B-A3B-Instruct-2507")
            profile: Dataset profile for schema information
            engine_dir: Directory containing TensorRT engine files (if pre-built)
            max_batch_size: Maximum batch size for inference
            max_input_len: Maximum input sequence length
            max_output_len: Maximum output sequence length
            dtype: Data type for inference (bfloat16, float16, float32)
            use_paged_attention: Enable paged attention for memory efficiency
            use_inflight_batching: Enable in-flight batching for better throughput
            verbose: Enable verbose logging
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT-LLM not available. "
                "Install with: pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com"
            ) from TENSORRT_IMPORT_ERROR
        
        self.model_name = model_name
        self.profile = profile
        self.engine_dir = engine_dir
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.dtype = dtype
        self.use_paged_attention = use_paged_attention
        self.use_inflight_batching = use_inflight_batching
        self.verbose = verbose
        self.kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=0.85,
            enable_block_reuse=True,
            event_buffer_max_size=16384,
        )
        
        # Initialize model
        self.llm = None
        self.tokenizer = None
        self._initialize_model()
        
        if verbose:
            print(f"✅ TensorRT-LLM generator initialized for {model_name}")
            print(f"   Max batch size: {max_batch_size}")
            print(f"   Max input/output length: {max_input_len}/{max_output_len}")
            print(f"   Data type: {dtype}")
            print(f"   Paged attention: {use_paged_attention}")
            print(f"   In-flight batching: {use_inflight_batching}")
    
    def _initialize_model(self):
        """Initialize TensorRT-LLM model"""
        if self.verbose:
            print(f"Initializing TensorRT-LLM model: {self.model_name}")
        
        # Initialize LLM with modern API
        if self.engine_dir and Path(self.engine_dir).exists():
            if self.verbose:
                print(f"Loading pre-built TensorRT engine from: {self.engine_dir}")
            self.llm = LLM(
                model=self.engine_dir,
                dtype=self.dtype,
                tensor_parallel_size=1,
                kv_cache_config=self.kv_cache_config,
            )
        else:
            if self.verbose:
                print(f"Building TensorRT engine from HuggingFace model: {self.model_name}")
            self.llm = LLM(
                model=self.model_name,
                dtype=self.dtype,
                tensor_parallel_size=1,
                trust_remote_code=True,
                kv_cache_config=self.kv_cache_config,
                max_num_tokens=self.max_input_len
            )
        
        # Get tokenizer
        self.tokenizer = self.llm.tokenizer
        
        if self.verbose:
            print("✅ TensorRT-LLM model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        num_return_sequences: int = 1,
        stop_words: Optional[List[str]] = None,
        repetition_penalty: Optional[float] = None,
    ) -> List[str]:
        """
        Generate text using TensorRT-LLM with optimized performance
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to generate
            stop_words: List of stop words
            repetition_penalty: Repetition penalty factor
        
        Returns:
            List of generated text strings
        """
        # Configure sampling parameters
        sampling_params = SamplingParams(
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            max_tokens=max_new_tokens or self.max_output_len,
            temperature=temperature or 0.7,
            top_p=top_p or 0.9,
            top_k=top_k or 50,
            min_p=min_p or 0.0,
            repetition_penalty=repetition_penalty or 1.0,
            stop=stop_words or None,
            n=num_return_sequences,
        )
        
        # Generate with TensorRT-LLM
        try:
            with torch.no_grad():
                outputs = self.llm.generate(
                    prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
        except Exception as e:
            print(f"TensorRT-LLM generation failed: {e}")
            raise e
        
        # Extract generated text
        request_outputs = outputs if isinstance(outputs, list) else [outputs]
        generated_texts: List[str] = []
        for request_output in request_outputs:
            for completion in request_output.outputs:
                text = completion.text or ""
                if text.startswith(prompt):
                    text = text[len(prompt):]
                generated_texts.append(text.strip())
        
        return generated_texts
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_words: Optional[List[str]] = None,
        repetition_penalty: Optional[float] = None,
        min_p: Optional[float] = None,
    ) -> List[List[str]]:
        """
        Generate text for multiple prompts in a single batch
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_words: List of stop words
            repetition_penalty: Repetition penalty factor
        
        Returns:
            List of lists, where each inner list contains generated texts for one prompt
        """
        # Configure sampling parameters
        sampling_params = SamplingParams(
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            max_tokens=max_new_tokens or self.max_output_len,
            temperature=temperature or 0.7,
            top_p=top_p or 0.9,
            top_k=top_k or 50,
            min_p=min_p or 0.0,
            repetition_penalty=repetition_penalty or 1.0,
            stop=stop_words or None,
            n=1,
        )
        
        # Generate batch
        try:
            with torch.no_grad():
                outputs = self.llm.generate(
                    prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
        except Exception as e:
            print(f"TensorRT-LLM batch generation failed: {e}")
            raise e
        
        # Extract generated texts
        request_outputs = outputs if isinstance(outputs, list) else [outputs]
        batch_results: List[List[str]] = []
        for prompt, request_output in zip(prompts, request_outputs):
            prompt_generations: List[str] = []
            for completion in request_output.outputs:
                text = completion.text or ""
                if text.startswith(prompt):
                    text = text[len(prompt):]
                prompt_generations.append(text.strip())
            batch_results.append(prompt_generations or [""])
        
        return batch_results
    
    def save_engine(self, engine_dir: str):
        """
        Save the TensorRT engine for faster loading in future runs
        
        Args:
            engine_dir: Directory to save the engine
        """
        print("Note: Engine saving is handled automatically by TensorRT-LLM")
        print("Pre-built engines can be loaded by specifying engine_dir in constructor")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "max_batch_size": self.max_batch_size,
            "max_input_len": self.max_input_len,
            "max_output_len": self.max_output_len,
            "dtype": self.dtype,
            "use_paged_attention": self.use_paged_attention,
            "use_inflight_batching": self.use_inflight_batching,
            "engine_dir": self.engine_dir,
        }

    def cleanup(self):
        """
        Release TensorRT-LLM resources to avoid hanging between runs.
        """
        if self.verbose:
            print("[Generator] Shutting down TensorRT-LLM Executor...")

        try:
            # 1. Explicitly shutdown the LLM instance if it exists
            # This kills the background C++ threads managing Paged Attention
            if self.llm is not None:
                # Try to find a shutdown method (common in newer TRT-LLM versions)
                if hasattr(self.llm, 'shutdown'):
                    self.llm.shutdown()
                elif hasattr(self.llm, 'unload'):
                    self.llm.unload()

                # Force delete the object
                self.llm = None

            self.tokenizer = None

        except Exception as e:
            if self.verbose:
                print(f"[Generator] Warning during LLM shutdown: {e}")
        finally:
            # 2. Aggressive Garbage Collection
            import gc
            gc.collect()

            # 3. CUDA Context Cleanup
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()  # Critical for cleaning up MPI/IPC handles
                    torch.cuda.synchronize()
            except Exception as e:
                if self.verbose:
                    print(f"[Generator] Warning: CUDA cleanup error: {e}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string
        
        Args:
            text: Input text
        
        Returns:
            Number of tokens
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def estimate_output_tokens(self, num_rows: int, num_columns: int) -> int:
        """
        Estimate the number of output tokens needed for generating rows
        
        Args:
            num_rows: Number of rows to generate
            num_columns: Number of columns in the dataset
        
        Returns:
            Estimated number of output tokens with safety margin
        """
        # Conservative estimate: row number (4) + column separators + data per column
        tokens_per_row = 4 + num_columns * 8  # 8 tokens per column average
        total_tokens = num_rows * tokens_per_row
        
        # Add 30% safety margin for formatting overhead
        return int(total_tokens * 1.3)
