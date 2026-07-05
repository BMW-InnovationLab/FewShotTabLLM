"""
Flash TabGen TensorRT-LLM: High-performance synthetic tabular data generation

TensorRT-LLM optimized implementation for maximum GPU utilization on B200 and other high-end GPUs.
Separate from the transformers-based implementation to avoid conflicts.
"""

# Heavy GPU backends are imported lazily so that lightweight sub-modules
# (e.g. TabGenRemoteVLLM) can be used without torch / tensorrt_llm installed.

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "TabGenTensorRT":
        from flash_tabgen_tensorrt.core.tabgen_tensorrt import TabGenTensorRT  # noqa: PLC0415

        return TabGenTensorRT
    if name == "TabGenVLLM":
        from flash_tabgen_tensorrt.core.tabgen_vllm import TabGenVLLM  # noqa: PLC0415

        return TabGenVLLM
    if name == "TabGenRemoteVLLM":
        from flash_tabgen_tensorrt.core.tabgen_remote_vllm import TabGenRemoteVLLM  # noqa: PLC0415

        return TabGenRemoteVLLM
    raise AttributeError(f"module 'flash_tabgen_tensorrt' has no attribute {name!r}")


__all__ = ["TabGenTensorRT", "TabGenVLLM", "TabGenRemoteVLLM"]
