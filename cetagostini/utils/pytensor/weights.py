"""Lazy access to validated GGUF and safetensors weight loaders."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_IMPLEMENTATION_PACKAGE = "cetagostini.utils.pytensor"

_GGUF_SYMBOLS = frozenset({
    "SmolLM2Config",
    "SmolLM2Weights",
    "build_inventory",
    "build_manifest",
    "load_smollm2_weights",
    "sanitize_weights_report",
})

_SAFETENSORS_SYMBOLS = frozenset({
    "Gemma3nWeightLoader",
    "Gemma3nTextConfig",
    "parse_safetensors_header",
    "parse_text_config",
    "dequantize_affine4",
    "bf16_to_float32",
    "gather_rows",
    "vocab_chunks",
    "substitute_per_layer_ids",
    "build_layer_manifest",
})

__all__ = sorted(_GGUF_SYMBOLS | _SAFETENSORS_SYMBOLS)


def __getattr__(name: str) -> Any:
    """Load format-specific code only when it is requested."""
    if name in _GGUF_SYMBOLS:
        module = import_module(f"{_IMPLEMENTATION_PACKAGE}.gguf_weights")
    elif name in _SAFETENSORS_SYMBOLS:
        module = import_module(f"{_IMPLEMENTATION_PACKAGE}.gemma3n_weights")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return getattr(module, name)


def __dir__() -> list[str]:
    """Expose lazy public symbols to interactive tooling."""
    return sorted(set(globals()) | set(__all__))
