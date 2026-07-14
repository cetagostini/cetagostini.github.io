"""Lazy access to the validated symbolic model builders."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_IMPLEMENTATION_PACKAGE = "cetagostini.utils.pytensor"

_SMOLLM2_SYMBOLS = frozenset({
    "SmolLM2Config",
    "compile_embedding",
    "compile_logits",
    "compile_prefill_layer",
    "compile_decode_layer",
    "build_rope_table",
})

_GEMMA3N_SYMBOLS = frozenset({
    "Gemma3nConfig",
    "compile_decoder_layer",
    "decoder_layer_symbolic",
    "compile_per_layer_projection",
    "per_layer_input_projection",
    "compile_logit_projection",
    "compile_per_chunk_logits",
    "compile_initial_projections",
    "initial_stream_projections",
    "compile_final_unembed",
    "final_unembed",
    "laurel_symbolic",
    "mlp_symbolic",
    "attention_block_symbolic",
})

__all__ = sorted(_SMOLLM2_SYMBOLS | _GEMMA3N_SYMBOLS)


def __getattr__(name: str) -> Any:
    """Load model-specific implementations on first access."""
    if name == "SmolLM2Config":
        module = import_module(f"{_IMPLEMENTATION_PACKAGE}.gguf_weights")
    elif name in _SMOLLM2_SYMBOLS:
        module = import_module(f"{_IMPLEMENTATION_PACKAGE}.smollm2_pytensor")
    elif name in _GEMMA3N_SYMBOLS:
        module = import_module(f"{_IMPLEMENTATION_PACKAGE}.gemma3n_pytensor")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return getattr(module, name)


def __dir__() -> list[str]:
    """Expose lazy public symbols to interactive tooling."""
    return sorted(set(globals()) | set(__all__))
