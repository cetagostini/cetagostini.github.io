"""Python-first inference API for the article's PyTensor runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from .reports import InferenceResult


ProgressCallback = Callable[[str], None]

_IMPLEMENTATION_PACKAGE = "cetagostini.utils.pytensor"


def _load_smollm2_runner():
    """Load the SmolLM2 implementation only when inference is requested."""
    return import_module(f"{_IMPLEMENTATION_PACKAGE}.run_smollm2_pytensor")


def _load_gemma3n_runner():
    """Load the Gemma 3n implementation only when inference is requested."""
    return import_module(f"{_IMPLEMENTATION_PACKAGE}.run_gemma3n_pytensor")


@runtime_checkable
class InferenceModel(Protocol):
    """Protocol implemented by reusable model descriptors."""

    def infer(
        self,
        input: str,
        max_tokens: int,
        reference: bool | None,
        progress: ProgressCallback | None,
    ) -> InferenceResult:
        """Run inference and return a structured result."""
        ...


@dataclass(frozen=True)
class SmolLM2:
    """A pinned SmolLM2 GGUF executed by PyTensor's MLX backend."""

    model_path: Path
    cache_capacity: int = 256
    verify_hash: bool = True
    reference: bool = True
    backend: str = "mlx"

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path).expanduser())

        if self.cache_capacity < 1:
            raise ValueError("cache_capacity must be at least 1")

        if self.backend.lower() != "mlx":
            raise ValueError("SmolLM2 currently supports only the PyTensor MLX backend")

    @classmethod
    def from_gguf(
        cls,
        model_path: str | Path,
        cache_capacity: int = 256,
        verify_hash: bool = True,
        reference: bool = True,
        backend: str = "mlx",
    ) -> SmolLM2:
        """Create a reusable descriptor for a pinned SmolLM2 GGUF file."""
        return cls(
            model_path=Path(model_path),
            cache_capacity=cache_capacity,
            verify_hash=verify_hash,
            reference=reference,
            backend=backend,
        )

    def infer(
        self,
        input: str,
        max_tokens: int,
        reference: bool | None = None,
        progress: ProgressCallback | None = None,
    ) -> InferenceResult:
        """Generate text with the PyTensor + MLX runtime."""
        runner = _load_smollm2_runner()

        raw_report = runner.run_inference(
            model_path=self.model_path,
            prompt=input,
            max_tokens=max_tokens,
            cache_capacity=self.cache_capacity,
            reference=reference if reference is not None else self.reference,
            verify_hash=self.verify_hash,
            progress=progress,
        )

        return InferenceResult.from_smollm2_report(raw_report)


@dataclass(frozen=True)
class Gemma3n:
    """A pinned Gemma 3n snapshot with cache-free PyTensor generation."""

    snapshot_path: Path
    backend: str = "c"
    reference: bool = True
    cache_layer_weights: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_path", Path(self.snapshot_path).expanduser())

        normalized = self.backend.lower()
        if normalized not in frozenset({"c", "numba", "mlx"}):
            raise ValueError("Gemma3n backend must be 'c', 'numba', or 'mlx'")

        object.__setattr__(self, "backend", normalized)

    @classmethod
    def from_snapshot(
        cls,
        snapshot_path: str | Path,
        backend: str = "c",
        cache_layer_weights: bool = False,
    ) -> Gemma3n:
        """Create a descriptor for the pinned affine-4 Gemma 3n snapshot."""
        return cls(
            snapshot_path=Path(snapshot_path),
            backend=backend,
            cache_layer_weights=cache_layer_weights,
        )

    def infer(
        self,
        input: str,
        max_tokens: int,
        reference: bool | None = None,
        progress: ProgressCallback | None = None,
    ) -> InferenceResult:
        """Generate by repeatedly evaluating the complete growing prefix."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        if reference is False:
            raise ValueError(
                "Gemma3n inference keeps the MLX-LM differential oracle enabled"
            )

        runner = _load_gemma3n_runner()

        raw_report = runner.run_inference(
            snapshot_path=self.snapshot_path,
            prompt=input,
            backend=self.backend,
            reference_only=False,
            max_tokens=max_tokens,
            cache_layer_weights=self.cache_layer_weights,
            progress=progress,
        )

        return InferenceResult.from_gemma3n_report(raw_report)


def inference(
    model: InferenceModel,
    input: str,
    max_tokens: int,
    reference: bool | None = None,
    progress: ProgressCallback | None = None,
) -> InferenceResult:
    """Run a model through a uniform Python interface.

    Parameters
    ----------
    model : InferenceModel
        A ``SmolLM2`` or ``Gemma3n`` descriptor.
    input : str
        User prompt.
    max_tokens : int
        Maximum generated tokens. Gemma 3n re-evaluates and recompiles the
        complete growing prefix for every token after the first; it does not
        retain a KV cache between tokens.
    reference : bool or None
        Override differential-reference execution when the model supports it.
    progress : callable or None
        Optional callback receiving progress messages.

    Returns
    -------
    InferenceResult
        Generated text/token plus timing and validation evidence.
    """
    if not isinstance(input, str) or not input.strip():
        raise ValueError("input must be a non-empty string")

    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")

    if not isinstance(model, InferenceModel):
        raise TypeError("model must implement the InferenceModel protocol")

    return model.infer(
        input=input,
        max_tokens=max_tokens,
        reference=reference,
        progress=progress,
    )
