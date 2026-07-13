"""Backend selection for reusable PyTensor graphs."""

from __future__ import annotations

from pytensor.compile.mode import Mode


def make_c_mode() -> Mode:
    """Build the optimized C virtual-machine mode."""
    return Mode(linker="cvm", optimizer="o2")


def make_numba_mode() -> Mode:
    """Build the Numba linker mode used by the Gemma 3n fixture."""
    return Mode(linker="numba", optimizer="fast_compile")


def make_mlx_mode() -> Mode:
    """Build the MLX linker mode used by the SmolLM2 fixture."""
    try:
        from pytensor.compile.mode import MLX
        from pytensor.graph.rewriting.db import RewriteDatabaseQuery
        from pytensor.link.mlx.linker import MLXLinker
    except ImportError as exc:
        raise ImportError(
            "The installed PyTensor build does not provide the MLX linker"
        ) from exc

    return Mode(
        MLXLinker(),
        RewriteDatabaseQuery(include=["mlx"]).exclude(MLX._optimizer),
    )


def get_mode(backend: str) -> Mode | str:
    """Resolve a backend name without changing the symbolic model."""
    normalized = backend.strip().lower()

    if normalized in {"c", "cvm"}:
        return make_c_mode()

    if normalized == "numba":
        return make_numba_mode()

    if normalized == "mlx":
        return make_mlx_mode()

    if normalized in {"fast_compile", "python"}:
        return "FAST_COMPILE"

    raise ValueError(f"Unknown backend: {backend!r}")
