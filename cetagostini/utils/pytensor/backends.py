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
    """Build the MLX linker mode for Gemma 3n graphs.

    Returns PyTensor 3.1.2's built-in ``MLX`` mode, which retains include
    tags ``fast_run`` and ``mlx`` with fusion excluded.

    This function does **not** mutate the built-in Clip dispatch registry.
    Gemma's AltUp clip sites use the repository-local
    :func:`mlx_compat.clip_symbolic` symbolic helper instead.

    Raises
    ------
    ImportError
        If the installed PyTensor build does not provide the MLX linker.
    """
    try:
        from pytensor.compile.mode import MLX
    except ImportError as exc:
        raise ImportError(
            "The installed PyTensor build does not provide the MLX linker"
        ) from exc

    return MLX


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
