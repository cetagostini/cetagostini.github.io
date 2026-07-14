"""Repository-local symbolic clip for Gemma 3n AltUp graphs.

Provides :func:`clip_symbolic` — a pure symbolic clip composed from nested
``pt.where`` that preserves PyTensor's ordered branch semantics (including
reversed bounds where ``lower > upper`` returns ``lower``).

This module intentionally does **not** register any MLX lowerings.  The
symbolic graph uses only standard PyTensor comparison and selection ops
(``<``, ``>``, ``pt.where``) that every PyTensor backend — C, Numba, and
the built-in MLX linker — already lowers natively.

The root ``cetagostini`` package import chain must **not** import
``mlx``, ``pytensor.link.mlx``, or this module.  Everything here is
imported lazily from :mod:`gemma3n_pytensor`.
"""

from __future__ import annotations

import pytensor.tensor as pt


def clip_symbolic(
    x: pt.TensorVariable,
    lower: pt.TensorVariable,
    upper: pt.TensorVariable,
) -> pt.TensorVariable:
    """Pure symbolic clip using nested ``pt.where``.

    Preserves PyTensor's ordered branch semantics: when ``lower > upper``,
    the result is ``lower`` (PyTensor is authority, not NumPy/MLX).

    Implementation::

        clamped_hi = where(x > upper, upper, x)
        result     = where(x < lower, lower, clamped_hi)

    Parameters
    ----------
    x : TensorVariable
        Values to clip.
    lower : TensorVariable
        Lower bound (broadcastable to *x*).
    upper : TensorVariable
        Upper bound (broadcastable to *x*).

    Returns
    -------
    TensorVariable
        Clipped values, same shape as *x* after broadcasting.
    """
    clamped_hi = pt.where(x > upper, upper, x)
    return pt.where(x < lower, lower, clamped_hi)
