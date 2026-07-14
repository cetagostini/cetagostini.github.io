"""Reusable symbolic tensor operations shared by the LLM examples."""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt


def rmsnorm_symbolic(
    x: pt.TensorVariable,
    gamma: pt.TensorVariable,
    eps: float = 1e-5,
) -> pt.TensorVariable:
    """Apply learned RMS normalization over the trailing axis."""
    variance = pt.mean(pt.square(x), axis=-1, keepdims=True)
    scale = np.float32(1.0) / pt.sqrt(variance + np.float32(eps))
    return x * scale * gamma


def rms_norm_no_scale(
    x: pt.TensorVariable,
    eps: float = 1e-5,
) -> pt.TensorVariable:
    """Apply RMS normalization without a learned scale."""
    variance = pt.mean(pt.square(x), axis=-1, keepdims=True)
    return x * (np.float32(1.0) / pt.sqrt(variance + np.float32(eps)))


def linear_proj(
    x: pt.TensorVariable,
    weight: pt.TensorVariable,
    batch_size: int,
    sequence_length: int,
    in_features: int,
    out_features: int,
) -> pt.TensorVariable:
    """Project rank-3 activations through a rank-2 matrix.

    Flattening before ``matmul`` avoids backend-specific rank-3 ``Linear``
    behavior while preserving the explicit output shape.
    """
    x_2d = x.reshape((batch_size * sequence_length, in_features))
    output_2d = pt.matmul(x_2d, weight)
    return output_2d.reshape((batch_size, sequence_length, out_features))


def build_rope_table(
    base: float,
    head_dim: int,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build half-split RoPE cosine and sine tables."""
    if base <= 0:
        raise ValueError("base must be positive")
    if head_dim < 2 or head_dim % 2:
        raise ValueError("head_dim must be a positive even integer")
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")

    half = head_dim // 2

    frequencies = np.float32(1.0) / (
        np.float32(base)
        ** (
            np.arange(half, dtype=np.float32) * np.float32(2.0)
            / np.float32(head_dim)
        )
    )

    positions = np.arange(sequence_length, dtype=np.float32)
    angles = np.outer(positions, frequencies)

    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope_symbolic(
    x: pt.TensorVariable,
    cos: pt.TensorVariable,
    sin: pt.TensorVariable,
    head_dim: int,
) -> pt.TensorVariable:
    """Apply half-split RoPE to the trailing axis."""
    if head_dim < 2 or head_dim % 2:
        raise ValueError("head_dim must be a positive even integer")

    half = head_dim // 2

    first_half = x[..., :half]
    second_half = x[..., half:]

    first_rotated = first_half * cos - second_half * sin
    second_rotated = first_half * sin + second_half * cos

    return pt.concatenate([first_rotated, second_rotated], axis=-1)


def causal_mask(sequence_length: int) -> np.ndarray:
    """Build an additive causal mask with shape ``[T, T]``."""
    if sequence_length < 1:
        raise ValueError("seq_len must be at least 1")

    row = np.arange(sequence_length)[None, :]
    column = np.arange(sequence_length)[:, None]

    return np.where(
        row <= column,
        np.float32(0.0),
        np.float32(-np.inf),
    )


def sliding_window_mask(sequence_length: int, window: int) -> np.ndarray:
    """Build an additive causal mask restricted to a trailing window."""
    if sequence_length < 1:
        raise ValueError("seq_len must be at least 1")
    if window < 1:
        raise ValueError("window must be at least 1")

    row = np.arange(sequence_length)[None, :]
    column = np.arange(sequence_length)[:, None]

    valid = (row <= column) & (row >= (column - window + 1))

    return np.where(
        valid,
        np.float32(0.0),
        np.float32(-np.inf),
    )


def gqa_attention(
    q: pt.TensorVariable,
    k: pt.TensorVariable,
    v: pt.TensorVariable,
    mask: pt.TensorVariable,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    batch_size: int,
    sequence_length: int,
    scale: float,
) -> pt.TensorVariable:
    """Compute grouped-query attention with linker-friendly primitives."""
    del head_dim, batch_size, sequence_length

    if n_heads < 1 or n_kv_heads < 1:
        raise ValueError("head counts must be positive")
    if n_heads % n_kv_heads:
        raise ValueError("n_heads must be divisible by n_kv_heads")

    repeats = n_heads // n_kv_heads

    if repeats > 1:
        k_parts = []
        v_parts = []
        for head_idx in range(n_kv_heads):
            k_head = k[:, :, head_idx : head_idx + 1, :]
            v_head = v[:, :, head_idx : head_idx + 1, :]
            k_parts.extend([k_head] * repeats)
            v_parts.extend([v_head] * repeats)

        k = pt.concatenate(k_parts, axis=2)
        v = pt.concatenate(v_parts, axis=2)

    q_expanded = q[:, :, :, None, :]
    k_expanded = k[:, :, :, None, :]

    scores = pt.sum(q_expanded * k_expanded, axis=-1) * np.float32(scale)
    scores = scores + mask

    scores_max = pt.max(scores, axis=-1, keepdims=True)
    scores_exp = pt.exp(scores - scores_max)
    attention_weights = scores_exp / pt.sum(scores_exp, axis=-1, keepdims=True)

    return pt.sum(
        attention_weights[:, :, :, :, None] * v[:, :, :, None, :],
        axis=3,
    )
