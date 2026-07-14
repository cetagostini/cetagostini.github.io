"""PyTensor symbolic graph builders for SmolLM2 transformer layers.

Constructs compiled functions for prefill and decode passes using
backend-neutral element-wise GQA attention.  All projections use the
flatten-to-rank2 -> matmul -> static-reshape pattern compatible with
both C and Numba linkers.

All tensors are explicitly declared as float32; this module does **not**
mutate ``pytensor.config.floatX`` on import.

Authoritative semantics follow
``mlx_lm.models.llama`` (Apple MLX reference implementation for Llama-style
transformers, of which SmolLM2 is a variant).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmolLM2Config:
    """Frozen configuration for SmolLM2-135M-Instruct (default values)."""

    vocab_size: int = 49152
    hidden_size: int = 576
    n_layers: int = 30
    n_heads: int = 9
    n_kv_heads: int = 3
    head_dim: int = 64
    intermediate_size: int = 1536
    context_length: int = 8192
    rms_eps: float = 1e-5
    rope_theta: float = 100_000.0
    bos: int = 1
    eos: int = 2

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if self.n_kv_heads < 1:
            raise ValueError("n_kv_heads must be >= 1")
        if self.head_dim < 1 or self.head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if self.intermediate_size < 1:
            raise ValueError("intermediate_size must be >= 1")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if self.context_length < 1:
            raise ValueError("context_length must be >= 1")
        if self.rms_eps <= 0:
            raise ValueError("rms_eps must be positive")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")


# ---------------------------------------------------------------------------
# Mode / compile helpers
# ---------------------------------------------------------------------------


def make_c_mode() -> Mode:
    """Build a PyTensor C-linker compilation mode."""
    return Mode(linker="cvm", optimizer="o2")


def make_numba_mode() -> Mode:
    """Build a PyTensor Numba-linker compilation mode."""
    return Mode(linker="numba", optimizer="fast_compile")


def _get_mode(backend: str):
    """Resolve a backend string to a PyTensor ``Mode``.

    Parameters
    ----------
    backend : str
        ``'c'``, ``'numba'``, or ``'FAST_COMPILE'``.

    Returns
    -------
    Mode or str
    """
    if backend == "c":
        return make_c_mode()
    if backend == "numba":
        return make_numba_mode()
    if backend == "FAST_COMPILE":
        return "FAST_COMPILE"
    raise ValueError(f"Unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# Float32 audit
# ---------------------------------------------------------------------------


def audit_float32(fn) -> bool:
    """Walk *fn*'s graph and assert every float variable is ``float32``.

    Parameters
    ----------
    fn : pytensor.compile.function.Function
        A compiled PyTensor function.

    Returns
    -------
    bool
        ``True`` when the audit passes.

    Raises
    ------
    RuntimeError
        If any float variable has a dtype other than ``float32``.
    """
    from pytensor.graph.traversal import ancestors

    for var in ancestors(fn.maker.fgraph.outputs):
        if hasattr(var, "dtype") and var.dtype.startswith("float"):
            if var.dtype != "float32":
                raise RuntimeError(
                    f"Non-float32 variable: {var} has dtype {var.dtype}"
                )
    return True


# ---------------------------------------------------------------------------
# Primitive symbolic functions
# ---------------------------------------------------------------------------


def build_rope_table(
    config: SmolLM2Config, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build NumPy cos/sin RoPE tables with half-split.

    Parameters
    ----------
    config : SmolLM2Config
    seq_len : int

    Returns
    -------
    cos, sin : np.ndarray
        Each of shape ``(seq_len, head_dim // 2)``, dtype float32.
    """
    base = config.rope_theta
    head_dim = config.head_dim
    half = head_dim // 2
    freqs = np.float32(1.0) / (
        np.float32(base)
        ** (np.arange(0, half, dtype=np.float32) * np.float32(2.0) / np.float32(head_dim))
    )
    positions = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def rotate_half(x: pt.TensorVariable, head_dim: int) -> pt.TensorVariable:
    """Rotate the last axis by swapping negated halves.

    Parameters
    ----------
    x : TensorVariable
        Shape ``(..., head_dim)``.
    head_dim : int
        Static head dimension.

    Returns
    -------
    TensorVariable
        ``[-x[..., half:], x[..., :half]]`` concatenated along last axis.
    """
    half = head_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return pt.concatenate([-x2, x1], axis=-1)


def rmsnorm_symbolic(
    x: pt.TensorVariable, gamma: pt.TensorVariable, eps: float
) -> pt.TensorVariable:
    """Symbolic RMSNorm with direct gamma multiplication.

    Works for any rank >= 2; normalisation is over the last axis.

    Parameters
    ----------
    x : TensorVariable
        Input tensor.
    gamma : TensorVariable
        Scale vector broadcastable over the trailing axis.
    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    TensorVariable
        Same shape as *x*.
    """
    variance = pt.mean(pt.square(x), axis=-1, keepdims=True)
    return x * (np.float32(1.0) / pt.sqrt(variance + np.float32(eps))) * gamma


def linear_proj(
    x: pt.TensorVariable,
    W: pt.TensorVariable,
    B: int,
    T: int,
    in_features: int,
    out_features: int,
) -> pt.TensorVariable:
    """Project *x* ``[B, T, in]`` through *W* ``[in, out]`` via rank-2 flatten.

    Parameters
    ----------
    x : TensorVariable
        Shape ``(B, T, in_features)``.
    W : TensorVariable
        Shape ``(in_features, out_features)``.
    B, T, in_features, out_features : int
        Static dimensions.

    Returns
    -------
    TensorVariable
        Shape ``(B, T, out_features)``.
    """
    x_flat = x.reshape((B * T, in_features))
    out_flat = pt.dot(x_flat, W)
    return out_flat.reshape((B, T, out_features))


def apply_rope(
    q: pt.TensorVariable,
    k: pt.TensorVariable,
    cos: pt.TensorVariable,
    sin: pt.TensorVariable,
    head_dim: int,
    seq_len: int,
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    """Apply rotary position embeddings to *q* and *k*.

    Uses the ``rotate_half`` formulation:
    ``rotated = x * cos_full + rotate_half(x) * sin_full``

    Parameters
    ----------
    q : TensorVariable
        Shape ``(B, n_heads, T, head_dim)``.
    k : TensorVariable
        Shape ``(B, n_kv_heads, T, head_dim)``.
    cos, sin : TensorVariable
        Shape ``(T, head_dim // 2)``.
    head_dim : int
        Static head dimension.
    seq_len : int
        Static sequence length (T).

    Returns
    -------
    q_rot, k_rot : TensorVariable
        Same shapes as inputs.
    """
    # Expand cos/sin from half-dim to full-dim by concatenating [cos, cos]
    cos_full = pt.concatenate([cos, cos], axis=-1).reshape(
        (1, 1, seq_len, head_dim)
    )
    sin_full = pt.concatenate([sin, sin], axis=-1).reshape(
        (1, 1, seq_len, head_dim)
    )

    q_rot = q * cos_full + rotate_half(q, head_dim) * sin_full
    k_rot = k * cos_full + rotate_half(k, head_dim) * sin_full
    return q_rot, k_rot


def silu_gated_mlp(
    x: pt.TensorVariable,
    gate_W: pt.TensorVariable,
    up_W: pt.TensorVariable,
    down_W: pt.TensorVariable,
    B: int,
    T: int,
    H: int,
    I: int,
) -> pt.TensorVariable:
    """SiLU-gated MLP: ``down(silu(gate(x)) * up(x))``.

    All projections use the flatten-to-rank2 pattern.

    Parameters
    ----------
    x : TensorVariable
        Shape ``(B, T, H)``.
    gate_W, up_W : TensorVariable
        Shape ``(H, I)``.
    down_W : TensorVariable
        Shape ``(I, H)``.
    B, T, H, I : int
        Static dimensions.

    Returns
    -------
    TensorVariable
        Shape ``(B, T, H)``.
    """
    gate = linear_proj(x, gate_W, B, T, H, I)
    up = linear_proj(x, up_W, B, T, H, I)
    gated = pt.sigmoid(gate) * gate * up
    gated_flat = gated.reshape((B * T, I))
    out_flat = pt.matmul(gated_flat, down_W)
    return out_flat.reshape((B, T, H))


def gqa_attention(
    q: pt.TensorVariable,
    k: pt.TensorVariable,
    v: pt.TensorVariable,
    mask: Optional[pt.TensorVariable],
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    B: int,
    T_q: int,
    T_k: int,
    scale: float,
    is_causal: bool = False,
) -> pt.TensorVariable:
    """GQA attention with head-repeat expansion.

    Parameters
    ----------
    q : TensorVariable
        Shape ``(B, n_heads, T_q, head_dim)``.
    k : TensorVariable
        Shape ``(B, n_kv_heads, T_k, head_dim)``.
    v : TensorVariable
        Shape ``(B, n_kv_heads, T_k, head_dim)``.
    mask : TensorVariable or None
        Additive float mask broadcastable to ``(B, n_heads, T_q, T_k)``.
    n_heads, n_kv_heads, head_dim : int
        Static head counts and dimension.
    B : int
        Static batch size.
    T_q, T_k : int
        Query and key sequence lengths.
    scale : float
        Attention scale factor.
    is_causal : bool
        If True and mask is None, apply causal masking.

    Returns
    -------
    TensorVariable
        Shape ``(B, n_heads, T_q, head_dim)``.
    """
    repeats = n_heads // n_kv_heads
    if repeats > 1:
        k_parts = []
        v_parts = []
        for i in range(n_kv_heads):
            k_head = k[:, i : i + 1, :, :]
            v_head = v[:, i : i + 1, :, :]
            k_parts.extend([k_head] * repeats)
            v_parts.extend([v_head] * repeats)
        k = pt.concatenate(k_parts, axis=1)
        v = pt.concatenate(v_parts, axis=1)

    # Element-wise attention scores
    q_exp = q[:, :, :, None, :]  # [B, n_h, T_q, 1, hd]
    k_exp = k[:, :, None, :, :]  # [B, n_h, 1, T_k, hd]
    scores = pt.sum(q_exp * k_exp, axis=-1) * np.float32(scale)

    if mask is not None:
        scores = scores + mask
    elif is_causal:
        # Build causal mask inline
        causal = np.full((T_q, T_k), -np.inf, dtype=np.float32)
        for i in range(T_q):
            causal[i, : i + 1] = 0.0
        scores = scores + causal.reshape((1, 1, T_q, T_k))

    # Numerically stable softmax
    scores_max = pt.max(scores, axis=-1, keepdims=True)
    scores_exp = pt.exp(scores - scores_max)
    scores_sum = pt.sum(scores_exp, axis=-1, keepdims=True)
    attn_weights = scores_exp / scores_sum

    # Element-wise output: attn_weights @ v
    attn_w_exp = attn_weights[:, :, :, :, None]  # [B, n_h, T_q, T_k, 1]
    v_exp = v[:, :, None, :, :]  # [B, n_h, 1, T_k, hd]
    return pt.sum(attn_w_exp * v_exp, axis=3)


# ---------------------------------------------------------------------------
# Compile functions
# ---------------------------------------------------------------------------


def compile_prefill_layer(
    config: SmolLM2Config,
    batch_size: int,
    seq_len: int,
    backend: str = "c",
):
    """Compile a single SmolLM2 transformer layer for the prefill pass.

    Parameters
    ----------
    config : SmolLM2Config
    batch_size, seq_len : int
        Static batch and sequence dimensions.  Both must be >= 1.
    backend : str
        ``'c'``, ``'numba'``, or ``'FAST_COMPILE'``.

    Raises
    ------
    ValueError
        If ``batch_size < 1`` or ``seq_len < 1``.

    Returns
    -------
    callable
        ``fn(hidden, q_w, k_w, v_w, o_w, gate_w, up_w, down_w,
             in_gamma, post_gamma, cos, sin)
          -> (hidden_out, rotated_k, raw_v)``

        Shapes
        ~~~~~~
        - hidden: ``(B, T, H)``
        - q_w: ``(H, n_heads * head_dim)``
        - k_w, v_w: ``(H, n_kv_heads * head_dim)``
        - o_w: ``(n_heads * head_dim, H)``
        - gate_w, up_w: ``(H, I)``
        - down_w: ``(I, H)``
        - in_gamma, post_gamma: ``(H,)``
        - cos, sin: ``(T, head_dim // 2)``
        - hidden_out: ``(B, T, H)``
        - rotated_k: ``(B, n_kv_heads, T, head_dim)``
        - raw_v: ``(B, n_kv_heads, T, head_dim)``
    """
    if batch_size < 1:
        raise ValueError(
            f"compile_prefill_layer requires batch_size >= 1, got {batch_size}"
        )
    if seq_len < 1:
        raise ValueError(
            f"compile_prefill_layer requires seq_len >= 1, got {seq_len}"
        )

    B, T = batch_size, seq_len
    H = config.hidden_size
    n_h = config.n_heads
    n_kv = config.n_kv_heads
    hd = config.head_dim
    I = config.intermediate_size
    eps = config.rms_eps
    scale = 1.0 / math.sqrt(hd)

    # Symbolic inputs
    hidden = pt.tensor("hidden", shape=(B, T, H), dtype="float32")
    q_w = pt.tensor("q_w", shape=(H, n_h * hd), dtype="float32")
    k_w = pt.tensor("k_w", shape=(H, n_kv * hd), dtype="float32")
    v_w = pt.tensor("v_w", shape=(H, n_kv * hd), dtype="float32")
    o_w = pt.tensor("o_w", shape=(n_h * hd, H), dtype="float32")
    gate_w = pt.tensor("gate_w", shape=(H, I), dtype="float32")
    up_w = pt.tensor("up_w", shape=(H, I), dtype="float32")
    down_w = pt.tensor("down_w", shape=(I, H), dtype="float32")
    in_gamma = pt.tensor("in_gamma", shape=(H,), dtype="float32")
    post_gamma = pt.tensor("post_gamma", shape=(H,), dtype="float32")
    cos = pt.tensor("cos", shape=(T, hd // 2), dtype="float32")
    sin = pt.tensor("sin", shape=(T, hd // 2), dtype="float32")

    # Pre-attention RMSNorm
    normed = rmsnorm_symbolic(hidden, in_gamma, eps)

    # Q/K/V projections
    q_proj = linear_proj(normed, q_w, B, T, H, n_h * hd)
    k_proj = linear_proj(normed, k_w, B, T, H, n_kv * hd)
    v_proj = linear_proj(normed, v_w, B, T, H, n_kv * hd)

    # Reshape to attention format: [B, heads, T, head_dim]
    q = q_proj.reshape((B, T, n_h, hd)).swapaxes(1, 2)
    k = k_proj.reshape((B, T, n_kv, hd)).swapaxes(1, 2)
    v = v_proj.reshape((B, T, n_kv, hd)).swapaxes(1, 2)

    # Apply RoPE to Q and K
    q_rot, k_rot = apply_rope(q, k, cos, sin, hd, T)

    # GQA attention (causal, with scale)
    attn_out = gqa_attention(
        q_rot, k_rot, v, None, n_h, n_kv, hd, B, T, T,
        scale=scale, is_causal=True,
    )

    # Output projection: [B, n_h, T, hd] -> [B, T, n_h*hd] -> [B, T, H]
    attn_flat = attn_out.swapaxes(1, 2).reshape((B * T, n_h * hd))
    o_out = pt.matmul(attn_flat, o_w).reshape((B, T, H))

    # Residual connection
    hidden2 = hidden + o_out

    # Post-attention RMSNorm
    normed2 = rmsnorm_symbolic(hidden2, post_gamma, eps)

    # SiLU-gated MLP
    mlp_out = silu_gated_mlp(normed2, gate_w, up_w, down_w, B, T, H, I)

    # Residual connection
    hidden_out = hidden2 + mlp_out

    inputs = [
        hidden, q_w, k_w, v_w, o_w, gate_w, up_w, down_w,
        in_gamma, post_gamma, cos, sin,
    ]
    outputs = [hidden_out, k_rot, v]
    return pytensor.function(inputs, outputs, mode=_get_mode(backend))


def compile_decode_layer(
    config: SmolLM2Config,
    batch_size: int,
    cache_capacity: int,
    backend: str = "c",
):
    """Compile a single SmolLM2 transformer layer for the decode pass.

    Cache is updated via ``pt.where`` (no symbolic slicing).

    .. note:: **batch_size must equal 1.**

        The current runtime and article target only batch-1 generation.
        Continuous batching is explicitly rejected.  A batched decode
        step would require per-request absolute positions, but this
        builder accepts a single ``(1, head_dim // 2)`` cos/sin pair
        shared across the batch.

    Parameters
    ----------
    config : SmolLM2Config
    batch_size : int
        Must be exactly 1.
    cache_capacity : int
        Fixed KV-cache slot capacity C (must be >= 1).
    backend : str
        ``'c'``, ``'numba'``, or ``'FAST_COMPILE'``.

    Raises
    ------
    ValueError
        If ``batch_size != 1`` or ``cache_capacity < 1``.

    Returns
    -------
    callable
        ``fn(hidden, q_w, k_w, v_w, o_w, gate_w, up_w, down_w,
             in_gamma, post_gamma, old_k, old_v,
             write_mask, attn_mask, cos, sin)
          -> (hidden_out, new_k, new_v)``

        Shapes
        ~~~~~~
        - hidden: ``(1, 1, H)``
        - old_k, old_v: ``(1, n_kv, C, hd)``
        - write_mask: ``(1, 1, C, 1)`` — one-hot float32
        - attn_mask: ``(1, 1, 1, C)`` — additive float32
        - cos, sin: ``(1, hd // 2)``
        - hidden_out: ``(1, 1, H)``
        - new_k, new_v: ``(1, n_kv, C, hd)``
    """
    if batch_size != 1:
        raise ValueError(
            f"compile_decode_layer requires batch_size == 1, got {batch_size}"
        )
    if cache_capacity < 1:
        raise ValueError(
            f"compile_decode_layer requires cache_capacity >= 1, got {cache_capacity}"
        )

    B = 1
    C = cache_capacity
    H = config.hidden_size
    n_h = config.n_heads
    n_kv = config.n_kv_heads
    hd = config.head_dim
    I = config.intermediate_size
    eps = config.rms_eps
    scale = 1.0 / math.sqrt(hd)

    # Symbolic inputs
    hidden = pt.tensor("hidden", shape=(B, 1, H), dtype="float32")
    q_w = pt.tensor("q_w", shape=(H, n_h * hd), dtype="float32")
    k_w = pt.tensor("k_w", shape=(H, n_kv * hd), dtype="float32")
    v_w = pt.tensor("v_w", shape=(H, n_kv * hd), dtype="float32")
    o_w = pt.tensor("o_w", shape=(n_h * hd, H), dtype="float32")
    gate_w = pt.tensor("gate_w", shape=(H, I), dtype="float32")
    up_w = pt.tensor("up_w", shape=(H, I), dtype="float32")
    down_w = pt.tensor("down_w", shape=(I, H), dtype="float32")
    in_gamma = pt.tensor("in_gamma", shape=(H,), dtype="float32")
    post_gamma = pt.tensor("post_gamma", shape=(H,), dtype="float32")
    old_k = pt.tensor("old_k", shape=(B, n_kv, C, hd), dtype="float32")
    old_v = pt.tensor("old_v", shape=(B, n_kv, C, hd), dtype="float32")
    write_mask = pt.tensor("write_mask", shape=(1, 1, C, 1), dtype="float32")
    attn_mask = pt.tensor("attn_mask", shape=(1, 1, 1, C), dtype="float32")
    cos = pt.tensor("cos", shape=(1, hd // 2), dtype="float32")
    sin = pt.tensor("sin", shape=(1, hd // 2), dtype="float32")

    # Pre-attention RMSNorm
    normed = rmsnorm_symbolic(hidden, in_gamma, eps)

    # Q/K/V projections (T=1 for decode)
    q_proj = linear_proj(normed, q_w, B, 1, H, n_h * hd)
    k_proj = linear_proj(normed, k_w, B, 1, H, n_kv * hd)
    v_proj = linear_proj(normed, v_w, B, 1, H, n_kv * hd)

    # Reshape to attention format: [B, heads, 1, head_dim]
    q = q_proj.reshape((B, 1, n_h, hd)).swapaxes(1, 2)
    new_k = k_proj.reshape((B, 1, n_kv, hd)).swapaxes(1, 2)
    new_v = v_proj.reshape((B, 1, n_kv, hd)).swapaxes(1, 2)

    # Apply RoPE to Q and new K (single position)
    q_rot, k_rot = apply_rope(q, new_k, cos, sin, hd, 1)

    # Cache update via pt.where: write_mask is [1, 1, C, 1]
    # Broadcast k_rot [B, n_kv, 1, hd] to [B, n_kv, C, hd] via write_mask
    write_bool = write_mask > np.float32(0.5)
    # Expand k_rot to cache shape: [B, n_kv, 1, hd] -> broadcast to [B, n_kv, C, hd]
    k_rot_expanded = pt.broadcast_to(k_rot, (B, n_kv, C, hd))
    new_v_expanded = pt.broadcast_to(new_v, (B, n_kv, C, hd))
    updated_k = pt.where(write_bool, k_rot_expanded, old_k)
    updated_v = pt.where(write_bool, new_v_expanded, old_v)

    # GQA attention over full cache: q [B, n_h, 1, hd] vs k [B, n_kv, C, hd]
    attn_out = gqa_attention(
        q_rot, updated_k, updated_v, attn_mask, n_h, n_kv, hd, B, 1, C,
        scale=scale,
    )

    # Output projection
    attn_flat = attn_out.swapaxes(1, 2).reshape((B * 1, n_h * hd))
    o_out = pt.matmul(attn_flat, o_w).reshape((B, 1, H))

    # Residual
    hidden2 = hidden + o_out

    # Post-attention RMSNorm
    normed2 = rmsnorm_symbolic(hidden2, post_gamma, eps)

    # SiLU-gated MLP
    mlp_out = silu_gated_mlp(normed2, gate_w, up_w, down_w, B, 1, H, I)

    # Residual
    hidden_out = hidden2 + mlp_out

    inputs = [
        hidden, q_w, k_w, v_w, o_w, gate_w, up_w, down_w,
        in_gamma, post_gamma, old_k, old_v,
        write_mask, attn_mask, cos, sin,
    ]
    outputs = [hidden_out, updated_k, updated_v]
    return pytensor.function(inputs, outputs, mode=_get_mode(backend))


def compile_embedding(
    config: SmolLM2Config,
    batch_size: int,
    seq_len: int,
    backend: str = "c",
):
    """Compile token embedding lookup.

    Parameters
    ----------
    config : SmolLM2Config
    batch_size, seq_len : int
        Static dimensions (both >= 1).
    backend : str

    Returns
    -------
    callable
        ``fn(token_ids, emb_table) -> hidden``

        - token_ids: ``(B, T)`` int32
        - emb_table: ``(V, H)`` float32
        - hidden: ``(B, T, H)`` float32
    """
    if batch_size < 1:
        raise ValueError(
            f"compile_embedding requires batch_size >= 1, got {batch_size}"
        )
    if seq_len < 1:
        raise ValueError(
            f"compile_embedding requires seq_len >= 1, got {seq_len}"
        )

    B, T = batch_size, seq_len
    H = config.hidden_size
    V = config.vocab_size

    token_ids = pt.tensor("token_ids", shape=(B, T), dtype="int32")
    emb_table = pt.tensor("emb_table", shape=(V, H), dtype="float32")

    # Flatten, index, reshape back
    flat_ids = token_ids.reshape((B * T,))
    flat_emb = emb_table[flat_ids]
    hidden = flat_emb.reshape((B, T, H))

    return pytensor.function(
        [token_ids, emb_table], hidden, mode=_get_mode(backend)
    )


def compile_logits(
    config: SmolLM2Config,
    batch_size: int,
    seq_len: int,
    backend: str = "c",
):
    """Compile final RMSNorm + tied logit projection.

    Parameters
    ----------
    config : SmolLM2Config
    batch_size, seq_len : int
        Static dimensions (both >= 1).
    backend : str

    Returns
    -------
    callable
        ``fn(hidden, final_norm_gamma, emb_table) -> logits``

        - hidden: ``(B, T, H)``
        - final_norm_gamma: ``(H,)``
        - emb_table: ``(V, H)``
        - logits: ``(B, T, V)``
    """
    if batch_size < 1:
        raise ValueError(
            f"compile_logits requires batch_size >= 1, got {batch_size}"
        )
    if seq_len < 1:
        raise ValueError(
            f"compile_logits requires seq_len >= 1, got {seq_len}"
        )

    B, T = batch_size, seq_len
    H = config.hidden_size
    V = config.vocab_size
    eps = config.rms_eps

    hidden = pt.tensor("hidden", shape=(B, T, H), dtype="float32")
    final_gamma = pt.tensor("final_norm_gamma", shape=(H,), dtype="float32")
    emb_table = pt.tensor("emb_table", shape=(V, H), dtype="float32")

    # Final RMSNorm
    normed = rmsnorm_symbolic(hidden, final_gamma, eps)

    # Tied logit projection: flatten -> matmul -> reshape
    normed_flat = normed.reshape((B * T, H))
    logits_flat = pt.matmul(normed_flat, emb_table.T)  # [B*T, V]
    logits = logits_flat.reshape((B, T, V))

    return pytensor.function(
        [hidden, final_gamma, emb_table], logits, mode=_get_mode(backend)
    )
