"""PyTensor symbolic graph builders for Gemma3n transformer primitives.

Pure symbolic PyTensor graphs — no custom Ops, no MLX-specific equations.
All weights are accepted as already-dequantized contiguous float32 NumPy
arrays.  Every compiled function uses explicit ``Mode``/linker objects for
the C and Numba backends; this module does **not** mutate
``pytensor.config.floatX`` on import.

Authoritative semantics follow
``mlx_lm.models.gemma3n`` (Apple MLX reference implementation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode

# Smallest positive normal float32, used to keep zero-magnitude streams finite.
# MLX-LM currently guards with finfo.min (the most negative finite value),
# which yields NaNs for an exactly zero projected stream. Production weights do
# not hit that state, but this implementation intentionally handles it safely.
_FINFO_MIN = np.float32(np.finfo(np.float32).tiny)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gemma3nConfig:
    """Frozen configuration for a Gemma3n text model (small-test defaults)."""

    hidden_size: int = 128
    num_hidden_layers: int = 2
    intermediate_size: int = 256
    num_attention_heads: int = 8
    head_dim: int = 32
    rms_norm_eps: float = 1e-5
    vocab_size: int = 256
    num_key_value_heads: int = 2
    sliding_window: int = 16
    rope_local_base_freq: float = 10_000.0
    rope_theta: float = 100_000.0
    final_logit_softcapping: float = 30.0
    activation_sparsity: float = 0.0
    hidden_size_per_layer_input: int = 64
    altup_num_inputs: int = 4
    altup_coef_clip: float = 1.0
    altup_correct_scale: bool = True
    altup_active_idx: int = 0
    laurel_rank: int = 32
    vocab_size_per_layer_input: int = 128

    def __post_init__(self) -> None:
        positive_fields = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "num_key_value_heads": self.num_key_value_heads,
            "sliding_window": self.sliding_window,
            "hidden_size_per_layer_input": self.hidden_size_per_layer_input,
            "altup_num_inputs": self.altup_num_inputs,
            "laurel_rank": self.laurel_rank,
            "vocab_size_per_layer_input": self.vocab_size_per_layer_input,
        }
        invalid = sorted(name for name, value in positive_fields.items() if value < 1)
        if invalid:
            raise ValueError(f"Configuration dimensions must be positive: {invalid}")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if not 0 <= self.altup_active_idx < self.altup_num_inputs:
            raise ValueError("altup_active_idx must select an AltUp stream")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")

    @classmethod
    def from_text_config(cls, text_config) -> "Gemma3nConfig":
        """Construct a ``Gemma3nConfig`` from a ``Gemma3nTextConfig``.

        Parameters
        ----------
        text_config : Gemma3nTextConfig
            Configuration from ``gemma3n_weights``.

        Returns
        -------
        Gemma3nConfig
        """
        return cls(
            hidden_size=text_config.hidden_size,
            num_hidden_layers=text_config.num_hidden_layers,
            intermediate_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            head_dim=text_config.head_dim,
            rms_norm_eps=text_config.rms_norm_eps,
            vocab_size=text_config.vocab_size,
            num_key_value_heads=text_config.num_key_value_heads,
            sliding_window=text_config.sliding_window,
            rope_local_base_freq=text_config.rope_local_base_freq,
            rope_theta=text_config.rope_theta,
            final_logit_softcapping=text_config.final_logit_softcapping,
            activation_sparsity=0.0,
            hidden_size_per_layer_input=text_config.hidden_size_per_layer_input,
            altup_num_inputs=text_config.altup_num_inputs,
            altup_coef_clip=text_config.altup_coef_clip,
            altup_correct_scale=text_config.altup_correct_scale,
            altup_active_idx=text_config.altup_active_idx,
            laurel_rank=text_config.laurel_rank,
            vocab_size_per_layer_input=text_config.vocab_size_per_layer_input,
        )


# ---------------------------------------------------------------------------
# Mode / compile helpers
# ---------------------------------------------------------------------------


def make_c_mode() -> Mode:
    """Build a PyTensor C-linker compilation mode.

    Uses the CVM (C Virtual Machine) linker which is more robust than
    the direct C linker while still leveraging C code generation.
    """
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


def rmsnorm_symbolic(
    x: pt.TensorVariable, gamma: pt.TensorVariable, eps: float
) -> pt.TensorVariable:
    """Learned RMSNorm matching MLX ``nn.RMSNorm`` semantics.

    .. math::

        y = \\frac{x}{\\sqrt{\\mathrm{mean}(x^2) + \\epsilon}} \\cdot \\gamma

    Weight **is** gamma (initialised to ones), *not* ``1 + weight``.

    Parameters
    ----------
    x : TensorVariable
        Input tensor of any rank >= 1.
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


def rms_norm_no_scale(x: pt.TensorVariable, eps: float) -> pt.TensorVariable:
    """RMS normalisation without learned scale (for V projections).

    Parameters
    ----------
    x : TensorVariable
        Input tensor.
    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    TensorVariable
        Same shape as *x*.
    """
    variance = pt.mean(pt.square(x), axis=-1, keepdims=True)
    return x * (np.float32(1.0) / pt.sqrt(variance + np.float32(eps)))


def gelu_approx_symbolic(x: pt.TensorVariable) -> pt.TensorVariable:
    """Approximate tanh GELU.

    .. math::

        0.5 \\, x \\, \\bigl(1 + \\tanh\\bigl(\\sqrt{2/\\pi}\\,(x + 0.044715\\,x^3)\\bigr)\\bigr)

    Parameters
    ----------
    x : TensorVariable

    Returns
    -------
    TensorVariable
        Same shape as *x*.
    """
    return np.float32(0.5) * x * (np.float32(1.0) + pt.tanh(
        np.float32(math.sqrt(2.0 / math.pi)) * (x + np.float32(0.044715) * x ** 3)
    ))


def sparse_gelu_symbolic(
    x: pt.TensorVariable, std_multiplier: float
) -> pt.TensorVariable:
    """Sparse GELU with exact mean / std / cutoff formula.

    .. math::

        \\text{cutoff} = \\mu + \\sigma \\cdot m, \\quad
        y = \\text{gelu\\_approx}(\\max(0,\\; x - \\text{cutoff}))

    where :math:`\\mu` and :math:`\\sigma` are the population mean and
    standard deviation over the last axis, and *m* is *std_multiplier*.

    Parameters
    ----------
    x : TensorVariable
        Input tensor.
    std_multiplier : float
        Pre-computed ``sqrt(2) * erfinv(2 * sparsity - 1)``.

    Returns
    -------
    TensorVariable
        Same shape as *x*.
    """
    inputs_mean = pt.mean(x, axis=-1, keepdims=True)
    diff = x - inputs_mean
    inputs_var = pt.mean(pt.square(diff), axis=-1, keepdims=True)
    inputs_std = pt.sqrt(inputs_var)
    cutoff_x = inputs_mean + inputs_std * np.float32(std_multiplier)
    return gelu_approx_symbolic(pt.maximum(np.float32(0.0), x - cutoff_x))


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


def build_rope_table(
    base: float, head_dim: int, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build NumPy cos/sin RoPE tables with half-split (``traditional=False``).

    Parameters
    ----------
    base : float
        RoPE base frequency.
    head_dim : int
        Head dimension (must be even).
    seq_len : int
        Number of positions.

    Returns
    -------
    cos, sin : np.ndarray
        Each ``(seq_len, head_dim // 2)``, dtype ``float32``.
    """
    half = head_dim // 2
    freqs = np.float32(1.0) / (
        np.float32(base)
        ** (np.arange(0, half, dtype=np.float32) * np.float32(2.0) / np.float32(head_dim))
    )
    positions = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope_symbolic(
    x: pt.TensorVariable,
    cos: pt.TensorVariable,
    sin: pt.TensorVariable,
    head_dim: int,
) -> pt.TensorVariable:
    """Apply half-split (``traditional=False``) RoPE.

    Parameters
    ----------
    x : TensorVariable
        Shape ``(..., head_dim)``.
    cos, sin : TensorVariable
        Broadcastable over leading dims, shape ``(..., head_dim // 2)``.
    head_dim : int
        Static head dimension.

    Returns
    -------
    TensorVariable
        Same shape as *x*.
    """
    half = head_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return pt.concatenate([rot1, rot2], axis=-1)


def causal_mask(seq_len: int) -> np.ndarray:
    """Build a float additive causal mask ``[T, T]``.

    Parameters
    ----------
    seq_len : int

    Returns
    -------
    np.ndarray
        Shape ``(seq_len, seq_len)``, dtype ``float32``.
        ``0`` for valid positions, ``-inf`` for masked.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")
    mask = np.full((seq_len, seq_len), -np.inf, dtype=np.float32)
    for i in range(seq_len):
        mask[i, : i + 1] = 0.0
    return mask


def sliding_window_mask(seq_len: int, window: int) -> np.ndarray:
    """Build a float additive sliding-window causal mask ``[T, T]``.

    Parameters
    ----------
    seq_len : int
    window : int
        Sliding window size.

    Returns
    -------
    np.ndarray
        Shape ``(seq_len, seq_len)``, dtype ``float32``.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")
    if window < 1:
        raise ValueError("window must be at least 1")
    mask = np.full((seq_len, seq_len), -np.inf, dtype=np.float32)
    for i in range(seq_len):
        start = max(0, i - window + 1)
        mask[i, start : i + 1] = 0.0
    return mask


def gqa_attention(
    q: pt.TensorVariable,
    k: pt.TensorVariable,
    v: pt.TensorVariable,
    mask: pt.TensorVariable,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    B: int,
    T: int,
    scale: float,
) -> pt.TensorVariable:
    """GQA attention with head-repeat expansion and additive mask.

    Parameters
    ----------
    q : TensorVariable
        Shape ``(B, n_heads, T, head_dim)``.
    k : TensorVariable
        Shape ``(B, n_kv_heads, T, head_dim)``.
    v : TensorVariable
        Shape ``(B, n_kv_heads, T, head_dim)``.
    mask : TensorVariable
        Additive float mask broadcastable to ``(B, n_heads, T, T)``.
    n_heads, n_kv_heads, head_dim : int
        Static head counts and dimension.
    B, T : int
        Static batch and sequence dimensions.
    scale : float
        Attention scale factor (``1.0`` for Gemma3n).

    Returns
    -------
    TensorVariable
        Shape ``(B, n_heads, T, head_dim)``.
    """
    repeats = n_heads // n_kv_heads
    if repeats > 1:
        # Use slice-and-concatenate expansion which is compatible with both
        # C and Numba linkers.  pt.repeat triggers a Blockwise op that the
        # C code generator does not implement.
        k_parts = []
        v_parts = []
        for i in range(n_kv_heads):
            k_head = k[:, i : i + 1, :, :]
            v_head = v[:, i : i + 1, :, :]
            k_parts.extend([k_head] * repeats)
            v_parts.extend([v_head] * repeats)
        k = pt.concatenate(k_parts, axis=1)
        v = pt.concatenate(v_parts, axis=1)

    # Element-wise attention scores (avoids Blockwise matmul which the C
    # linker does not implement for rank >= 3).
    # scores[b, h, i, j] = sum_d q[b, h, i, d] * k[b, h, j, d]
    q_exp = q[:, :, :, None, :]  # [B, n_h, T_q, 1, hd]
    k_exp = k[:, :, None, :, :]  # [B, n_h, 1, T_k, hd]
    scores = pt.sum(q_exp * k_exp, axis=-1) * np.float32(scale)
    scores = scores + mask

    # Numerically stable softmax
    scores_max = pt.max(scores, axis=-1, keepdims=True)
    scores_exp = pt.exp(scores - scores_max)
    scores_sum = pt.sum(scores_exp, axis=-1, keepdims=True)
    attn_weights = scores_exp / scores_sum

    # Element-wise output: attn_weights @ v
    # output[b, h, i, d] = sum_j attn_weights[b, h, i, j] * v[b, h, j, d]
    attn_w_exp = attn_weights[:, :, :, :, None]  # [B, n_h, T_q, T_k, 1]
    v_exp = v[:, :, None, :, :]  # [B, n_h, 1, T_k, hd]
    return pt.sum(attn_w_exp * v_exp, axis=3)


# ---------------------------------------------------------------------------
# Composite symbolic functions
# ---------------------------------------------------------------------------


def laurel_symbolic(
    x: pt.TensorVariable,
    W_left: pt.TensorVariable,
    W_right: pt.TensorVariable,
    norm_gamma: pt.TensorVariable,
    B: int,
    T: int,
    H: int,
    rank: int,
    eps: float,
) -> pt.TensorVariable:
    """Learned Augmented Residual Layer (LAuReL).

    ``x + RMSNorm(W_right @ W_left @ x)``

    Parameters
    ----------
    x : TensorVariable
        Shape ``(B, T, H)``.
    W_left : TensorVariable
        Shape ``(H, rank)``.
    W_right : TensorVariable
        Shape ``(rank, H)``.
    norm_gamma : TensorVariable
        Shape ``(H,)``.
    B, T, H, rank : int
        Static dimensions.
    eps : float
        RMSNorm epsilon.

    Returns
    -------
    TensorVariable
        Shape ``(B, T, H)``.
    """
    laurel_x = linear_proj(x, W_left, B, T, H, rank)
    laurel_x = linear_proj(laurel_x, W_right, B, T, rank, H)
    normed_laurel = rmsnorm_symbolic(laurel_x, norm_gamma, eps)
    return x + normed_laurel


def mlp_symbolic(
    x: pt.TensorVariable,
    gate_W: pt.TensorVariable,
    up_W: pt.TensorVariable,
    down_W: pt.TensorVariable,
    B: int,
    T: int,
    H: int,
    I: int,
    std_multiplier: Optional[float] = None,
) -> pt.TensorVariable:
    """GELU-gated MLP with optional activation sparsity.

    ``down(gelu(gate(x)) * up(x))`` or sparse variant.

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
    std_multiplier : float or None
        If not ``None`` and ``> 0``, use sparse GELU with this multiplier.

    Returns
    -------
    TensorVariable
        Shape ``(B, T, H)``.
    """
    gate_proj = linear_proj(x, gate_W, B, T, H, I)
    if std_multiplier is not None and std_multiplier > 0.0:
        activations = sparse_gelu_symbolic(gate_proj, std_multiplier)
    else:
        activations = gelu_approx_symbolic(gate_proj)
    up_proj = linear_proj(x, up_W, B, T, H, I)
    gated = activations * up_proj
    gated_flat = gated.reshape((B * T, I))
    out_flat = pt.dot(gated_flat, down_W)
    return out_flat.reshape((B, T, H))


def altup_router_modalities_symbolic(
    x: pt.TensorVariable,
    router_norm_gamma: pt.TensorVariable,
    router_W: pt.TensorVariable,
    B: int,
    T: int,
    H: int,
    n: int,
    eps: float,
) -> pt.TensorVariable:
    """Compute AltUp router modalities.

    ``tanh(Linear(RMSNorm(x) / H))``

    Parameters
    ----------
    x : TensorVariable
        Shape ``(B, T, H)``.
    router_norm_gamma : TensorVariable
        Shape ``(H,)``.
    router_W : TensorVariable
        Shape ``(H, n)``.
    B, T, H, n : int
        Static dimensions.
    eps : float
        RMSNorm epsilon.

    Returns
    -------
    TensorVariable
        Shape ``(B, T, n)``, float32.
    """
    router_inputs = rmsnorm_symbolic(x, router_norm_gamma, eps) * np.float32(
        1.0 / H
    )
    routed = linear_proj(router_inputs, router_W, B, T, H, n)
    return pt.tanh(routed)


def altup_predict_symbolic(
    x: pt.TensorVariable,
    prediction_coefs_W: pt.TensorVariable,
    router_norm_gamma: pt.TensorVariable,
    router_W: pt.TensorVariable,
    B: int,
    T: int,
    H: int,
    n: int,
    eps: float,
    coef_clip: Optional[float] = None,
    active_idx: int = 0,
) -> pt.TensorVariable:
    """AltUp predict step.

    Parameters
    ----------
    x : TensorVariable
        Shape ``(n, B, T, H)`` — stacked streams.
    prediction_coefs_W : TensorVariable
        Shape ``(n, n * n)`` — prediction coefficient weights.
    router_norm_gamma : TensorVariable
        Shape ``(H,)``.
    router_W : TensorVariable
        Shape ``(H, n)``.
    B, T, H, n : int
        Static dimensions.
    eps : float
        RMSNorm epsilon.
    coef_clip : float or None
        If not ``None``, clip coefficient weights to ``[-clip, clip]``.
    active_idx : int
        Index of the active stream for routing.

    Returns
    -------
    TensorVariable
        Shape ``(n, B, T, H)`` — predicted streams (FP32).
    """
    modalities = altup_router_modalities_symbolic(
        x[active_idx], router_norm_gamma, router_W, B, T, H, n, eps
    )
    # Clip weights
    W = prediction_coefs_W
    if coef_clip is not None:
        W = pt.clip(W, np.float32(-coef_clip), np.float32(coef_clip))

    # modalities: [B, T, n]  ->  linear -> [B, T, n^2]
    coefs_flat = linear_proj(modalities, W, B, T, n, n * n)
    all_coefs = coefs_flat.reshape((B, T, n, n))
    # Transpose last two dims (MLX: .transpose(0, 1, 3, 2))
    all_coefs = all_coefs.transpose(0, 1, 3, 2)

    # x_permuted: [n, B, T, H] -> [B, T, H, n]
    x_up = x.astype("float32")
    x_permuted = x_up.transpose(1, 2, 3, 0)

    # Element-wise batched matmul (avoids Blockwise which C linker rejects):
    # predictions[b, t, h, j] = sum_i x_permuted[b, t, h, i] * all_coefs[b, t, i, j]
    x_exp = x_permuted[:, :, :, :, None]  # [B, T, H, n, 1]
    c_exp = all_coefs[:, :, None, :, :]  # [B, T, 1, n, n]
    predictions = pt.sum(x_exp * c_exp, axis=3)  # [B, T, H, n]
    # Back to [n, B, T, H]
    predictions = predictions.transpose(3, 0, 1, 2)
    predictions = predictions + x_up
    return predictions


def altup_correct_symbolic(
    predictions: pt.TensorVariable,
    activated: pt.TensorVariable,
    correction_coefs_W: pt.TensorVariable,
    router_norm_gamma: pt.TensorVariable,
    router_W: pt.TensorVariable,
    B: int,
    T: int,
    H: int,
    n: int,
    eps: float,
    coef_clip: Optional[float] = None,
    active_idx: int = 0,
) -> pt.TensorVariable:
    """AltUp correct step.

    Parameters
    ----------
    predictions : TensorVariable
        Shape ``(n, B, T, H)`` — predicted streams from ``altup_predict``.
    activated : TensorVariable
        Shape ``(B, T, H)`` — the activated (post-MLP) stream.
    correction_coefs_W : TensorVariable
        Shape ``(n, n)`` — correction coefficient weights.
    router_norm_gamma : TensorVariable
        Shape ``(H,)``.
    router_W : TensorVariable
        Shape ``(H, n)``.
    B, T, H, n : int
        Static dimensions.
    eps : float
        RMSNorm epsilon.
    coef_clip : float or None
        If not ``None``, clip coefficient weights to ``[-clip, clip]``.
    active_idx : int
        Index of the active stream.

    Returns
    -------
    TensorVariable
        Shape ``(n, B, T, H)`` — corrected streams.
    """
    modalities = altup_router_modalities_symbolic(
        activated, router_norm_gamma, router_W, B, T, H, n, eps
    )

    W = correction_coefs_W
    if coef_clip is not None:
        W = pt.clip(W, np.float32(-coef_clip), np.float32(coef_clip))

    # modalities: [B, T, n] -> linear -> [B, T, n]
    coefs = linear_proj(modalities, W, B, T, n, n)
    all_coefs = coefs + np.float32(1.0)

    active_x = predictions[active_idx]  # [B, T, H]
    innovation = activated - active_x  # [B, T, H]

    # all_coefs: [B, T, n] -> [n, B, T]
    all_coefs_t = all_coefs.transpose(2, 0, 1)
    # innovation[None]: [1, B, T, H]
    # all_coefs_t[..., None]: [n, B, T, 1]
    corrected = innovation[None] * all_coefs_t[..., None]
    corrected = corrected + predictions
    return corrected


def attention_block_symbolic(
    x: pt.TensorVariable,
    q_w: pt.TensorVariable,
    k_w: pt.TensorVariable,
    v_w: pt.TensorVariable,
    o_w: pt.TensorVariable,
    q_norm_gamma: pt.TensorVariable,
    k_norm_gamma: pt.TensorVariable,
    cos: pt.TensorVariable,
    sin: pt.TensorVariable,
    mask: pt.TensorVariable,
    B: int,
    T: int,
    H: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    eps: float,
    shared_k: Optional[pt.TensorVariable] = None,
    shared_v: Optional[pt.TensorVariable] = None,
    return_kv: bool = False,
) -> "pt.TensorVariable | tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]":
    """Full attention block with Q/K RMSNorm, V RMSNoScale, RoPE, GQA.

    Parameters
    ----------
    x : TensorVariable
        Shape ``(B, T, H)``.
    q_w : TensorVariable
        Shape ``(H, n_heads * head_dim)``.
    k_w, v_w : TensorVariable
        Shape ``(H, n_kv_heads * head_dim)``.
    o_w : TensorVariable
        Shape ``(n_heads * head_dim, H)``.
    q_norm_gamma, k_norm_gamma : TensorVariable
        Shape ``(head_dim,)``.
    cos, sin : TensorVariable
        Shape ``(T, head_dim // 2)``.
    mask : TensorVariable
        Additive mask broadcastable to ``(B, n_heads, T, T)``.
    B, T, H, n_heads, n_kv_heads, head_dim : int
        Static dimensions.
    eps : float
        RMSNorm epsilon.
    shared_k, shared_v : TensorVariable or None
        Precomputed RoPE-rotated K and normalized V from a concrete cache
        source layer. Both must be provided together.
    return_kv : bool
        Return the concrete K/V tensors alongside the attention output.

    Returns
    -------
    TensorVariable
        Shape ``(B, T, H)``.
    """
    if (shared_k is not None) != (shared_v is not None):
        raise ValueError("shared_k and shared_v must be provided together")

    # Q projection + reshape + RMSNorm + RoPE (always computed)
    q_proj = linear_proj(x, q_w, B, T, H, n_heads * head_dim)
    q = q_proj.reshape((B, T, n_heads, head_dim)).swapaxes(1, 2)
    q = rmsnorm_symbolic(q, q_norm_gamma, eps)

    cos_exp = cos.reshape((1, 1, T, head_dim // 2))
    sin_exp = sin.reshape((1, 1, T, head_dim // 2))
    q = apply_rope_symbolic(q, cos_exp, sin_exp, head_dim)

    # K/V: either project from x or use shared inputs
    if shared_k is None:
        k_proj = linear_proj(x, k_w, B, T, H, n_kv_heads * head_dim)
        v_proj = linear_proj(x, v_w, B, T, H, n_kv_heads * head_dim)
        k = k_proj.reshape((B, T, n_kv_heads, head_dim)).swapaxes(1, 2)
        v = v_proj.reshape((B, T, n_kv_heads, head_dim)).swapaxes(1, 2)
        k = rmsnorm_symbolic(k, k_norm_gamma, eps)
        v = rms_norm_no_scale(v, eps)
        k = apply_rope_symbolic(k, cos_exp, sin_exp, head_dim)
    else:
        k = shared_k
        v = shared_v

    # GQA attention (scale=1.0 for Gemma3n)
    attn_out = gqa_attention(
        q, k, v, mask, n_heads, n_kv_heads, head_dim, B, T, scale=1.0,
    )

    # Output projection: [B, n_heads, T, hd] -> [B, T, n_heads*hd] -> [B, T, H]
    attn_flat = attn_out.swapaxes(1, 2).reshape((B * T, n_heads * head_dim))
    o_out = pt.dot(attn_flat, o_w).reshape((B, T, H))

    if return_kv:
        return o_out, k, v
    return o_out


def decoder_layer_symbolic(
    x: pt.TensorVariable,
    mask: pt.TensorVariable,
    per_layer_input: pt.TensorVariable,
    cos: pt.TensorVariable,
    sin: pt.TensorVariable,
    # Attention weights
    q_w: pt.TensorVariable,
    k_w: pt.TensorVariable,
    v_w: pt.TensorVariable,
    o_w: pt.TensorVariable,
    q_norm_gamma: pt.TensorVariable,
    k_norm_gamma: pt.TensorVariable,
    # MLP weights
    gate_w: pt.TensorVariable,
    up_w: pt.TensorVariable,
    down_w: pt.TensorVariable,
    # LAuReL weights
    laurel_left_w: pt.TensorVariable,
    laurel_right_w: pt.TensorVariable,
    laurel_norm_gamma: pt.TensorVariable,
    # AltUp weights
    prediction_coefs_w: pt.TensorVariable,
    correction_coefs_w: pt.TensorVariable,
    modality_router_w: pt.TensorVariable,
    router_norm_gamma: pt.TensorVariable,
    correct_output_scale: pt.TensorVariable,
    # Layer norms
    input_ln_gamma: pt.TensorVariable,
    post_attn_ln_gamma: pt.TensorVariable,
    pre_ffw_ln_gamma: pt.TensorVariable,
    post_ffw_ln_gamma: pt.TensorVariable,
    post_pli_ln_gamma: pt.TensorVariable,
    # Per-layer gate/projection
    pli_gate_w: pt.TensorVariable,
    pli_proj_w: pt.TensorVariable,
    # Config scalars
    config: Gemma3nConfig,
    B: int,
    T: int,
    std_multiplier: Optional[float] = None,
    shared_k: Optional[pt.TensorVariable] = None,
    shared_v: Optional[pt.TensorVariable] = None,
    return_kv: bool = False,
) -> "pt.TensorVariable | tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]":
    """Complete Gemma3n decoder layer with four streams.

    Implements the exact operation order from ``Gemma3nDecoderLayer.__call__``
    including AltUp predict -> LAuReL + attention -> MLP -> AltUp correct ->
    per-layer gate injection (streams 1: only).

    Parameters
    ----------
    x : TensorVariable
        Shape ``(n, B, T, H)`` — stacked streams (``n = altup_num_inputs``).
    mask : TensorVariable
        Additive attention mask broadcastable to ``(B, n_heads, T, T)``.
    per_layer_input : TensorVariable
        Shape ``(B, T, H_per_layer)``.
    cos, sin : TensorVariable
        Shape ``(T, head_dim // 2)``.
    (many weight TensorVariables) :
        See source for shapes.
    config : Gemma3nConfig
    B, T : int
        Static batch and sequence dimensions.
    std_multiplier : float or None
        Sparse GELU multiplier (``None`` or ``0`` -> dense GELU).
    shared_k, shared_v : TensorVariable or None
        Precomputed RoPE-rotated K and normalized V from a concrete cache
        source layer. Both must be provided together.
    return_kv : bool
        Return the concrete K/V tensors alongside the layer output.

    Returns
    -------
    TensorVariable
        Shape ``(n, B, T, H)`` — corrected predictions with per-layer
        injection applied to streams ``1:``.
    """
    n = config.altup_num_inputs
    H = config.hidden_size
    eps = config.rms_norm_eps
    n_h = config.num_attention_heads
    n_kv = config.num_key_value_heads
    hd = config.head_dim
    I = config.intermediate_size
    rank = config.laurel_rank
    H_pl = config.hidden_size_per_layer_input
    active_idx = config.altup_active_idx

    # 1. AltUp predict
    predictions = altup_predict_symbolic(
        x, prediction_coefs_w, router_norm_gamma, modality_router_w,
        B, T, H, n, eps, config.altup_coef_clip, active_idx,
    )
    active_prediction = predictions[active_idx]  # [B, T, H]

    # 2. Input layernorm
    active_prediction_normed = rmsnorm_symbolic(
        active_prediction, input_ln_gamma, eps
    )

    # 3. LAuReL
    laurel_output = laurel_symbolic(
        active_prediction_normed, laurel_left_w, laurel_right_w,
        laurel_norm_gamma, B, T, H, rank, eps,
    )

    # 4. Attention (with optional shared K/V and return_kv)
    attention_result = attention_block_symbolic(
        active_prediction_normed, q_w, k_w, v_w, o_w,
        q_norm_gamma, k_norm_gamma, cos, sin, mask,
        B, T, H, n_h, n_kv, hd, eps,
        shared_k=shared_k, shared_v=shared_v, return_kv=return_kv,
    )

    if return_kv:
        attn, concrete_k, concrete_v = attention_result
    else:
        attn = attention_result

    # 5. Post-attention layernorm
    attn = rmsnorm_symbolic(attn, post_attn_ln_gamma, eps)

    # 6. Gate + LAuReL combine
    attn_gated = active_prediction + attn
    attn_laurel = (attn_gated + laurel_output) * np.float32(2.0 ** -0.5)

    # 7. Pre-feedforward layernorm -> MLP -> post-feedforward layernorm
    attn_norm = rmsnorm_symbolic(attn_laurel, pre_ffw_ln_gamma, eps)
    attn_ffw = mlp_symbolic(
        attn_norm, gate_w, up_w, down_w, B, T, H, I, std_multiplier,
    )
    attn_ffw_norm = rmsnorm_symbolic(attn_ffw, post_ffw_ln_gamma, eps)

    # 8. Residual
    attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm

    # 9. AltUp correct
    corrected = altup_correct_symbolic(
        predictions, attn_ffw_laurel_gated, correction_coefs_w,
        router_norm_gamma, modality_router_w,
        B, T, H, n, eps, config.altup_coef_clip, active_idx,
    )

    # 10. Correct output scale
    first_prediction = corrected[active_idx]  # [B, T, H]
    if config.altup_correct_scale:
        first_prediction = first_prediction * correct_output_scale

    # 11. Per-layer gate: Linear -> GELU -> multiply per_layer_input
    first_prediction = linear_proj(
        first_prediction, pli_gate_w, B, T, H, H_pl,
    )
    first_prediction = gelu_approx_symbolic(first_prediction)
    first_prediction = first_prediction * per_layer_input

    # 12. Per-layer projection -> norm
    first_prediction = linear_proj(
        first_prediction, pli_proj_w, B, T, H_pl, H,
    )
    first_prediction = rmsnorm_symbolic(
        first_prediction, post_pli_ln_gamma, eps,
    )

    # 13. Gate injection to streams 1: only
    stream0 = corrected[0:1]  # [1, B, T, H]
    streams_rest = corrected[1:]  # [n-1, B, T, H]
    streams_rest_updated = streams_rest + first_prediction[None]
    corrected_updated = pt.concatenate(
        [stream0, streams_rest_updated], axis=0
    )

    if return_kv:
        return corrected_updated, concrete_k, concrete_v
    return corrected_updated


def initial_stream_projections(
    h0: pt.TensorVariable,
    altup_proj_weights: List[pt.TensorVariable],
    B: int,
    T: int,
    H: int,
    n: int,
) -> pt.TensorVariable:
    """Initial four-stream projections + magnitude matching.

    Parameters
    ----------
    h0 : TensorVariable
        Shape ``(B, T, H)`` — embedded hidden states (already scaled by
        ``sqrt(H)``).
    altup_proj_weights : list of TensorVariable
        ``n - 1`` weight matrices, each ``(H, H)``.
    B, T, H, n : int
        Static dimensions.

    Returns
    -------
    TensorVariable
        Shape ``(n, B, T, H)``.
    """
    target_magnitude = pt.sqrt(
        pt.mean(pt.square(h0), axis=-1, keepdims=True)
    )  # [B, T, 1]

    h_list = [h0]
    for w in altup_proj_weights:
        h_list.append(linear_proj(h0, w, B, T, H, H))
    h = pt.stack(h_list, axis=0)  # [n, B, T, H]

    # Magnitude matching for streams 1:
    h_others = h[1:]  # [n-1, B, T, H]
    mags = pt.sqrt(
        pt.mean(pt.square(h_others), axis=-1, keepdims=True)
    )  # [n-1, B, T, 1]
    h_others_scaled = h_others * (
        target_magnitude / pt.maximum(mags, _FINFO_MIN)
    )
    return pt.concatenate([h[0:1], h_others_scaled], axis=0)


def per_layer_input_projection(
    inputs_embeds: pt.TensorVariable,
    per_layer_model_w: pt.TensorVariable,
    per_layer_proj_norm_gamma: pt.TensorVariable,
    per_layer_embeds: pt.TensorVariable,
    B: int,
    T: int,
    H: int,
    num_layers: int,
    H_pl: int,
    eps: float,
) -> pt.TensorVariable:
    """Per-layer model projection + norm combination.

    Computes::

        proj = RMSNorm(reshape(Linear(embeds) * H^{-0.5}))
        result = (proj + per_layer_embeds) * 2^{-0.5}

    Parameters
    ----------
    inputs_embeds : TensorVariable
        Shape ``(B, T, H)``.
    per_layer_model_w : TensorVariable
        Shape ``(H, num_layers * H_pl)``.
    per_layer_proj_norm_gamma : TensorVariable
        Shape ``(H_pl,)``.
    per_layer_embeds : TensorVariable
        Shape ``(B, T, num_layers, H_pl)``.
    B, T, H, num_layers, H_pl : int
        Static dimensions.
    eps : float
        RMSNorm epsilon.

    Returns
    -------
    TensorVariable
        Shape ``(B, T, num_layers, H_pl)``.
    """
    proj = linear_proj(
        inputs_embeds, per_layer_model_w, B, T, H, num_layers * H_pl,
    )
    proj = proj * np.float32(H ** -0.5)
    proj = proj.reshape((B, T, num_layers, H_pl))
    proj = rmsnorm_symbolic(proj, per_layer_proj_norm_gamma, eps)
    return (proj + per_layer_embeds) * np.float32(2.0 ** -0.5)


def final_unembed(
    h: pt.TensorVariable,
    unembed_proj_weights: List[pt.TensorVariable],
    final_norm_gamma: pt.TensorVariable,
    B: int,
    T: int,
    H: int,
    n: int,
    eps: float,
) -> pt.TensorVariable:
    """Final unembed projections + magnitude matching + stream mean + final norm.

    Parameters
    ----------
    h : TensorVariable
        Shape ``(n, B, T, H)``.
    unembed_proj_weights : list of TensorVariable
        ``n - 1`` weight matrices, each ``(H, H)``.
    final_norm_gamma : TensorVariable
        Shape ``(H,)``.
    B, T, H, n : int
        Static dimensions.
    eps : float
        RMSNorm epsilon.

    Returns
    -------
    TensorVariable
        Shape ``(B, T, H)`` — final normalised hidden states.
    """
    target_magnitude = pt.sqrt(
        pt.mean(pt.square(h[0]), axis=-1, keepdims=True)
    )  # [B, T, 1]

    h_others = []
    for i, w in enumerate(unembed_proj_weights):
        h_others.append(linear_proj(h[i + 1], w, B, T, H, H))
    if h_others:
        h_others_stacked = pt.stack(h_others, axis=0)  # [n-1, B, T, H]
        mags = pt.sqrt(
            pt.mean(pt.square(h_others_stacked), axis=-1, keepdims=True)
        )  # [n-1, B, T, 1]
        h_others_scaled = h_others_stacked * (
            target_magnitude / pt.maximum(mags, _FINFO_MIN)
        )
        h_combined = pt.concatenate([h[0:1], h_others_scaled], axis=0)
    else:
        h_combined = h[0:1]
    h_mean = pt.mean(h_combined, axis=0)  # [B, T, H]
    return rmsnorm_symbolic(h_mean, final_norm_gamma, eps)


def chunked_logit_projection(
    hidden: pt.TensorVariable,
    emb_table: pt.TensorVariable,
    vocab_size: int,
    chunk_size: int,
    softcap: Optional[float] = None,
) -> pt.TensorVariable:
    """Chunked tied logit projection with optional soft-capping.

    Splits the vocabulary into chunks of *chunk_size*, computes logits
    per chunk, applies ``softcap * tanh(logits / softcap)`` when
    *softcap* is not ``None``, and concatenates.

    Parameters
    ----------
    hidden : TensorVariable
        Shape ``(B, H)``.
    emb_table : TensorVariable
        Shape ``(vocab_size, H)``.
    vocab_size : int
    chunk_size : int
    softcap : float or None

    Returns
    -------
    TensorVariable
        Shape ``(B, vocab_size)``.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks = []
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk_emb = emb_table[start:end]  # [chunk, H]
        chunk_logits = pt.dot(hidden, chunk_emb.T)  # [B, chunk]
        if softcap is not None:
            chunk_logits = np.float32(softcap) * pt.tanh(
                chunk_logits / np.float32(softcap)
            )
        chunks.append(chunk_logits)
    return pt.concatenate(chunks, axis=-1)


# ---------------------------------------------------------------------------
# Compile helpers
# ---------------------------------------------------------------------------


def compile_decoder_layer(
    config: Gemma3nConfig,
    batch_size: int,
    seq_len: int,
    has_sparsity: bool = False,
    backend: str = "c",
    kv_mode: str = "none",
):
    """Compile a Gemma3n decoder layer for fixed shapes.

    Parameters
    ----------
    config : Gemma3nConfig
    batch_size, seq_len : int
        Static dimensions (both >= 1).
    has_sparsity : bool
        If ``True``, use sparse GELU in the MLP.
    backend : str
        ``'c'``, ``'numba'``, or ``'FAST_COMPILE'``.
    kv_mode : str
        ``'none'`` computes K/V internally, ``'return'`` also returns concrete
        K/V, and ``'shared'`` accepts K/V from a lower source layer.

    Returns
    -------
    callable
        Compiled function accepting 29 positional NumPy arrays and
        returning corrected predictions ``(n, B, T, H)``.

    Raises
    ------
    ValueError
        If ``batch_size < 1`` or ``seq_len < 1``.
    """
    if batch_size < 1:
        raise ValueError(
            f"compile_decoder_layer requires batch_size >= 1, got {batch_size}"
        )
    if seq_len < 1:
        raise ValueError(
            f"compile_decoder_layer requires seq_len >= 1, got {seq_len}"
        )
    if kv_mode not in {"none", "return", "shared"}:
        raise ValueError(
            f"kv_mode must be one of 'none', 'return', or 'shared', got {kv_mode!r}"
        )

    B, T = batch_size, seq_len
    H = config.hidden_size
    n = config.altup_num_inputs
    n_h = config.num_attention_heads
    n_kv = config.num_key_value_heads
    hd = config.head_dim
    I = config.intermediate_size
    rank = config.laurel_rank
    H_pl = config.hidden_size_per_layer_input
    eps = config.rms_norm_eps

    if has_sparsity and not 0.0 < config.activation_sparsity < 1.0:
        raise ValueError(
            "activation_sparsity must be in (0, 1) for a sparse layer"
        )

    std_mult = None
    if has_sparsity and config.activation_sparsity > 0.0:
        std_mult = float(
            math.sqrt(2.0) * _erfinv(2.0 * config.activation_sparsity - 1.0)
        )

    # -- Symbolic inputs --
    hidden = pt.tensor("hidden", shape=(n, B, T, H), dtype="float32")
    mask_in = pt.tensor("mask", shape=(T, T), dtype="float32")
    pli = pt.tensor("per_layer_input", shape=(B, T, H_pl), dtype="float32")
    cos_in = pt.tensor("cos", shape=(T, hd // 2), dtype="float32")
    sin_in = pt.tensor("sin", shape=(T, hd // 2), dtype="float32")

    # Shared K/V inputs (only for kv_mode='shared')
    shared_k = None
    shared_v = None
    if kv_mode == "shared":
        shared_k = pt.tensor("shared_k", shape=(B, n_kv, T, hd), dtype="float32")
        shared_v = pt.tensor("shared_v", shape=(B, n_kv, T, hd), dtype="float32")

    q_w = pt.tensor("q_w", shape=(H, n_h * hd), dtype="float32")
    k_w = pt.tensor("k_w", shape=(H, n_kv * hd), dtype="float32")
    v_w = pt.tensor("v_w", shape=(H, n_kv * hd), dtype="float32")
    o_w = pt.tensor("o_w", shape=(n_h * hd, H), dtype="float32")
    q_ng = pt.tensor("q_norm_gamma", shape=(hd,), dtype="float32")
    k_ng = pt.tensor("k_norm_gamma", shape=(hd,), dtype="float32")

    gate_w = pt.tensor("gate_w", shape=(H, I), dtype="float32")
    up_w = pt.tensor("up_w", shape=(H, I), dtype="float32")
    down_w = pt.tensor("down_w", shape=(I, H), dtype="float32")

    ll_w = pt.tensor("laurel_left_w", shape=(H, rank), dtype="float32")
    lr_w = pt.tensor("laurel_right_w", shape=(rank, H), dtype="float32")
    ln_g = pt.tensor("laurel_norm_gamma", shape=(H,), dtype="float32")

    pred_w = pt.tensor("pred_coefs_w", shape=(n, n * n), dtype="float32")
    corr_w = pt.tensor("corr_coefs_w", shape=(n, n), dtype="float32")
    mr_w = pt.tensor("modality_router_w", shape=(H, n), dtype="float32")
    rn_g = pt.tensor("router_norm_gamma", shape=(H,), dtype="float32")
    cos_scale = pt.tensor("correct_output_scale", shape=(H,), dtype="float32")

    iln_g = pt.tensor("input_ln_gamma", shape=(H,), dtype="float32")
    paln_g = pt.tensor("post_attn_ln_gamma", shape=(H,), dtype="float32")
    pfln_g = pt.tensor("pre_ffw_ln_gamma", shape=(H,), dtype="float32")
    pfln2_g = pt.tensor("post_ffw_ln_gamma", shape=(H,), dtype="float32")
    plin_g = pt.tensor("post_pli_ln_gamma", shape=(H,), dtype="float32")

    pli_gw = pt.tensor("pli_gate_w", shape=(H, H_pl), dtype="float32")
    pli_pw = pt.tensor("pli_proj_w", shape=(H_pl, H), dtype="float32")

    # Expand mask to [1, 1, T, T] for broadcasting
    mask_4d = mask_in.reshape((1, 1, T, T))

    out = decoder_layer_symbolic(
        hidden, mask_4d, pli, cos_in, sin_in,
        q_w, k_w, v_w, o_w, q_ng, k_ng,
        gate_w, up_w, down_w,
        ll_w, lr_w, ln_g,
        pred_w, corr_w, mr_w, rn_g, cos_scale,
        iln_g, paln_g, pfln_g, pfln2_g, plin_g,
        pli_gw, pli_pw,
        config, B, T, std_mult,
        shared_k=shared_k, shared_v=shared_v, return_kv=(kv_mode == "return"),
    )

    inputs = [hidden, mask_in, pli, cos_in, sin_in]
    if kv_mode == "shared":
        inputs.extend([shared_k, shared_v])
    inputs.extend([
        q_w, k_w, v_w, o_w, q_ng, k_ng,
        gate_w, up_w, down_w,
        ll_w, lr_w, ln_g,
        pred_w, corr_w, mr_w, rn_g, cos_scale,
        iln_g, paln_g, pfln_g, pfln2_g, plin_g,
        pli_gw, pli_pw,
    ])
    return pytensor.function(
        inputs, out, mode=_get_mode(backend),
        on_unused_input="ignore" if kv_mode == "shared" else "raise",
    )


def compile_initial_projections(
    config: Gemma3nConfig,
    batch_size: int,
    seq_len: int,
    backend: str = "c",
):
    """Compile initial four-stream projections + magnitude matching.

    Parameters
    ----------
    config : Gemma3nConfig
    batch_size, seq_len : int
    backend : str

    Returns
    -------
    callable
        ``fn(h0, *altup_proj_weights) -> h_stacked``

        - h0: ``(B, T, H)``
        - altup_proj_weights: ``n - 1`` arrays of ``(H, H)``
        - h_stacked: ``(n, B, T, H)``
    """
    if batch_size < 1 or seq_len < 1:
        raise ValueError("batch_size and seq_len must be >= 1")

    B, T = batch_size, seq_len
    H = config.hidden_size
    n = config.altup_num_inputs

    h0 = pt.tensor("h0", shape=(B, T, H), dtype="float32")
    proj_ws = [
        pt.tensor(f"altup_proj_w_{i}", shape=(H, H), dtype="float32")
        for i in range(n - 1)
    ]

    out = initial_stream_projections(h0, proj_ws, B, T, H, n)
    return pytensor.function([h0] + proj_ws, out, mode=_get_mode(backend))


def compile_per_layer_projection(
    config: Gemma3nConfig,
    batch_size: int,
    seq_len: int,
    backend: str = "c",
):
    """Compile per-layer model projection + norm combination.

    Parameters
    ----------
    config : Gemma3nConfig
    batch_size, seq_len : int
    backend : str

    Returns
    -------
    callable
        ``fn(embeds, proj_w, norm_gamma, per_layer_embeds) -> result``

        - embeds: ``(B, T, H)``
        - proj_w: ``(H, num_layers * H_pl)``
        - norm_gamma: ``(H_pl,)``
        - per_layer_embeds: ``(B, T, num_layers, H_pl)``
        - result: ``(B, T, num_layers, H_pl)``
    """
    if batch_size < 1 or seq_len < 1:
        raise ValueError("batch_size and seq_len must be >= 1")

    B, T = batch_size, seq_len
    H = config.hidden_size
    L = config.num_hidden_layers
    H_pl = config.hidden_size_per_layer_input
    eps = config.rms_norm_eps

    embeds = pt.tensor("embeds", shape=(B, T, H), dtype="float32")
    proj_w = pt.tensor("proj_w", shape=(H, L * H_pl), dtype="float32")
    norm_g = pt.tensor("norm_gamma", shape=(H_pl,), dtype="float32")
    ple = pt.tensor("per_layer_embeds", shape=(B, T, L, H_pl), dtype="float32")

    out = per_layer_input_projection(
        embeds, proj_w, norm_g, ple, B, T, H, L, H_pl, eps,
    )
    return pytensor.function([embeds, proj_w, norm_g, ple], out, mode=_get_mode(backend))


def compile_final_unembed(
    config: Gemma3nConfig,
    batch_size: int,
    seq_len: int,
    backend: str = "c",
):
    """Compile final unembed projections + magnitude matching + mean + norm.

    Parameters
    ----------
    config : Gemma3nConfig
    batch_size, seq_len : int
    backend : str

    Returns
    -------
    callable
        ``fn(h, *unembed_proj_ws, final_norm_gamma) -> h_normed``

        - h: ``(n, B, T, H)``
        - unembed_proj_ws: ``n - 1`` arrays of ``(H, H)``
        - final_norm_gamma: ``(H,)``
        - h_normed: ``(B, T, H)``
    """
    if batch_size < 1 or seq_len < 1:
        raise ValueError("batch_size and seq_len must be >= 1")

    B, T = batch_size, seq_len
    H = config.hidden_size
    n = config.altup_num_inputs
    eps = config.rms_norm_eps

    h = pt.tensor("h", shape=(n, B, T, H), dtype="float32")
    unembed_ws = [
        pt.tensor(f"unembed_w_{i}", shape=(H, H), dtype="float32")
        for i in range(n - 1)
    ]
    fn_g = pt.tensor("final_norm_gamma", shape=(H,), dtype="float32")

    out = final_unembed(h, unembed_ws, fn_g, B, T, H, n, eps)
    return pytensor.function(
        [h] + unembed_ws + [fn_g], out, mode=_get_mode(backend)
    )


def compile_logit_projection(
    vocab_size: int,
    hidden_size: int,
    batch_size: int,
    chunk_size: int,
    softcap: Optional[float] = None,
    backend: str = "c",
):
    """Compile chunked tied logit projection with optional softcap.

    Parameters
    ----------
    vocab_size, hidden_size, batch_size, chunk_size : int
    softcap : float or None
    backend : str

    Returns
    -------
    callable
        ``fn(hidden, emb_table) -> logits``

        - hidden: ``(B, H)``
        - emb_table: ``(V, H)``
        - logits: ``(B, V)``
    """
    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1")
    if hidden_size < 1:
        raise ValueError("hidden_size must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    B = batch_size
    H = hidden_size
    V = vocab_size

    hidden = pt.tensor("hidden", shape=(B, H), dtype="float32")
    emb_table = pt.tensor("emb_table", shape=(V, H), dtype="float32")

    out = chunked_logit_projection(hidden, emb_table, V, chunk_size, softcap)
    return pytensor.function(
        [hidden, emb_table], out, mode=_get_mode(backend)
    )


def compile_per_chunk_logits(
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    softcap: Optional[float] = None,
    backend: str = "c",
):
    """Compile per-chunk logit projection for streaming inference.

    Accepts hidden states ``[B, T, H]`` and one embedding chunk
    ``[chunk, H]``, returns logits ``[B, T, chunk]``.  The runner never
    passes a full embedding table.

    Parameters
    ----------
    hidden_size : int
    batch_size : int
    seq_len : int
    chunk_size : int
    softcap : float or None
        If not ``None``, apply ``softcap * tanh(logits / softcap)``.
    backend : str

    Returns
    -------
    callable
        ``fn(hidden, chunk_emb) -> logits``

        - hidden: ``(B, T, H)``
        - chunk_emb: ``(chunk_size, H)``
        - logits: ``(B, T, chunk_size)``
    """
    if batch_size < 1 or seq_len < 1 or chunk_size < 1:
        raise ValueError("batch_size, seq_len, and chunk_size must be >= 1")

    B, T, H = batch_size, seq_len, hidden_size
    C = chunk_size

    hidden = pt.tensor("hidden", shape=(B, T, H), dtype="float32")
    chunk_emb = pt.tensor("chunk_emb", shape=(C, H), dtype="float32")

    # Flatten to [B*T, H], project, reshape back
    hidden_flat = hidden.reshape((B * T, H))
    logits_flat = pt.dot(hidden_flat, chunk_emb.T)  # [B*T, C]
    logits = logits_flat.reshape((B, T, C))

    if softcap is not None:
        logits = np.float32(softcap) * pt.tanh(logits / np.float32(softcap))

    return pytensor.function(
        [hidden, chunk_emb], logits, mode=_get_mode(backend)
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _erfinv(x: float) -> float:
    """Compute inverse error function using scipy if available, else fallback.

    Parameters
    ----------
    x : float
        Input in ``(-1, 1)``.

    Returns
    -------
    float
    """
    try:
        from scipy.special import erfinv

        return float(erfinv(x))
    except ImportError:
        # Rational approximation (Winitzki 2008)
        a = 0.147
        ln1mx2 = math.log(1.0 - x * x)
        t1 = 2.0 / (math.pi * a) + ln1mx2 / 2.0
        t2 = ln1mx2 / a
        sign = 1.0 if x >= 0 else -1.0
        return sign * math.sqrt(math.sqrt(t1 * t1 - t2) - t1)
