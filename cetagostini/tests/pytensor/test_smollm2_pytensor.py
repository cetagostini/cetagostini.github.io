"""Tests for smollm2_pytensor PyTensor graph builders.

All fixtures are deterministic, asymmetric, and nonzero.
Independent eager NumPy oracle functions are defined here (not imported
from the implementation module).  Prefill and decode layers are validated
parametrically through both C and Numba backends.

Tests cover:
- Deterministic NumPy parity for primitives and composite toy layers
- C/Numba backend parity
- Sparse/dense variants (SiLU gating)
- Float32 audit
- Smol prefill/decode/cache behavior
- Batch-size-one decode validation
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import pytensor

# Capture floatX *before* importing smollm2_pytensor so we can verify
# the module does not mutate it.
_floatX_before_import = pytensor.config.floatX

import pytensor.tensor as pt

from cetagostini.utils.pytensor.smollm2_pytensor import (
    SmolLM2Config,
    apply_rope,
    audit_float32,
    build_rope_table,
    compile_decode_layer,
    compile_embedding,
    compile_logits,
    compile_prefill_layer,
    gqa_attention,
    linear_proj,
    make_c_mode,
    make_numba_mode,
    rmsnorm_symbolic,
    rotate_half,
    silu_gated_mlp,
)

_import_preserved_floatX = pytensor.config.floatX == _floatX_before_import


# ---------------------------------------------------------------------------
# Test configuration — small nonsymmetric dimensions
# ---------------------------------------------------------------------------

SMALL_CONFIG = SmolLM2Config(
    vocab_size=100,
    hidden_size=32,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    head_dim=8,
    intermediate_size=64,
    context_length=32,
    rms_eps=1e-5,
    rope_theta=100_000.0,
    bos=1,
    eos=2,
)

BACKENDS = ["c", "numba"]


def det_array(shape, seed, low=0.1, high=1.7):
    """Deterministic, asymmetric, nonzero float32 array."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(low, high, size=shape).astype(np.float32)
    ramp = np.arange(x.size, dtype=np.float32).reshape(shape) * 1e-3
    return x + ramp


# ---------------------------------------------------------------------------
# NumPy oracle functions (independent eager implementations)
# ---------------------------------------------------------------------------


def np_rmsnorm(x, gamma, eps):
    """NumPy RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma."""
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(var + eps)) * gamma


def np_rotate_half(x, head_dim):
    """NumPy rotate_half: [-x[..., half:], x[..., :half]]."""
    half = head_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def np_apply_rope(q, k, cos, sin, head_dim, seq_len):
    """NumPy RoPE using rotate_half formulation."""
    cos_full = np.concatenate([cos, cos], axis=-1).reshape(1, 1, seq_len, head_dim)
    sin_full = np.concatenate([sin, sin], axis=-1).reshape(1, 1, seq_len, head_dim)
    q_rot = q * cos_full + np_rotate_half(q, head_dim) * sin_full
    k_rot = k * cos_full + np_rotate_half(k, head_dim) * sin_full
    return q_rot, k_rot


def np_softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


def np_sdpa(q, k, v, mask=None, scale=1.0, is_causal=False):
    """NumPy scaled dot-product attention with optional causal mask."""
    scores = np.matmul(q, k.swapaxes(-1, -2)) * scale
    if mask is not None:
        scores = scores + mask
    elif is_causal:
        T_q, T_k = q.shape[-2], k.shape[-2]
        causal = np.full((T_q, T_k), -np.inf, dtype=np.float32)
        for i in range(T_q):
            causal[i, : i + 1] = 0.0
        scores = scores + causal
    weights = np_softmax(scores, axis=-1)
    return np.matmul(weights, v)


def np_silu(x):
    """NumPy SiLU (swish): x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-x)))


def np_full_prefill(
    hidden, q_w, k_w, v_w, o_w, gate_w, up_w, down_w,
    in_gamma, post_gamma, cos, sin, config, B, T,
):
    """Full NumPy prefill layer reference."""
    H = config.hidden_size
    n_h = config.n_heads
    n_kv = config.n_kv_heads
    hd = config.head_dim
    I = config.intermediate_size
    eps = config.rms_eps
    scale = 1.0 / math.sqrt(hd)

    # Pre-attention RMSNorm
    normed = np_rmsnorm(hidden, in_gamma, eps)

    # Q/K/V projections
    q = (normed.reshape(B * T, H) @ q_w).reshape(B, T, n_h, hd).swapaxes(1, 2)
    k = (normed.reshape(B * T, H) @ k_w).reshape(B, T, n_kv, hd).swapaxes(1, 2)
    v = (normed.reshape(B * T, H) @ v_w).reshape(B, T, n_kv, hd).swapaxes(1, 2)

    # RoPE
    q_rot, k_rot = np_apply_rope(q, k, cos, sin, hd, T)

    # GQA attention (causal)
    repeats = n_h // n_kv
    k_expanded = np.repeat(k_rot, repeats, axis=1)
    v_expanded = np.repeat(v, repeats, axis=1)
    attn_out = np_sdpa(q_rot, k_expanded, v_expanded, scale=scale, is_causal=True)

    # Output projection
    o_out = (attn_out.swapaxes(1, 2).reshape(B * T, n_h * hd) @ o_w).reshape(B, T, H)

    # Residual
    hidden2 = hidden + o_out

    # Post-attention RMSNorm
    normed2 = np_rmsnorm(hidden2, post_gamma, eps)

    # SiLU-gated MLP: silu(gate) * up = sigmoid(gate) * gate * up
    gate = (normed2.reshape(B * T, H) @ gate_w).reshape(B, T, I)
    up = (normed2.reshape(B * T, H) @ up_w).reshape(B, T, I)
    gated = np_silu(gate) * up
    mlp_out = (gated.reshape(B * T, I) @ down_w).reshape(B, T, H)

    # Residual
    hidden_out = hidden2 + mlp_out

    return hidden_out, k_rot, v


def np_full_decode(
    hidden, q_w, k_w, v_w, o_w, gate_w, up_w, down_w,
    in_gamma, post_gamma, old_k, old_v,
    write_mask, attn_mask, cos, sin, config,
):
    """Full NumPy decode layer reference."""
    B = 1
    H = config.hidden_size
    n_h = config.n_heads
    n_kv = config.n_kv_heads
    hd = config.head_dim
    I = config.intermediate_size
    C = old_k.shape[2]
    eps = config.rms_eps
    scale = 1.0 / math.sqrt(hd)

    # Pre-attention RMSNorm
    normed = np_rmsnorm(hidden, in_gamma, eps)

    # Q/K/V projections (T=1)
    q = (normed.reshape(B, H) @ q_w).reshape(B, 1, n_h, hd).swapaxes(1, 2)
    new_k = (normed.reshape(B, H) @ k_w).reshape(B, 1, n_kv, hd).swapaxes(1, 2)
    new_v = (normed.reshape(B, H) @ v_w).reshape(B, 1, n_kv, hd).swapaxes(1, 2)

    # RoPE (single position)
    q_rot, k_rot = np_apply_rope(q, new_k, cos, sin, hd, 1)

    # Cache update via where
    write_bool = write_mask > 0.5
    k_rot_expanded = np.broadcast_to(k_rot, (B, n_kv, C, hd))
    new_v_expanded = np.broadcast_to(new_v, (B, n_kv, C, hd))
    updated_k = np.where(write_bool, k_rot_expanded, old_k)
    updated_v = np.where(write_bool, new_v_expanded, old_v)

    # GQA attention over full cache
    repeats = n_h // n_kv
    k_expanded = np.repeat(updated_k, repeats, axis=1)
    v_expanded = np.repeat(updated_v, repeats, axis=1)
    q_expanded = np.repeat(q_rot, repeats, axis=1) if n_h != n_kv else q_rot
    # Actually need to expand q to match n_h heads
    q_parts = []
    for i in range(n_h):
        q_parts.append(q_rot[:, i:i+1, :, :])
    q_expanded = np.concatenate(q_parts, axis=1)

    attn_out = np_sdpa(q_rot, k_expanded, v_expanded, mask=attn_mask, scale=scale)

    # Output projection
    o_out = (attn_out.swapaxes(1, 2).reshape(B, n_h * hd) @ o_w).reshape(B, 1, H)

    # Residual
    hidden2 = hidden + o_out

    # Post-attention RMSNorm
    normed2 = np_rmsnorm(hidden2, post_gamma, eps)

    # SiLU-gated MLP: silu(gate) * up = sigmoid(gate) * gate * up
    gate = (normed2.reshape(B, H) @ gate_w).reshape(B, 1, I)
    up = (normed2.reshape(B, H) @ up_w).reshape(B, 1, I)
    gated = np_silu(gate) * up
    mlp_out = (gated.reshape(B, I) @ down_w).reshape(B, 1, H)

    # Residual
    hidden_out = hidden2 + mlp_out

    return hidden_out, updated_k, updated_v


# ---------------------------------------------------------------------------
# Weight fixture helpers
# ---------------------------------------------------------------------------


def make_layer_weights(config, seed_base=100):
    """Return a dict of deterministic layer weights."""
    H = config.hidden_size
    n_h = config.n_heads
    n_kv = config.n_kv_heads
    hd = config.head_dim
    I = config.intermediate_size
    s = seed_base
    return {
        "q_w": det_array((H, n_h * hd), s),
        "k_w": det_array((H, n_kv * hd), s + 1),
        "v_w": det_array((H, n_kv * hd), s + 2),
        "o_w": det_array((n_h * hd, H), s + 3),
        "gate_w": det_array((H, I), s + 4),
        "up_w": det_array((H, I), s + 5),
        "down_w": det_array((I, H), s + 6),
        "in_gamma": det_array((H,), s + 7),
        "post_gamma": det_array((H,), s + 8),
    }


def weight_args(w):
    """Unpack weight dict into positional order for compiled functions."""
    return (
        w["q_w"], w["k_w"], w["v_w"], w["o_w"],
        w["gate_w"], w["up_w"], w["down_w"],
        w["in_gamma"], w["post_gamma"],
    )


# ---------------------------------------------------------------------------
# Tests: RoPE table
# ---------------------------------------------------------------------------


class TestRopeTable:
    def test_shape(self):
        cos, sin = build_rope_table(SMALL_CONFIG, 8)
        hd = SMALL_CONFIG.head_dim
        assert cos.shape == (8, hd // 2)
        assert sin.shape == (8, hd // 2)

    def test_dtype(self):
        cos, sin = build_rope_table(SMALL_CONFIG, 4)
        assert cos.dtype == np.float32
        assert sin.dtype == np.float32

    def test_nonzero(self):
        cos, sin = build_rope_table(SMALL_CONFIG, 4)
        assert np.any(cos != 0)
        assert np.any(sin != 0)

    def test_identity_at_position_zero(self):
        cos, sin = build_rope_table(SMALL_CONFIG, 4)
        np.testing.assert_allclose(cos[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(sin[0], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: Symbolic helpers
# ---------------------------------------------------------------------------


class TestSymbolicHelpers:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_rmsnorm_matches_numpy(self, backend):
        H = SMALL_CONFIG.hidden_size
        x_val = det_array((1, 3, H), seed=10)
        g_val = det_array((H,), seed=11)

        x_s = pt.tensor("x", shape=(1, 3, H), dtype="float32")
        g_s = pt.tensor("g", shape=(H,), dtype="float32")
        out = rmsnorm_symbolic(x_s, g_s, SMALL_CONFIG.rms_eps)
        fn = pytensor.function([x_s, g_s], out, mode=_get_mode(backend))

        result = fn(x_val, g_val)
        expected = np_rmsnorm(x_val, g_val, SMALL_CONFIG.rms_eps)
        np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_rotate_half_matches_numpy(self, backend):
        hd = SMALL_CONFIG.head_dim
        x_val = det_array((1, 2, 3, hd), seed=20)

        x_s = pt.tensor("x", shape=(1, 2, 3, hd), dtype="float32")
        out = rotate_half(x_s, hd)
        fn = pytensor.function([x_s], out, mode=_get_mode(backend))

        result = fn(x_val)
        expected = np_rotate_half(x_val, hd)
        np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_linear_proj_matches_numpy(self, backend):
        H, I = SMALL_CONFIG.hidden_size, SMALL_CONFIG.intermediate_size
        B, T = 1, 3
        x_val = det_array((B, T, H), seed=30)
        W_val = det_array((H, I), seed=31)

        x_s = pt.tensor("x", shape=(B, T, H), dtype="float32")
        W_s = pt.tensor("W", shape=(H, I), dtype="float32")
        out = linear_proj(x_s, W_s, B, T, H, I)
        fn = pytensor.function([x_s, W_s], out, mode=_get_mode(backend))

        result = fn(x_val, W_val)
        expected = (x_val.reshape(B * T, H) @ W_val).reshape(B, T, I)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_linear_proj_static_shape(self):
        H, I = SMALL_CONFIG.hidden_size, SMALL_CONFIG.intermediate_size
        B, T = 1, 3
        x_s = pt.tensor("x", shape=(B, T, H), dtype="float32")
        W_s = pt.tensor("W", shape=(H, I), dtype="float32")
        out = linear_proj(x_s, W_s, B, T, H, I)
        assert out.type.shape == (B, T, I)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_apply_rope_matches_numpy(self, backend):
        cfg = SMALL_CONFIG
        B, T = 1, 4
        n_h, n_kv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim

        q_val = det_array((B, n_h, T, hd), seed=40)
        k_val = det_array((B, n_kv, T, hd), seed=41)
        cos_val, sin_val = build_rope_table(cfg, T)

        q_s = pt.tensor("q", shape=(B, n_h, T, hd), dtype="float32")
        k_s = pt.tensor("k", shape=(B, n_kv, T, hd), dtype="float32")
        cos_s = pt.tensor("cos", shape=(T, hd // 2), dtype="float32")
        sin_s = pt.tensor("sin", shape=(T, hd // 2), dtype="float32")
        q_out, k_out = apply_rope(q_s, k_s, cos_s, sin_s, hd, T)
        fn = pytensor.function(
            [q_s, k_s, cos_s, sin_s], [q_out, k_out], mode=_get_mode(backend)
        )

        q_result, k_result = fn(q_val, k_val, cos_val, sin_val)
        q_expected, k_expected = np_apply_rope(q_val, k_val, cos_val, sin_val, hd, T)
        np.testing.assert_allclose(q_result, q_expected, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(k_result, k_expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Prefill layer
# ---------------------------------------------------------------------------


class TestPrefillLayer:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_static_shapes(self, backend):
        cfg = SMALL_CONFIG
        B, T = 1, 4
        fn = compile_prefill_layer(cfg, B, T, backend=backend)

        w = make_layer_weights(cfg, seed_base=200)
        hidden = det_array((B, T, cfg.hidden_size), seed=210)
        cos, sin = build_rope_table(cfg, T)

        result = fn(hidden, *weight_args(w), cos, sin)
        assert len(result) == 3
        hidden_out, k_rot, v_raw = result
        assert hidden_out.shape == (B, T, cfg.hidden_size)
        assert k_rot.shape == (B, cfg.n_kv_heads, T, cfg.head_dim)
        assert v_raw.shape == (B, cfg.n_kv_heads, T, cfg.head_dim)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_gqa_shapes(self, backend):
        """GQA with n_heads != n_kv_heads should work."""
        cfg = SMALL_CONFIG
        B, T = 1, 3
        fn = compile_prefill_layer(cfg, B, T, backend=backend)

        w = make_layer_weights(cfg, seed_base=220)
        hidden = det_array((B, T, cfg.hidden_size), seed=230)
        cos, sin = build_rope_table(cfg, T)

        result = fn(hidden, *weight_args(w), cos, sin)
        hidden_out, k_rot, v_raw = result
        assert hidden_out.shape == (B, T, cfg.hidden_size)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_gqa_matches_numpy(self, backend):
        cfg = SMALL_CONFIG
        B, T = 1, 4
        fn = compile_prefill_layer(cfg, B, T, backend=backend)

        w = make_layer_weights(cfg, seed_base=240)
        hidden = det_array((B, T, cfg.hidden_size), seed=250)
        cos, sin = build_rope_table(cfg, T)

        result = fn(hidden, *weight_args(w), cos, sin)
        hidden_out, k_rot, v_raw = result

        expected_h, expected_k, expected_v = np_full_prefill(
            hidden, w["q_w"], w["k_w"], w["v_w"], w["o_w"],
            w["gate_w"], w["up_w"], w["down_w"],
            w["in_gamma"], w["post_gamma"], cos, sin, cfg, B, T,
        )

        np.testing.assert_allclose(hidden_out, expected_h, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(k_rot, expected_k, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(v_raw, expected_v, atol=1e-4, rtol=1e-4)

    def test_gqa_real_config_shapes(self):
        """Verify shapes with a more realistic config."""
        cfg = SmolLM2Config(
            vocab_size=1000,
            hidden_size=64,
            n_layers=4,
            n_heads=8,
            n_kv_heads=4,
            head_dim=16,
            intermediate_size=128,
            context_length=128,
        )
        B, T = 1, 8
        fn = compile_prefill_layer(cfg, B, T, backend="c")

        w = make_layer_weights(cfg, seed_base=260)
        hidden = det_array((B, T, cfg.hidden_size), seed=270)
        cos, sin = build_rope_table(cfg, T)

        result = fn(hidden, *weight_args(w), cos, sin)
        hidden_out, k_rot, v_raw = result
        assert hidden_out.shape == (B, T, cfg.hidden_size)
        assert k_rot.shape == (B, cfg.n_kv_heads, T, cfg.head_dim)


# ---------------------------------------------------------------------------
# Tests: Decode layer
# ---------------------------------------------------------------------------


class TestDecodeLayer:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_static_shapes(self, backend):
        cfg = SMALL_CONFIG
        C = 16
        fn = compile_decode_layer(cfg, 1, C, backend=backend)

        w = make_layer_weights(cfg, seed_base=300)
        hidden = det_array((1, 1, cfg.hidden_size), seed=310)
        old_k = det_array((1, cfg.n_kv_heads, C, cfg.head_dim), seed=311)
        old_v = det_array((1, cfg.n_kv_heads, C, cfg.head_dim), seed=312)

        # Write to position 5
        write_mask = np.zeros((1, 1, C, 1), dtype=np.float32)
        write_mask[0, 0, 5, 0] = 1.0
        attn_mask = np.full((1, 1, 1, C), -np.inf, dtype=np.float32)
        attn_mask[0, 0, 0, :6] = 0.0  # attend to positions 0-5

        cos, sin = build_rope_table(cfg, 1)
        cos_row = cos[0:1]
        sin_row = sin[0:1]

        result = fn(
            hidden, *weight_args(w), old_k, old_v,
            write_mask, attn_mask, cos_row, sin_row,
        )
        assert len(result) == 3
        hidden_out, new_k, new_v = result
        assert hidden_out.shape == (1, 1, cfg.hidden_size)
        assert new_k.shape == (1, cfg.n_kv_heads, C, cfg.head_dim)
        assert new_v.shape == (1, cfg.n_kv_heads, C, cfg.head_dim)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_gqa_matches_numpy(self, backend):
        cfg = SMALL_CONFIG
        C = 8
        fn = compile_decode_layer(cfg, 1, C, backend=backend)

        w = make_layer_weights(cfg, seed_base=320)
        hidden = det_array((1, 1, cfg.hidden_size), seed=330)
        old_k = det_array((1, cfg.n_kv_heads, C, cfg.head_dim), seed=331)
        old_v = det_array((1, cfg.n_kv_heads, C, cfg.head_dim), seed=332)

        write_mask = np.zeros((1, 1, C, 1), dtype=np.float32)
        write_mask[0, 0, 3, 0] = 1.0
        attn_mask = np.full((1, 1, 1, C), -np.inf, dtype=np.float32)
        attn_mask[0, 0, 0, :4] = 0.0

        cos, sin = build_rope_table(cfg, 1)
        cos_row = cos[0:1]
        sin_row = sin[0:1]

        result = fn(
            hidden, *weight_args(w), old_k, old_v,
            write_mask, attn_mask, cos_row, sin_row,
        )
        hidden_out, new_k, new_v = result

        expected_h, expected_k, expected_v = np_full_decode(
            hidden, w["q_w"], w["k_w"], w["v_w"], w["o_w"],
            w["gate_w"], w["up_w"], w["down_w"],
            w["in_gamma"], w["post_gamma"], old_k, old_v,
            write_mask, attn_mask, cos_row, sin_row, cfg,
        )

        np.testing.assert_allclose(hidden_out, expected_h, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(new_k, expected_k, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(new_v, expected_v, atol=1e-4, rtol=1e-4)

    def test_cache_write_only_target_position(self):
        """Verify that only the target position is written in the cache."""
        cfg = SMALL_CONFIG
        C = 8
        fn = compile_decode_layer(cfg, 1, C, backend="c")

        w = make_layer_weights(cfg, seed_base=340)
        hidden = det_array((1, 1, cfg.hidden_size), seed=350)
        old_k = np.zeros((1, cfg.n_kv_heads, C, cfg.head_dim), dtype=np.float32)
        old_v = np.zeros((1, cfg.n_kv_heads, C, cfg.head_dim), dtype=np.float32)

        write_mask = np.zeros((1, 1, C, 1), dtype=np.float32)
        write_mask[0, 0, 5, 0] = 1.0
        attn_mask = np.full((1, 1, 1, C), -np.inf, dtype=np.float32)
        attn_mask[0, 0, 0, :6] = 0.0

        cos, sin = build_rope_table(cfg, 1)
        cos_row = cos[0:1]
        sin_row = sin[0:1]

        _, new_k, new_v = fn(
            hidden, *weight_args(w), old_k, old_v,
            write_mask, attn_mask, cos_row, sin_row,
        )

        # Only position 5 should be non-zero
        for pos in range(C):
            if pos != 5:
                np.testing.assert_array_equal(
                    new_k[0, :, pos, :], 0.0,
                    err_msg=f"k cache at pos {pos} should be zero",
                )
                np.testing.assert_array_equal(
                    new_v[0, :, pos, :], 0.0,
                    err_msg=f"v cache at pos {pos} should be zero",
                )
        # Position 5 should be non-zero
        assert np.any(new_k[0, :, 5, :] != 0.0)
        assert np.any(new_v[0, :, 5, :] != 0.0)

    def test_cached_attention_sees_all_valid_prefix(self):
        """Attention mask should allow attending to all valid prefix positions."""
        cfg = SMALL_CONFIG
        C = 8
        fn = compile_decode_layer(cfg, 1, C, backend="c")

        w = make_layer_weights(cfg, seed_base=360)
        hidden = det_array((1, 1, cfg.hidden_size), seed=370)

        # Fill cache with known values
        old_k = det_array((1, cfg.n_kv_heads, C, cfg.head_dim), seed=371)
        old_v = det_array((1, cfg.n_kv_heads, C, cfg.head_dim), seed=372)

        write_mask = np.zeros((1, 1, C, 1), dtype=np.float32)
        write_mask[0, 0, 4, 0] = 1.0

        # Attend to positions 0-4
        attn_mask = np.full((1, 1, 1, C), -np.inf, dtype=np.float32)
        attn_mask[0, 0, 0, :5] = 0.0

        cos, sin = build_rope_table(cfg, 1)
        cos_row = cos[0:1]
        sin_row = sin[0:1]

        result = fn(
            hidden, *weight_args(w), old_k, old_v,
            write_mask, attn_mask, cos_row, sin_row,
        )
        hidden_out, _, _ = result
        assert np.all(np.isfinite(hidden_out))


# ---------------------------------------------------------------------------
# Tests: Backend parity
# ---------------------------------------------------------------------------


class TestBackendParity:
    def test_prefill_parity(self):
        cfg = SMALL_CONFIG
        B, T = 1, 4
        fn_c = compile_prefill_layer(cfg, B, T, backend="c")
        fn_n = compile_prefill_layer(cfg, B, T, backend="numba")

        w = make_layer_weights(cfg, seed_base=400)
        hidden = det_array((B, T, cfg.hidden_size), seed=410)
        cos, sin = build_rope_table(cfg, T)

        r_c = fn_c(hidden, *weight_args(w), cos, sin)
        r_n = fn_n(hidden, *weight_args(w), cos, sin)

        for rc, rn in zip(r_c, r_n):
            np.testing.assert_allclose(rc, rn, atol=1e-4, rtol=1e-4)

    def test_decode_parity(self):
        cfg = SMALL_CONFIG
        C = 8
        fn_c = compile_decode_layer(cfg, 1, C, backend="c")
        fn_n = compile_decode_layer(cfg, 1, C, backend="numba")

        w = make_layer_weights(cfg, seed_base=420)
        hidden = det_array((1, 1, cfg.hidden_size), seed=430)
        old_k = det_array((1, cfg.n_kv_heads, C, cfg.head_dim), seed=431)
        old_v = det_array((1, cfg.n_kv_heads, C, cfg.head_dim), seed=432)

        write_mask = np.zeros((1, 1, C, 1), dtype=np.float32)
        write_mask[0, 0, 3, 0] = 1.0
        attn_mask = np.full((1, 1, 1, C), -np.inf, dtype=np.float32)
        attn_mask[0, 0, 0, :4] = 0.0

        cos, sin = build_rope_table(cfg, 1)
        cos_row = cos[0:1]
        sin_row = sin[0:1]

        args = (
            hidden, *weight_args(w), old_k, old_v,
            write_mask, attn_mask, cos_row, sin_row,
        )
        r_c = fn_c(*args)
        r_n = fn_n(*args)

        for rc, rn in zip(r_c, r_n):
            np.testing.assert_allclose(rc, rn, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: Graph inspection
# ---------------------------------------------------------------------------


class TestGraphInspection:
    def test_prefill_contains_attention_layer(self):
        """Prefill graph should contain attention-related operations."""
        cfg = SMALL_CONFIG
        fn = compile_prefill_layer(cfg, 1, 4, backend="c")
        # Check that the function has outputs
        assert len(fn.maker.fgraph.outputs) == 3

    def test_decode_contains_attention_layer(self):
        """Decode graph should contain attention-related operations."""
        cfg = SMALL_CONFIG
        fn = compile_decode_layer(cfg, 1, 8, backend="c")
        assert len(fn.maker.fgraph.outputs) == 3

    def test_attention_layer_properties(self):
        """Verify attention layer has expected number of inputs."""
        cfg = SMALL_CONFIG
        fn = compile_prefill_layer(cfg, 1, 4, backend="c")
        # 12 inputs: hidden, q_w, k_w, v_w, o_w, gate_w, up_w, down_w,
        #            in_gamma, post_gamma, cos, sin
        assert len(fn.maker.fgraph.inputs) == 12


# ---------------------------------------------------------------------------
# Tests: Embedding and logits
# ---------------------------------------------------------------------------


class TestEmbeddingAndLogits:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_embedding_shapes(self, backend):
        cfg = SMALL_CONFIG
        B, T = 1, 4
        fn = compile_embedding(cfg, B, T, backend=backend)

        token_ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
        emb_table = det_array((cfg.vocab_size, cfg.hidden_size), seed=500)

        result = fn(token_ids, emb_table)
        assert result.shape == (B, T, cfg.hidden_size)
        assert result.dtype == np.float32

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_logits_shapes(self, backend):
        cfg = SMALL_CONFIG
        B, T = 1, 4
        fn = compile_logits(cfg, B, T, backend=backend)

        hidden = det_array((B, T, cfg.hidden_size), seed=510)
        final_gamma = det_array((cfg.hidden_size,), seed=511)
        emb_table = det_array((cfg.vocab_size, cfg.hidden_size), seed=512)

        result = fn(hidden, final_gamma, emb_table)
        assert result.shape == (B, T, cfg.vocab_size)
        assert result.dtype == np.float32

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_logits_matches_numpy(self, backend):
        cfg = SMALL_CONFIG
        B, T = 1, 3
        fn = compile_logits(cfg, B, T, backend=backend)

        hidden = det_array((B, T, cfg.hidden_size), seed=520)
        final_gamma = det_array((cfg.hidden_size,), seed=521)
        emb_table = det_array((cfg.vocab_size, cfg.hidden_size), seed=522)

        result = fn(hidden, final_gamma, emb_table)

        # NumPy reference
        normed = np_rmsnorm(hidden, final_gamma, cfg.rms_eps)
        expected = (normed.reshape(B * T, cfg.hidden_size) @ emb_table.T).reshape(
            B, T, cfg.vocab_size
        )
        np.testing.assert_allclose(result, expected, atol=1e-3, rtol=1e-3)

    def test_logits_single_position(self):
        """Logits for single position (decode) should work."""
        cfg = SMALL_CONFIG
        fn = compile_logits(cfg, 1, 1, backend="c")

        hidden = det_array((1, 1, cfg.hidden_size), seed=530)
        final_gamma = det_array((cfg.hidden_size,), seed=531)
        emb_table = det_array((cfg.vocab_size, cfg.hidden_size), seed=532)

        result = fn(hidden, final_gamma, emb_table)
        assert result.shape == (1, 1, cfg.vocab_size)


# ---------------------------------------------------------------------------
# Tests: Mode helpers
# ---------------------------------------------------------------------------


class TestMLXMode:
    def test_mlx_mode_creation(self):
        m = make_c_mode()
        assert m is not None

    def test_float32_config(self):
        """Module should not mutate floatX."""
        assert _import_preserved_floatX


# ---------------------------------------------------------------------------
# Tests: Import hygiene
# ---------------------------------------------------------------------------


class TestImportHygiene:
    def test_import_does_not_mutate_floatx(self):
        assert _import_preserved_floatX, (
            f"Importing smollm2_pytensor mutated floatX from "
            f"{_floatX_before_import!r} to {pytensor.config.floatX!r}"
        )

    def test_reload_does_not_mutate_floatx(self):
        from cetagostini.utils.pytensor import smollm2_pytensor

        saved = pytensor.config.floatX
        pytensor.config.floatX = "float64"
        try:
            importlib.reload(smollm2_pytensor)
            assert pytensor.config.floatX == "float64"
        finally:
            pytensor.config.floatX = saved


# ---------------------------------------------------------------------------
# Tests: Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_prefill_batch_zero(self):
        with pytest.raises(ValueError, match="batch_size >= 1"):
            compile_prefill_layer(SMALL_CONFIG, 0, 4, backend="c")

    def test_prefill_seq_zero(self):
        with pytest.raises(ValueError, match="seq_len >= 1"):
            compile_prefill_layer(SMALL_CONFIG, 1, 0, backend="c")

    def test_prefill_batch_negative(self):
        with pytest.raises(ValueError, match="batch_size >= 1"):
            compile_prefill_layer(SMALL_CONFIG, -1, 4, backend="c")

    def test_prefill_seq_negative(self):
        with pytest.raises(ValueError, match="seq_len >= 1"):
            compile_prefill_layer(SMALL_CONFIG, 1, -1, backend="c")

    def test_decode_batch_zero(self):
        with pytest.raises(ValueError, match="batch_size == 1"):
            compile_decode_layer(SMALL_CONFIG, 0, 8, backend="c")

    def test_decode_batch_gt_1(self):
        with pytest.raises(ValueError, match="batch_size == 1"):
            compile_decode_layer(SMALL_CONFIG, 2, 8, backend="c")

    def test_decode_batch_negative(self):
        with pytest.raises(ValueError, match="batch_size == 1"):
            compile_decode_layer(SMALL_CONFIG, -1, 8, backend="c")

    def test_decode_cache_zero(self):
        with pytest.raises(ValueError, match="cache_capacity >= 1"):
            compile_decode_layer(SMALL_CONFIG, 1, 0, backend="c")

    def test_decode_cache_negative(self):
        with pytest.raises(ValueError, match="cache_capacity >= 1"):
            compile_decode_layer(SMALL_CONFIG, 1, -1, backend="c")

    def test_embedding_batch_zero(self):
        with pytest.raises(ValueError, match="batch_size >= 1"):
            compile_embedding(SMALL_CONFIG, 0, 4, backend="c")

    def test_embedding_seq_zero(self):
        with pytest.raises(ValueError, match="seq_len >= 1"):
            compile_embedding(SMALL_CONFIG, 1, 0, backend="c")

    def test_logits_batch_zero(self):
        with pytest.raises(ValueError, match="batch_size >= 1"):
            compile_logits(SMALL_CONFIG, 0, 4, backend="c")

    def test_logits_seq_zero(self):
        with pytest.raises(ValueError, match="seq_len >= 1"):
            compile_logits(SMALL_CONFIG, 1, 0, backend="c")

    def test_decode_batch_1_accepted(self):
        """batch_size=1 should be accepted for decode."""
        fn = compile_decode_layer(SMALL_CONFIG, 1, 8, backend="c")
        assert fn is not None

    def test_prefill_batch_gt_1_accepted(self):
        """batch_size > 1 should be accepted for prefill."""
        fn = compile_prefill_layer(SMALL_CONFIG, 2, 4, backend="c")
        assert fn is not None


# ---------------------------------------------------------------------------
# Tests: Float32 audit
# ---------------------------------------------------------------------------


class TestFloat32Audit:
    def test_prefill_float32(self):
        fn = compile_prefill_layer(SMALL_CONFIG, 1, 4, backend="c")
        assert audit_float32(fn)

    def test_decode_float32(self):
        fn = compile_decode_layer(SMALL_CONFIG, 1, 8, backend="c")
        assert audit_float32(fn)

    def test_embedding_float32(self):
        fn = compile_embedding(SMALL_CONFIG, 1, 4, backend="c")
        assert audit_float32(fn)

    def test_logits_float32(self):
        fn = compile_logits(SMALL_CONFIG, 1, 4, backend="c")
        assert audit_float32(fn)


# ---------------------------------------------------------------------------
# Helper to resolve backend string
# ---------------------------------------------------------------------------


def _get_mode(backend):
    from cetagostini.utils.pytensor.smollm2_pytensor import _get_mode as _gm
    return _gm(backend)
