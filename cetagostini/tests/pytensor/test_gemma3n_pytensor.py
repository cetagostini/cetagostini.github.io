"""Tests for gemma3n_pytensor PyTensor graph builders.

All fixtures are deterministic, asymmetric, and nonzero.
Independent eager NumPy oracle functions are defined here (not imported
from the implementation module).  Every primitive and all four layer
kinds are validated parametrically through both C and Numba backends.

Shared-KV equivalence tests verify that providing precomputed K/V
produces identical attention output to computing them internally.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import pytensor

# Capture floatX *before* importing gemma3n_pytensor so we can verify
# the module does not mutate it.
_floatX_before_import = pytensor.config.floatX

import pytensor.tensor as pt

from cetagostini.utils.pytensor.gemma3n_pytensor import (
    Gemma3nConfig,
    altup_correct_symbolic,
    altup_predict_symbolic,
    altup_router_modalities_symbolic,
    apply_rope_symbolic,
    attention_block_symbolic,
    audit_float32,
    build_rope_table,
    causal_mask,
    chunked_logit_projection,
    compile_decoder_layer,
    compile_final_unembed,
    compile_initial_projections,
    compile_logit_projection,
    compile_per_chunk_logits,
    compile_per_layer_projection,
    decoder_layer_symbolic,
    final_unembed,
    gqa_attention,
    gelu_approx_symbolic,
    initial_stream_projections,
    laurel_symbolic,
    linear_proj,
    make_c_mode,
    make_numba_mode,
    mlp_symbolic,
    per_layer_input_projection,
    rms_norm_no_scale,
    rmsnorm_symbolic,
    sliding_window_mask,
    sparse_gelu_symbolic,
)

_import_preserved_floatX = pytensor.config.floatX == _floatX_before_import


# ---------------------------------------------------------------------------
# Test configuration — small nonsymmetric dimensions
# ---------------------------------------------------------------------------

SMALL_CONFIG = Gemma3nConfig(
    hidden_size=24,
    num_hidden_layers=2,
    intermediate_size=36,
    num_attention_heads=4,
    head_dim=12,
    rms_norm_eps=1e-5,
    vocab_size=64,
    num_key_value_heads=2,
    sliding_window=4,
    rope_local_base_freq=10_000.0,
    rope_theta=100_000.0,
    final_logit_softcapping=30.0,
    activation_sparsity=0.6,
    hidden_size_per_layer_input=8,
    altup_num_inputs=3,
    altup_coef_clip=0.5,
    altup_correct_scale=True,
    altup_active_idx=0,
    laurel_rank=6,
    vocab_size_per_layer_input=32,
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


def np_rms_no_scale(x, eps):
    """NumPy RMSNorm without scale."""
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(var + eps))


def np_gelu_approx(x):
    """NumPy approximate tanh GELU."""
    return 0.5 * x * (1.0 + np.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    ))


def np_sparse_gelu(x, std_multiplier):
    """NumPy sparse GELU with exact mean/std/cutoff."""
    mu = np.mean(x, axis=-1, keepdims=True)
    diff = x - mu
    sigma = np.sqrt(np.mean(diff ** 2, axis=-1, keepdims=True))
    cutoff = mu + sigma * np.float32(std_multiplier)
    return np_gelu_approx(np.maximum(0.0, x - cutoff))


def np_apply_rope(x, cos, sin, head_dim):
    """NumPy half-split RoPE (traditional=False)."""
    half = head_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return np.concatenate([rot1, rot2], axis=-1)


def np_gqa_attention(q, k, v, mask, n_heads, n_kv_heads, scale):
    """NumPy GQA attention with head repeat."""
    repeats = n_heads // n_kv_heads
    if repeats > 1:
        k = np.repeat(k, repeats, axis=1)
        v = np.repeat(v, repeats, axis=1)
    scores = np.matmul(q, k.swapaxes(-1, -2)) * scale
    if mask is not None:
        scores = scores + mask
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
    attn_weights = scores_exp / scores_sum
    return np.matmul(attn_weights, v)


def np_laurel(x, W_left, W_right, norm_gamma, B, T, H, rank, eps):
    """NumPy LAuReL."""
    lx = (x.reshape(B * T, H) @ W_left).reshape(B, T, rank)
    lx = (lx.reshape(B * T, rank) @ W_right).reshape(B, T, H)
    normed = np_rmsnorm(lx, norm_gamma, eps)
    return x + normed


def np_mlp(x, gate_W, up_W, down_W, B, T, H, I, std_mult=None):
    """NumPy GELU-gated MLP."""
    gate = (x.reshape(B * T, H) @ gate_W).reshape(B, T, I)
    if std_mult is not None and std_mult > 0:
        act = np_sparse_gelu(gate, std_mult)
    else:
        act = np_gelu_approx(gate)
    up = (x.reshape(B * T, H) @ up_W).reshape(B, T, I)
    gated = act * up
    out = (gated.reshape(B * T, I) @ down_W).reshape(B, T, H)
    return out


def np_router_modalities(x, router_norm_gamma, router_W, B, T, H, n, eps):
    """NumPy AltUp router modalities."""
    rn = np_rmsnorm(x, router_norm_gamma, eps) * (1.0 / H)
    routed = (rn.reshape(B * T, H) @ router_W).reshape(B, T, n)
    return np.tanh(routed)


def np_altup_predict(
    x, pred_W, rn_gamma, router_W, B, T, H, n, eps, clip, active_idx
):
    """NumPy AltUp predict."""
    modalities = np_router_modalities(
        x[active_idx], rn_gamma, router_W, B, T, H, n, eps
    )
    W = np.clip(pred_W, -clip, clip) if clip is not None else pred_W
    coefs_flat = (modalities.reshape(B * T, n) @ W).reshape(B, T, n * n)
    all_coefs = coefs_flat.reshape(B, T, n, n).transpose(0, 1, 3, 2)
    x_up = x.astype(np.float32)
    x_perm = x_up.transpose(1, 2, 3, 0)
    preds = np.matmul(x_perm, all_coefs)
    preds = preds.transpose(3, 0, 1, 2)
    preds = preds + x_up
    return preds


def np_altup_correct(
    predictions, activated, corr_W, rn_gamma, router_W,
    B, T, H, n, eps, clip, active_idx,
):
    """NumPy AltUp correct."""
    modalities = np_router_modalities(
        activated, rn_gamma, router_W, B, T, H, n, eps
    )
    W = np.clip(corr_W, -clip, clip) if clip is not None else corr_W
    coefs = (modalities.reshape(B * T, n) @ W).reshape(B, T, n) + 1.0
    active_x = predictions[active_idx]
    innovation = activated - active_x
    coefs_t = coefs.transpose(2, 0, 1)
    corrected = innovation[None] * coefs_t[..., None]
    corrected = corrected + predictions
    return corrected


def np_attention_block(
    x, q_w, k_w, v_w, o_w, q_ng, k_ng, cos, sin, mask,
    B, T, H, n_h, n_kv, hd, eps,
):
    """NumPy attention block."""
    q = (x.reshape(B * T, H) @ q_w).reshape(B, T, n_h, hd).swapaxes(1, 2)
    k = (x.reshape(B * T, H) @ k_w).reshape(B, T, n_kv, hd).swapaxes(1, 2)
    v = (x.reshape(B * T, H) @ v_w).reshape(B, T, n_kv, hd).swapaxes(1, 2)
    q = np_rmsnorm(q, q_ng, eps)
    k = np_rmsnorm(k, k_ng, eps)
    v = np_rms_no_scale(v, eps)
    cos_exp = cos.reshape(1, 1, T, hd // 2)
    sin_exp = sin.reshape(1, 1, T, hd // 2)
    q = np_apply_rope(q, cos_exp, sin_exp, hd)
    k = np_apply_rope(k, cos_exp, sin_exp, hd)
    attn = np_gqa_attention(q, k, v, mask, n_h, n_kv, 1.0)
    out = attn.swapaxes(1, 2).reshape(B * T, n_h * hd) @ o_w
    return out.reshape(B, T, H)


def np_decoder_layer(
    x, mask, pli, cos, sin,
    q_w, k_w, v_w, o_w, q_ng, k_ng,
    gate_w, up_w, down_w,
    ll_w, lr_w, ln_g,
    pred_w, corr_w, mr_w, rn_g, cos_scale,
    iln_g, paln_g, pfln_g, pfln2_g, plin_g,
    pli_gw, pli_pw,
    config, B, T, std_mult=None,
):
    """Full NumPy decoder layer reference."""
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
    clip = config.altup_coef_clip

    predictions = np_altup_predict(
        x, pred_w, rn_g, mr_w, B, T, H, n, eps, clip, active_idx,
    )
    active_pred = predictions[active_idx]
    active_pred_normed = np_rmsnorm(active_pred, iln_g, eps)
    laurel_out = np_laurel(
        active_pred_normed, ll_w, lr_w, ln_g, B, T, H, rank, eps,
    )
    attn = np_attention_block(
        active_pred_normed, q_w, k_w, v_w, o_w, q_ng, k_ng,
        cos, sin, mask, B, T, H, n_h, n_kv, hd, eps,
    )
    attn = np_rmsnorm(attn, paln_g, eps)
    attn_gated = active_pred + attn
    attn_laurel = (attn_gated + laurel_out) * (2.0 ** -0.5)
    attn_norm = np_rmsnorm(attn_laurel, pfln_g, eps)
    attn_ffw = np_mlp(
        attn_norm, gate_w, up_w, down_w, B, T, H, I, std_mult,
    )
    attn_ffw_norm = np_rmsnorm(attn_ffw, pfln2_g, eps)
    attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm

    corrected = np_altup_correct(
        predictions, attn_ffw_laurel_gated, corr_w, rn_g, mr_w,
        B, T, H, n, eps, clip, active_idx,
    )

    first_pred = corrected[active_idx]
    if config.altup_correct_scale:
        first_pred = first_pred * cos_scale

    first_pred = (first_pred.reshape(B * T, H) @ pli_gw).reshape(B, T, H_pl)
    first_pred = np_gelu_approx(first_pred)
    first_pred = first_pred * pli
    first_pred = (first_pred.reshape(B * T, H_pl) @ pli_pw).reshape(B, T, H)
    first_pred = np_rmsnorm(first_pred, plin_g, eps)

    result = corrected.copy()
    result[1:] = result[1:] + first_pred[None]
    return result


def np_initial_projections(h0, proj_ws, B, T, H, n):
    """NumPy initial stream projections + magnitude matching."""
    target_mag = np.sqrt(np.mean(h0 ** 2, axis=-1, keepdims=True))
    h_list = [h0]
    for w in proj_ws:
        h_list.append((h0.reshape(B * T, H) @ w).reshape(B, T, H))
    h = np.stack(h_list, axis=0)
    h_others = h[1:]
    mags = np.sqrt(np.mean(h_others ** 2, axis=-1, keepdims=True))
    finfo_min = np.finfo(np.float32).tiny
    h_others_scaled = h_others * (target_mag / np.maximum(mags, finfo_min))
    return np.concatenate([h[0:1], h_others_scaled], axis=0)


def np_per_layer_projection(
    embeds, proj_w, norm_gamma, per_layer_embeds, B, T, H, L, H_pl, eps,
):
    """NumPy per-layer model projection + norm."""
    proj = (embeds.reshape(B * T, H) @ proj_w).reshape(B, T, L * H_pl)
    proj = proj * (H ** -0.5)
    proj = proj.reshape(B, T, L, H_pl)
    proj = np_rmsnorm(proj, norm_gamma, eps)
    return (proj + per_layer_embeds) * (2.0 ** -0.5)


def np_final_unembed(h, unembed_ws, fn_gamma, B, T, H, n, eps):
    """NumPy final unembed."""
    target_mag = np.sqrt(np.mean(h[0] ** 2, axis=-1, keepdims=True))
    h_others = []
    for i, w in enumerate(unembed_ws):
        h_others.append((h[i + 1].reshape(B * T, H) @ w).reshape(B, T, H))
    h_others = np.stack(h_others, axis=0)
    mags = np.sqrt(np.mean(h_others ** 2, axis=-1, keepdims=True))
    finfo_min = np.finfo(np.float32).tiny
    h_others_scaled = h_others * (target_mag / np.maximum(mags, finfo_min))
    h_combined = np.concatenate([h[0:1], h_others_scaled], axis=0)
    h_mean = np.mean(h_combined, axis=0)
    return np_rmsnorm(h_mean, fn_gamma, eps)


def np_chunked_logits(hidden, emb_table, V, chunk_size, softcap=None):
    """NumPy chunked logit projection."""
    chunks = []
    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        chunk = emb_table[start:end]
        logits = hidden @ chunk.T
        if softcap is not None:
            logits = softcap * np.tanh(logits / softcap)
        chunks.append(logits)
    return np.concatenate(chunks, axis=-1)


# ---------------------------------------------------------------------------
# Weight fixture helpers
# ---------------------------------------------------------------------------


def make_layer_weights(config, seed_base=100):
    """Return a dict of deterministic layer weights."""
    H = config.hidden_size
    n_h = config.num_attention_heads
    n_kv = config.num_key_value_heads
    hd = config.head_dim
    I = config.intermediate_size
    rank = config.laurel_rank
    H_pl = config.hidden_size_per_layer_input
    n = config.altup_num_inputs
    s = seed_base
    return {
        "q_w": det_array((H, n_h * hd), s),
        "k_w": det_array((H, n_kv * hd), s + 1),
        "v_w": det_array((H, n_kv * hd), s + 2),
        "o_w": det_array((n_h * hd, H), s + 3),
        "q_ng": det_array((hd,), s + 4),
        "k_ng": det_array((hd,), s + 5),
        "gate_w": det_array((H, I), s + 6),
        "up_w": det_array((H, I), s + 7),
        "down_w": det_array((I, H), s + 8),
        "ll_w": det_array((H, rank), s + 9),
        "lr_w": det_array((rank, H), s + 10),
        "ln_g": det_array((H,), s + 11),
        "pred_w": det_array((n, n * n), s + 12),
        "corr_w": det_array((n, n), s + 13),
        "mr_w": det_array((H, n), s + 14),
        "rn_g": det_array((H,), s + 15),
        "cos_scale": det_array((H,), s + 16),
        "iln_g": det_array((H,), s + 17),
        "paln_g": det_array((H,), s + 18),
        "pfln_g": det_array((H,), s + 19),
        "pfln2_g": det_array((H,), s + 20),
        "plin_g": det_array((H,), s + 21),
        "pli_gw": det_array((H, H_pl), s + 22),
        "pli_pw": det_array((H_pl, H), s + 23),
    }


def layer_weight_args(w):
    """Unpack weight dict into positional order for compiled decoder layer."""
    return (
        w["q_w"], w["k_w"], w["v_w"], w["o_w"], w["q_ng"], w["k_ng"],
        w["gate_w"], w["up_w"], w["down_w"],
        w["ll_w"], w["lr_w"], w["ln_g"],
        w["pred_w"], w["corr_w"], w["mr_w"], w["rn_g"], w["cos_scale"],
        w["iln_g"], w["paln_g"], w["pfln_g"], w["pfln2_g"], w["plin_g"],
        w["pli_gw"], w["pli_pw"],
    )


# ---------------------------------------------------------------------------
# Tests: RMSNorm primitives
# ---------------------------------------------------------------------------


class TestRMSNorm:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_learned_rmsnorm(self, backend):
        cfg = SMALL_CONFIG
        H = cfg.hidden_size
        x_val = det_array((2, 3, H), seed=10)
        g_val = det_array((H,), seed=11)

        x_s = pt.tensor("x", shape=(2, 3, H), dtype="float32")
        g_s = pt.tensor("g", shape=(H,), dtype="float32")
        out = rmsnorm_symbolic(x_s, g_s, cfg.rms_norm_eps)
        fn = pytensor.function([x_s, g_s], out, mode=_get_mode(backend))

        result = fn(x_val, g_val)
        expected = np_rmsnorm(x_val, g_val, cfg.rms_norm_eps)
        np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_rms_no_scale(self, backend):
        x_val = det_array((2, 5, 7), seed=20)
        x_s = pt.tensor("x", shape=(2, 5, 7), dtype="float32")
        out = rms_norm_no_scale(x_s, 1e-5)
        fn = pytensor.function([x_s], out, mode=_get_mode(backend))

        result = fn(x_val)
        expected = np_rms_no_scale(x_val, 1e-5)
        np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)

    def test_rmsnorm_identity_gamma(self):
        """With gamma=1, output should be x / rms(x)."""
        H = 8
        x_val = det_array((1, 1, H), seed=30)
        g_val = np.ones(H, dtype=np.float32)

        x_s = pt.tensor("x", shape=(1, 1, H), dtype="float32")
        g_s = pt.tensor("g", shape=(H,), dtype="float32")
        out = rmsnorm_symbolic(x_s, g_s, 1e-5)
        fn = pytensor.function([x_s, g_s], out, mode="FAST_COMPILE")

        result = fn(x_val, g_val)
        rms = np.sqrt(np.mean(x_val ** 2, axis=-1, keepdims=True))
        expected = x_val / rms
        np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: GELU primitives
# ---------------------------------------------------------------------------


class TestGELU:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_gelu_approx(self, backend):
        x_val = det_array((3, 7), seed=40)
        x_s = pt.tensor("x", shape=(3, 7), dtype="float32")
        out = gelu_approx_symbolic(x_s)
        fn = pytensor.function([x_s], out, mode=_get_mode(backend))

        result = fn(x_val)
        expected = np_gelu_approx(x_val)
        np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_sparse_gelu(self, backend):
        x_val = det_array((2, 5, 11), seed=50)
        std_mult = 0.5  # arbitrary test value

        x_s = pt.tensor("x", shape=(2, 5, 11), dtype="float32")
        out = sparse_gelu_symbolic(x_s, std_mult)
        fn = pytensor.function([x_s], out, mode=_get_mode(backend))

        result = fn(x_val)
        expected = np_sparse_gelu(x_val, std_mult)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_gelu_negative_inputs(self):
        """GELU should handle negative inputs correctly."""
        x_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        x_s = pt.tensor("x", shape=(5,), dtype="float32")
        out = gelu_approx_symbolic(x_s)
        fn = pytensor.function([x_s], out, mode="FAST_COMPILE")
        result = fn(x_val)
        expected = np_gelu_approx(x_val)
        np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: RoPE
# ---------------------------------------------------------------------------


class TestRoPE:
    def test_build_rope_table_shape(self):
        cos, sin = build_rope_table(10000.0, 12, 8)
        assert cos.shape == (8, 6)
        assert sin.shape == (8, 6)
        assert cos.dtype == np.float32
        assert sin.dtype == np.float32

    def test_build_rope_table_position_zero(self):
        cos, sin = build_rope_table(10000.0, 12, 4)
        np.testing.assert_allclose(cos[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(sin[0], 0.0, atol=1e-6)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_apply_rope_symbolic(self, backend):
        hd = SMALL_CONFIG.head_dim
        T = 5
        x_val = det_array((1, 2, T, hd), seed=60)
        cos_val, sin_val = build_rope_table(10000.0, hd, T)
        cos_exp = cos_val.reshape(1, 1, T, hd // 2)
        sin_exp = sin_val.reshape(1, 1, T, hd // 2)

        x_s = pt.tensor("x", shape=(1, 2, T, hd), dtype="float32")
        c_s = pt.tensor("cos", shape=(1, 1, T, hd // 2), dtype="float32")
        s_s = pt.tensor("sin", shape=(1, 1, T, hd // 2), dtype="float32")
        out = apply_rope_symbolic(x_s, c_s, s_s, hd)
        fn = pytensor.function([x_s, c_s, s_s], out, mode=_get_mode(backend))

        result = fn(x_val, cos_exp, sin_exp)
        expected = np_apply_rope(x_val, cos_exp, sin_exp, hd)
        np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Masks
# ---------------------------------------------------------------------------


class TestMasks:
    def test_causal_mask_shape(self):
        m = causal_mask(5)
        assert m.shape == (5, 5)
        assert m.dtype == np.float32

    def test_causal_mask_values(self):
        m = causal_mask(4)
        assert m[0, 0] == 0.0
        assert m[0, 1] == -np.inf
        assert m[3, 0] == 0.0
        assert m[3, 3] == 0.0

    def test_causal_mask_rejects_empty_sequence(self):
        with pytest.raises(ValueError, match="seq_len"):
            causal_mask(0)

    def test_sliding_mask_shape(self):
        m = sliding_window_mask(6, 3)
        assert m.shape == (6, 6)
        assert m.dtype == np.float32

    def test_sliding_mask_values(self):
        m = sliding_window_mask(5, 3)
        # Position 0: only sees 0
        assert m[0, 0] == 0.0
        assert m[0, 1] == -np.inf
        # Position 2: sees 0, 1, 2
        assert m[2, 0] == 0.0
        assert m[2, 1] == 0.0
        assert m[2, 2] == 0.0
        assert m[2, 3] == -np.inf
        # Position 4: sees 2, 3, 4
        assert m[4, 1] == -np.inf
        assert m[4, 2] == 0.0
        assert m[4, 3] == 0.0
        assert m[4, 4] == 0.0

    def test_sliding_mask_rejects_zero_window(self):
        with pytest.raises(ValueError, match="window"):
            sliding_window_mask(3, 0)


# ---------------------------------------------------------------------------
# Tests: GQA Attention
# ---------------------------------------------------------------------------


class TestGQAAttention:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_gqa_attention(self, backend):
        cfg = SMALL_CONFIG
        B, T = 1, 4
        n_h, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

        q_val = det_array((B, n_h, T, hd), seed=70)
        k_val = det_array((B, n_kv, T, hd), seed=71)
        v_val = det_array((B, n_kv, T, hd), seed=72)
        mask_val = causal_mask(T).reshape(1, 1, T, T)

        q_s = pt.tensor("q", shape=(B, n_h, T, hd), dtype="float32")
        k_s = pt.tensor("k", shape=(B, n_kv, T, hd), dtype="float32")
        v_s = pt.tensor("v", shape=(B, n_kv, T, hd), dtype="float32")
        m_s = pt.tensor("mask", shape=(1, 1, T, T), dtype="float32")
        out = gqa_attention(q_s, k_s, v_s, m_s, n_h, n_kv, hd, B, T, 1.0)
        fn = pytensor.function([q_s, k_s, v_s, m_s], out, mode=_get_mode(backend))

        result = fn(q_val, k_val, v_val, mask_val)
        expected = np_gqa_attention(q_val, k_val, v_val, mask_val, n_h, n_kv, 1.0)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: LAuReL
# ---------------------------------------------------------------------------


class TestLAuReL:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_laurel(self, backend):
        cfg = SMALL_CONFIG
        B, T, H = 1, 3, cfg.hidden_size
        rank = cfg.laurel_rank
        eps = cfg.rms_norm_eps

        x_val = det_array((B, T, H), seed=80)
        wl = det_array((H, rank), seed=81)
        wr = det_array((rank, H), seed=82)
        ng = det_array((H,), seed=83)

        x_s = pt.tensor("x", shape=(B, T, H), dtype="float32")
        wl_s = pt.tensor("wl", shape=(H, rank), dtype="float32")
        wr_s = pt.tensor("wr", shape=(rank, H), dtype="float32")
        ng_s = pt.tensor("ng", shape=(H,), dtype="float32")
        out = laurel_symbolic(x_s, wl_s, wr_s, ng_s, B, T, H, rank, eps)
        fn = pytensor.function(
            [x_s, wl_s, wr_s, ng_s], out, mode=_get_mode(backend)
        )

        result = fn(x_val, wl, wr, ng)
        expected = np_laurel(x_val, wl, wr, ng, B, T, H, rank, eps)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: MLP
# ---------------------------------------------------------------------------


class TestMLP:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_dense_mlp(self, backend):
        cfg = SMALL_CONFIG
        B, T, H, I = 1, 3, cfg.hidden_size, cfg.intermediate_size

        x_val = det_array((B, T, H), seed=90)
        gw = det_array((H, I), seed=91)
        uw = det_array((H, I), seed=92)
        dw = det_array((I, H), seed=93)

        x_s = pt.tensor("x", shape=(B, T, H), dtype="float32")
        gw_s = pt.tensor("gw", shape=(H, I), dtype="float32")
        uw_s = pt.tensor("uw", shape=(H, I), dtype="float32")
        dw_s = pt.tensor("dw", shape=(I, H), dtype="float32")
        out = mlp_symbolic(x_s, gw_s, uw_s, dw_s, B, T, H, I, None)
        fn = pytensor.function(
            [x_s, gw_s, uw_s, dw_s], out, mode=_get_mode(backend)
        )

        result = fn(x_val, gw, uw, dw)
        expected = np_mlp(x_val, gw, uw, dw, B, T, H, I, None)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_sparse_mlp(self, backend):
        cfg = SMALL_CONFIG
        B, T, H, I = 1, 3, cfg.hidden_size, cfg.intermediate_size
        std_mult = 0.5

        x_val = det_array((B, T, H), seed=95)
        gw = det_array((H, I), seed=96)
        uw = det_array((H, I), seed=97)
        dw = det_array((I, H), seed=98)

        x_s = pt.tensor("x", shape=(B, T, H), dtype="float32")
        gw_s = pt.tensor("gw", shape=(H, I), dtype="float32")
        uw_s = pt.tensor("uw", shape=(H, I), dtype="float32")
        dw_s = pt.tensor("dw", shape=(I, H), dtype="float32")
        out = mlp_symbolic(x_s, gw_s, uw_s, dw_s, B, T, H, I, std_mult)
        fn = pytensor.function(
            [x_s, gw_s, uw_s, dw_s], out, mode=_get_mode(backend)
        )

        result = fn(x_val, gw, uw, dw)
        expected = np_mlp(x_val, gw, uw, dw, B, T, H, I, std_mult)
        np.testing.assert_allclose(result, expected, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Tests: AltUp
# ---------------------------------------------------------------------------


class TestAltUp:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_predict(self, backend):
        cfg = SMALL_CONFIG
        B, T, H = 1, 3, cfg.hidden_size
        n = cfg.altup_num_inputs

        x_val = det_array((n, B, T, H), seed=100)
        pw = det_array((n, n * n), seed=101)
        rng = det_array((H,), seed=102)
        rw = det_array((H, n), seed=103)

        x_s = pt.tensor("x", shape=(n, B, T, H), dtype="float32")
        pw_s = pt.tensor("pw", shape=(n, n * n), dtype="float32")
        rng_s = pt.tensor("rng", shape=(H,), dtype="float32")
        rw_s = pt.tensor("rw", shape=(H, n), dtype="float32")
        out = altup_predict_symbolic(
            x_s, pw_s, rng_s, rw_s, B, T, H, n,
            cfg.rms_norm_eps, cfg.altup_coef_clip, cfg.altup_active_idx,
        )
        fn = pytensor.function(
            [x_s, pw_s, rng_s, rw_s], out, mode=_get_mode(backend)
        )

        result = fn(x_val, pw, rng, rw)
        expected = np_altup_predict(
            x_val, pw, rng, rw, B, T, H, n,
            cfg.rms_norm_eps, cfg.altup_coef_clip, cfg.altup_active_idx,
        )
        np.testing.assert_allclose(result, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_correct(self, backend):
        cfg = SMALL_CONFIG
        B, T, H = 1, 3, cfg.hidden_size
        n = cfg.altup_num_inputs

        preds_val = det_array((n, B, T, H), seed=110)
        act_val = det_array((B, T, H), seed=111)
        cw = det_array((n, n), seed=112)
        rng = det_array((H,), seed=113)
        rw = det_array((H, n), seed=114)

        p_s = pt.tensor("p", shape=(n, B, T, H), dtype="float32")
        a_s = pt.tensor("a", shape=(B, T, H), dtype="float32")
        cw_s = pt.tensor("cw", shape=(n, n), dtype="float32")
        rng_s = pt.tensor("rng", shape=(H,), dtype="float32")
        rw_s = pt.tensor("rw", shape=(H, n), dtype="float32")
        out = altup_correct_symbolic(
            p_s, a_s, cw_s, rng_s, rw_s, B, T, H, n,
            cfg.rms_norm_eps, cfg.altup_coef_clip, cfg.altup_active_idx,
        )
        fn = pytensor.function(
            [p_s, a_s, cw_s, rng_s, rw_s], out, mode=_get_mode(backend)
        )

        result = fn(preds_val, act_val, cw, rng, rw)
        expected = np_altup_correct(
            preds_val, act_val, cw, rng, rw, B, T, H, n,
            cfg.rms_norm_eps, cfg.altup_coef_clip, cfg.altup_active_idx,
        )
        np.testing.assert_allclose(result, expected, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Tests: Decoder Layer (all four kinds)
# ---------------------------------------------------------------------------


class TestDecoderLayer:
    """Parametric tests for all four layer kinds:
    1. Full attention, no sparsity
    2. Full attention, with sparsity
    3. Sliding attention, no sparsity
    4. Sliding attention, with sparsity
    """

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize(
        "is_sliding,has_sparsity",
        [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ],
        ids=[
            "full_dense",
            "full_sparse",
            "sliding_dense",
            "sliding_sparse",
        ],
    )
    def test_decoder_layer(self, backend, is_sliding, has_sparsity):
        cfg = SMALL_CONFIG
        B, T = 1, 4
        n = cfg.altup_num_inputs
        H = cfg.hidden_size
        H_pl = cfg.hidden_size_per_layer_input
        hd = cfg.head_dim

        fn = compile_decoder_layer(
            cfg, B, T, has_sparsity=has_sparsity, backend=backend,
        )

        w = make_layer_weights(cfg, seed_base=200)
        hidden = det_array((n, B, T, H), seed=210)
        pli = det_array((B, T, H_pl), seed=211)

        if is_sliding:
            mask = sliding_window_mask(T, cfg.sliding_window)
        else:
            mask = causal_mask(T)

        cos, sin = build_rope_table(cfg.rope_theta, hd, T)

        result = fn(hidden, mask, pli, cos, sin, *layer_weight_args(w))

        # Compute std_multiplier for sparse case
        std_mult = None
        if has_sparsity and cfg.activation_sparsity > 0.0:
            from cetagostini.utils.pytensor.gemma3n_pytensor import _erfinv
            std_mult = float(
                math.sqrt(2.0) * _erfinv(2.0 * cfg.activation_sparsity - 1.0)
            )

        mask_4d = mask.reshape(1, 1, T, T)
        expected = np_decoder_layer(
            hidden, mask_4d, pli, cos, sin,
            w["q_w"], w["k_w"], w["v_w"], w["o_w"], w["q_ng"], w["k_ng"],
            w["gate_w"], w["up_w"], w["down_w"],
            w["ll_w"], w["lr_w"], w["ln_g"],
            w["pred_w"], w["corr_w"], w["mr_w"], w["rn_g"], w["cos_scale"],
            w["iln_g"], w["paln_g"], w["pfln_g"], w["pfln2_g"], w["plin_g"],
            w["pli_gw"], w["pli_pw"],
            cfg, B, T, std_mult,
        )

        np.testing.assert_allclose(result, expected, atol=1e-2, rtol=1e-2)

    def test_decoder_layer_output_shape(self):
        cfg = SMALL_CONFIG
        B, T = 1, 4
        fn = compile_decoder_layer(cfg, B, T, backend="c")
        n, H = cfg.altup_num_inputs, cfg.hidden_size
        assert fn.maker.fgraph.outputs[0].type.shape == (n, B, T, H)

    def test_decoder_layer_output_finite(self):
        """All outputs must be finite (no NaN or Inf)."""
        cfg = SMALL_CONFIG
        B, T = 1, 4
        fn = compile_decoder_layer(cfg, B, T, backend="c")
        n, H = cfg.altup_num_inputs, cfg.hidden_size
        H_pl = cfg.hidden_size_per_layer_input
        hd = cfg.head_dim

        w = make_layer_weights(cfg, seed_base=300)
        hidden = det_array((n, B, T, H), seed=310) * 0.1
        pli = det_array((B, T, H_pl), seed=311) * 0.1
        mask = causal_mask(T)
        cos, sin = build_rope_table(cfg.rope_theta, hd, T)

        result = fn(hidden, mask, pli, cos, sin, *layer_weight_args(w))
        assert np.all(np.isfinite(result)), "Output contains non-finite values"

    def test_decoder_layer_output_contiguous(self):
        """Output must be C-contiguous."""
        cfg = SMALL_CONFIG
        B, T = 1, 4
        fn = compile_decoder_layer(cfg, B, T, backend="c")
        n, H = cfg.altup_num_inputs, cfg.hidden_size
        H_pl = cfg.hidden_size_per_layer_input
        hd = cfg.head_dim

        w = make_layer_weights(cfg, seed_base=400)
        hidden = det_array((n, B, T, H), seed=410)
        pli = det_array((B, T, H_pl), seed=411)
        mask = causal_mask(T)
        cos, sin = build_rope_table(cfg.rope_theta, hd, T)

        result = fn(hidden, mask, pli, cos, sin, *layer_weight_args(w))
        assert result.flags["C_CONTIGUOUS"], "Output is not C-contiguous"


# ---------------------------------------------------------------------------
# Tests: Shared-KV equivalence
# ---------------------------------------------------------------------------


class TestSharedKV:
    """Tests for shared K/V input mode."""

    def test_shared_kv_produces_same_output(self):
        """Providing precomputed K/V should yield identical attention output."""
        cfg = SMALL_CONFIG
        B, T = 1, 4
        H = cfg.hidden_size
        n_h = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        eps = cfg.rms_norm_eps

        x_val = det_array((B, T, H), seed=1200)
        q_w = det_array((H, n_h * hd), seed=1201)
        k_w = det_array((H, n_kv * hd), seed=1202)
        v_w = det_array((H, n_kv * hd), seed=1203)
        o_w = det_array((n_h * hd, H), seed=1204)
        q_ng = det_array((hd,), seed=1205)
        k_ng = det_array((hd,), seed=1206)
        cos_val, sin_val = build_rope_table(cfg.rope_theta, hd, T)
        mask_val = causal_mask(T).reshape(1, 1, T, T)

        # Compute K/V internally (normal mode)
        x_s = pt.tensor("x", shape=(B, T, H), dtype="float32")
        q_w_s = pt.tensor("q_w", shape=(H, n_h * hd), dtype="float32")
        k_w_s = pt.tensor("k_w", shape=(H, n_kv * hd), dtype="float32")
        v_w_s = pt.tensor("v_w", shape=(H, n_kv * hd), dtype="float32")
        o_w_s = pt.tensor("o_w", shape=(n_h * hd, H), dtype="float32")
        q_ng_s = pt.tensor("q_ng", shape=(hd,), dtype="float32")
        k_ng_s = pt.tensor("k_ng", shape=(hd,), dtype="float32")
        cos_s = pt.tensor("cos", shape=(T, hd // 2), dtype="float32")
        sin_s = pt.tensor("sin", shape=(T, hd // 2), dtype="float32")
        mask_s = pt.tensor("mask", shape=(1, 1, T, T), dtype="float32")

        out_normal = attention_block_symbolic(
            x_s, q_w_s, k_w_s, v_w_s, o_w_s, q_ng_s, k_ng_s,
            cos_s, sin_s, mask_s, B, T, H, n_h, n_kv, hd, eps,
        )
        fn_normal = pytensor.function(
            [x_s, q_w_s, k_w_s, v_w_s, o_w_s, q_ng_s, k_ng_s,
             cos_s, sin_s, mask_s],
            out_normal, mode="FAST_COMPILE",
        )
        result_normal = fn_normal(
            x_val, q_w, k_w, v_w, o_w, q_ng, k_ng, cos_val, sin_val, mask_val,
        )

        # Compute K/V and pass as shared
        # First compute what K/V would be
        k_proj = (x_val.reshape(B * T, H) @ k_w).reshape(B, T, n_kv, hd).swapaxes(1, 2)
        v_proj = (x_val.reshape(B * T, H) @ v_w).reshape(B, T, n_kv, hd).swapaxes(1, 2)
        k_normed = np_rmsnorm(k_proj, k_ng, eps)
        v_normed = np_rms_no_scale(v_proj, eps)
        cos_exp = cos_val.reshape(1, 1, T, hd // 2)
        sin_exp = sin_val.reshape(1, 1, T, hd // 2)
        k_rotated = np_apply_rope(k_normed, cos_exp, sin_exp, hd)

        shared_k_s = pt.tensor("shared_k", shape=(B, n_kv, T, hd), dtype="float32")
        shared_v_s = pt.tensor("shared_v", shape=(B, n_kv, T, hd), dtype="float32")

        out_shared = attention_block_symbolic(
            x_s, q_w_s, k_w_s, v_w_s, o_w_s, q_ng_s, k_ng_s,
            cos_s, sin_s, mask_s, B, T, H, n_h, n_kv, hd, eps,
            shared_k=shared_k_s, shared_v=shared_v_s,
        )
        fn_shared = pytensor.function(
            [x_s, q_w_s, k_w_s, v_w_s, o_w_s, q_ng_s, k_ng_s,
             cos_s, sin_s, mask_s, shared_k_s, shared_v_s],
            out_shared, mode="FAST_COMPILE",
            on_unused_input="ignore",
        )
        result_shared = fn_shared(
            x_val, q_w, k_w, v_w, o_w, q_ng, k_ng, cos_val, sin_val, mask_val,
            k_rotated, v_normed,
        )

        np.testing.assert_allclose(
            result_normal, result_shared, atol=1e-5, rtol=1e-5,
        )

    def test_return_kv_returns_tuple(self):
        """return_kv=True should return (output, k, v) tuple."""
        cfg = SMALL_CONFIG
        B, T = 1, 4
        H = cfg.hidden_size
        n_h = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        eps = cfg.rms_norm_eps

        x_s = pt.tensor("x", shape=(B, T, H), dtype="float32")
        q_w_s = pt.tensor("q_w", shape=(H, n_h * hd), dtype="float32")
        k_w_s = pt.tensor("k_w", shape=(H, n_kv * hd), dtype="float32")
        v_w_s = pt.tensor("v_w", shape=(H, n_kv * hd), dtype="float32")
        o_w_s = pt.tensor("o_w", shape=(n_h * hd, H), dtype="float32")
        q_ng_s = pt.tensor("q_ng", shape=(hd,), dtype="float32")
        k_ng_s = pt.tensor("k_ng", shape=(hd,), dtype="float32")
        cos_s = pt.tensor("cos", shape=(T, hd // 2), dtype="float32")
        sin_s = pt.tensor("sin", shape=(T, hd // 2), dtype="float32")
        mask_s = pt.tensor("mask", shape=(1, 1, T, T), dtype="float32")

        out = attention_block_symbolic(
            x_s, q_w_s, k_w_s, v_w_s, o_w_s, q_ng_s, k_ng_s,
            cos_s, sin_s, mask_s, B, T, H, n_h, n_kv, hd, eps,
            return_kv=True,
        )
        assert isinstance(out, tuple)
        assert len(out) == 3

        fn = pytensor.function(
            [x_s, q_w_s, k_w_s, v_w_s, o_w_s, q_ng_s, k_ng_s,
             cos_s, sin_s, mask_s],
            out, mode="FAST_COMPILE",
        )

        x_val = det_array((B, T, H), seed=1210)
        q_w = det_array((H, n_h * hd), seed=1211)
        k_w = det_array((H, n_kv * hd), seed=1212)
        v_w = det_array((H, n_kv * hd), seed=1213)
        o_w = det_array((n_h * hd, H), seed=1214)
        q_ng = det_array((hd,), seed=1215)
        k_ng = det_array((hd,), seed=1216)
        cos_val, sin_val = build_rope_table(cfg.rope_theta, hd, T)
        mask_val = causal_mask(T).reshape(1, 1, T, T)

        result = fn(x_val, q_w, k_w, v_w, o_w, q_ng, k_ng,
                    cos_val, sin_val, mask_val)
        assert len(result) == 3
        o_out, k_out, v_out = result
        assert o_out.shape == (B, T, H)
        assert k_out.shape == (B, n_kv, T, hd)
        assert v_out.shape == (B, n_kv, T, hd)

    def test_shared_kv_mismatch_raises(self):
        """Providing only shared_k without shared_v should raise."""
        cfg = SMALL_CONFIG
        B, T = 1, 4
        H = cfg.hidden_size
        n_h = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        eps = cfg.rms_norm_eps

        x_s = pt.tensor("x", shape=(B, T, H), dtype="float32")
        q_w_s = pt.tensor("q_w", shape=(H, n_h * hd), dtype="float32")
        k_w_s = pt.tensor("k_w", shape=(H, n_kv * hd), dtype="float32")
        v_w_s = pt.tensor("v_w", shape=(H, n_kv * hd), dtype="float32")
        o_w_s = pt.tensor("o_w", shape=(n_h * hd, H), dtype="float32")
        q_ng_s = pt.tensor("q_ng", shape=(hd,), dtype="float32")
        k_ng_s = pt.tensor("k_ng", shape=(hd,), dtype="float32")
        cos_s = pt.tensor("cos", shape=(T, hd // 2), dtype="float32")
        sin_s = pt.tensor("sin", shape=(T, hd // 2), dtype="float32")
        mask_s = pt.tensor("mask", shape=(1, 1, T, T), dtype="float32")
        shared_k_s = pt.tensor("shared_k", shape=(B, n_kv, T, hd), dtype="float32")

        with pytest.raises(ValueError, match="shared_k and shared_v"):
            attention_block_symbolic(
                x_s, q_w_s, k_w_s, v_w_s, o_w_s, q_ng_s, k_ng_s,
                cos_s, sin_s, mask_s, B, T, H, n_h, n_kv, hd, eps,
                shared_k=shared_k_s, shared_v=None,
            )

    def test_compile_decoder_kv_mode_return(self):
        """compile_decoder_layer with kv_mode='return' returns 3 outputs."""
        cfg = SMALL_CONFIG
        B, T = 1, 4
        fn = compile_decoder_layer(cfg, B, T, backend="c", kv_mode="return")
        assert len(fn.maker.fgraph.outputs) == 3

    def test_compile_decoder_kv_mode_shared(self):
        """compile_decoder_layer with kv_mode='shared' accepts shared inputs."""
        cfg = SMALL_CONFIG
        B, T = 1, 4
        fn = compile_decoder_layer(cfg, B, T, backend="c", kv_mode="shared")
        # Should have 2 extra inputs (shared_k, shared_v)
        n_inputs = len(fn.maker.fgraph.inputs)
        # 5 base + 2 shared + 24 weights = 31
        assert n_inputs == 31

    def test_compile_decoder_kv_mode_invalid(self):
        """Invalid kv_mode should raise ValueError."""
        cfg = SMALL_CONFIG
        with pytest.raises(ValueError, match="kv_mode"):
            compile_decoder_layer(cfg, 1, 4, backend="c", kv_mode="invalid")


# ---------------------------------------------------------------------------
# Tests: Initial projections
# ---------------------------------------------------------------------------


class TestInitialProjections:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_initial_projections(self, backend):
        cfg = SMALL_CONFIG
        B, T, H = 1, 3, cfg.hidden_size
        n = cfg.altup_num_inputs

        fn = compile_initial_projections(cfg, B, T, backend=backend)

        h0 = det_array((B, T, H), seed=500)
        proj_ws = [det_array((H, H), seed=510 + i) for i in range(n - 1)]

        result = fn(h0, *proj_ws)
        expected = np_initial_projections(h0, proj_ws, B, T, H, n)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_initial_projections_shape(self):
        cfg = SMALL_CONFIG
        B, T = 1, 3
        fn = compile_initial_projections(cfg, B, T, backend="c")
        n, H = cfg.altup_num_inputs, cfg.hidden_size
        assert fn.maker.fgraph.outputs[0].type.shape == (n, B, T, H)

    def test_zero_projected_stream_remains_finite(self):
        cfg = SMALL_CONFIG
        B, T, H = 1, 2, cfg.hidden_size
        n = cfg.altup_num_inputs
        fn = compile_initial_projections(cfg, B, T, backend="c")
        h0 = det_array((B, T, H), seed=515)
        zero_weights = [np.zeros((H, H), dtype=np.float32) for _ in range(n - 1)]

        result = fn(h0, *zero_weights)

        assert np.all(np.isfinite(result))
        np.testing.assert_array_equal(result[1:], 0.0)


# ---------------------------------------------------------------------------
# Tests: Per-layer projection
# ---------------------------------------------------------------------------


class TestPerLayerProjection:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_per_layer_projection(self, backend):
        cfg = SMALL_CONFIG
        B, T = 1, 3
        H = cfg.hidden_size
        L = cfg.num_hidden_layers
        H_pl = cfg.hidden_size_per_layer_input

        fn = compile_per_layer_projection(cfg, B, T, backend=backend)

        embeds = det_array((B, T, H), seed=600)
        pw = det_array((H, L * H_pl), seed=601)
        ng = det_array((H_pl,), seed=602)
        ple = det_array((B, T, L, H_pl), seed=603)

        result = fn(embeds, pw, ng, ple)
        expected = np_per_layer_projection(
            embeds, pw, ng, ple, B, T, H, L, H_pl, cfg.rms_norm_eps,
        )
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: Final unembed
# ---------------------------------------------------------------------------


class TestFinalUnembed:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_final_unembed(self, backend):
        cfg = SMALL_CONFIG
        B, T, H = 1, 3, cfg.hidden_size
        n = cfg.altup_num_inputs

        fn = compile_final_unembed(cfg, B, T, backend=backend)

        h = det_array((n, B, T, H), seed=700)
        unembed_ws = [det_array((H, H), seed=710 + i) for i in range(n - 1)]
        fn_g = det_array((H,), seed=720)

        result = fn(h, *unembed_ws, fn_g)
        expected = np_final_unembed(h, unembed_ws, fn_g, B, T, H, n, cfg.rms_norm_eps)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_zero_projected_streams_remain_finite(self):
        cfg = SMALL_CONFIG
        B, T, H = 1, 2, cfg.hidden_size
        n = cfg.altup_num_inputs
        fn = compile_final_unembed(cfg, B, T, backend="c")
        h = det_array((n, B, T, H), seed=725)
        zero_weights = [np.zeros((H, H), dtype=np.float32) for _ in range(n - 1)]
        gamma = np.ones(H, dtype=np.float32)

        result = fn(h, *zero_weights, gamma)

        assert np.all(np.isfinite(result))

    def test_single_stream(self):
        from dataclasses import replace

        cfg = replace(SMALL_CONFIG, altup_num_inputs=1)
        B, T, H = 1, 2, cfg.hidden_size
        fn = compile_final_unembed(cfg, B, T, backend="c")
        h = det_array((1, B, T, H), seed=726)
        gamma = np.ones(H, dtype=np.float32)

        result = fn(h, gamma)
        expected = np_rmsnorm(h[0], gamma, cfg.rms_norm_eps)

        np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Chunked logit projection
# ---------------------------------------------------------------------------


class TestLogitProjection:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_chunked_logits_no_softcap(self, backend):
        V, H, B = 64, 24, 1
        fn = compile_logit_projection(V, H, B, chunk_size=16, backend=backend)

        hidden = det_array((B, H), seed=800)
        emb = det_array((V, H), seed=801)

        result = fn(hidden, emb)
        expected = np_chunked_logits(hidden, emb, V, 16, None)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_chunked_logits_with_softcap(self, backend):
        V, H, B = 64, 24, 1
        softcap = 30.0
        fn = compile_logit_projection(
            V, H, B, chunk_size=20, softcap=softcap, backend=backend,
        )

        hidden = det_array((B, H), seed=810)
        emb = det_array((V, H), seed=811)

        result = fn(hidden, emb)
        expected = np_chunked_logits(hidden, emb, V, 20, softcap)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_chunked_logits_uneven_chunks(self):
        """Vocab size not divisible by chunk size."""
        V, H, B = 65, 24, 1
        fn = compile_logit_projection(V, H, B, chunk_size=16, backend="c")

        hidden = det_array((B, H), seed=820)
        emb = det_array((V, H), seed=821)

        result = fn(hidden, emb)
        assert result.shape == (B, V)
        expected = np_chunked_logits(hidden, emb, V, 16, None)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: Float32 audit
# ---------------------------------------------------------------------------


class TestFloat32Audit:
    def test_decoder_layer_float32(self):
        cfg = SMALL_CONFIG
        fn = compile_decoder_layer(cfg, 1, 4, backend="c")
        assert audit_float32(fn)

    def test_initial_projections_float32(self):
        cfg = SMALL_CONFIG
        fn = compile_initial_projections(cfg, 1, 3, backend="c")
        assert audit_float32(fn)

    def test_per_layer_projection_float32(self):
        cfg = SMALL_CONFIG
        fn = compile_per_layer_projection(cfg, 1, 3, backend="c")
        assert audit_float32(fn)

    def test_final_unembed_float32(self):
        cfg = SMALL_CONFIG
        fn = compile_final_unembed(cfg, 1, 3, backend="c")
        assert audit_float32(fn)

    def test_logit_projection_float32(self):
        fn = compile_logit_projection(64, 24, 1, 16, backend="c")
        assert audit_float32(fn)

    def test_numba_decoder_layer_float32(self):
        cfg = SMALL_CONFIG
        fn = compile_decoder_layer(cfg, 1, 4, backend="numba")
        assert audit_float32(fn)


# ---------------------------------------------------------------------------
# Tests: Backend parity (C vs Numba)
# ---------------------------------------------------------------------------


class TestBackendParity:
    @pytest.mark.parametrize("has_sparsity", [False, True])
    def test_decoder_layer_parity(self, has_sparsity):
        from dataclasses import replace

        cfg = replace(
            SMALL_CONFIG,
            activation_sparsity=0.6 if has_sparsity else 0.0,
        )
        B, T = 1, 4
        fn_c = compile_decoder_layer(
            cfg, B, T, has_sparsity=has_sparsity, backend="c"
        )
        fn_n = compile_decoder_layer(
            cfg, B, T, has_sparsity=has_sparsity, backend="numba"
        )

        n, H = cfg.altup_num_inputs, cfg.hidden_size
        H_pl = cfg.hidden_size_per_layer_input
        hd = cfg.head_dim

        w = make_layer_weights(cfg, seed_base=900)
        hidden = det_array((n, B, T, H), seed=910)
        pli = det_array((B, T, H_pl), seed=911)
        mask = causal_mask(T)
        cos, sin = build_rope_table(cfg.rope_theta, hd, T)

        args = (hidden, mask, pli, cos, sin, *layer_weight_args(w))
        r_c = fn_c(*args)
        r_n = fn_n(*args)
        np.testing.assert_allclose(r_c, r_n, atol=1e-4, rtol=1e-4)

    def test_initial_projections_parity(self):
        cfg = SMALL_CONFIG
        B, T, H = 1, 3, cfg.hidden_size
        n = cfg.altup_num_inputs

        fn_c = compile_initial_projections(cfg, B, T, backend="c")
        fn_n = compile_initial_projections(cfg, B, T, backend="numba")

        h0 = det_array((B, T, H), seed=920)
        proj_ws = [det_array((H, H), seed=930 + i) for i in range(n - 1)]

        r_c = fn_c(h0, *proj_ws)
        r_n = fn_n(h0, *proj_ws)
        np.testing.assert_allclose(r_c, r_n, atol=1e-5, rtol=1e-5)

    def test_final_unembed_parity(self):
        cfg = SMALL_CONFIG
        B, T, H = 1, 3, cfg.hidden_size
        n = cfg.altup_num_inputs

        fn_c = compile_final_unembed(cfg, B, T, backend="c")
        fn_n = compile_final_unembed(cfg, B, T, backend="numba")

        h = det_array((n, B, T, H), seed=940)
        unembed_ws = [det_array((H, H), seed=950 + i) for i in range(n - 1)]
        fn_g = det_array((H,), seed=960)

        r_c = fn_c(h, *unembed_ws, fn_g)
        r_n = fn_n(h, *unembed_ws, fn_g)
        np.testing.assert_allclose(r_c, r_n, atol=1e-5, rtol=1e-5)

    def test_logit_projection_parity(self):
        V, H, B = 64, 24, 1
        fn_c = compile_logit_projection(V, H, B, 16, softcap=30.0, backend="c")
        fn_n = compile_logit_projection(V, H, B, 16, softcap=30.0, backend="numba")

        hidden = det_array((B, H), seed=970)
        emb = det_array((V, H), seed=971)

        r_c = fn_c(hidden, emb)
        r_n = fn_n(hidden, emb)
        np.testing.assert_allclose(r_c, r_n, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Import hygiene and input validation
# ---------------------------------------------------------------------------


class TestImportHygiene:
    def test_import_does_not_mutate_floatx(self):
        assert _import_preserved_floatX, (
            f"Importing gemma3n_pytensor mutated floatX from "
            f"{_floatX_before_import!r} to {pytensor.config.floatX!r}"
        )

    def test_reload_does_not_mutate_floatx(self):
        from cetagostini.utils.pytensor import gemma3n_pytensor

        saved = pytensor.config.floatX
        pytensor.config.floatX = "float64"
        try:
            importlib.reload(gemma3n_pytensor)
            assert pytensor.config.floatX == "float64"
        finally:
            pytensor.config.floatX = saved


class TestInputValidation:
    def test_decoder_batch_zero(self):
        with pytest.raises(ValueError, match="batch_size >= 1"):
            compile_decoder_layer(SMALL_CONFIG, 0, 4, backend="c")

    def test_decoder_seq_zero(self):
        with pytest.raises(ValueError, match="seq_len >= 1"):
            compile_decoder_layer(SMALL_CONFIG, 1, 0, backend="c")

    def test_initial_proj_batch_zero(self):
        with pytest.raises(ValueError, match=">= 1"):
            compile_initial_projections(SMALL_CONFIG, 0, 4, backend="c")

    def test_initial_proj_seq_zero(self):
        with pytest.raises(ValueError, match=">= 1"):
            compile_initial_projections(SMALL_CONFIG, 1, 0, backend="c")

    def test_logit_batch_zero(self):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            compile_logit_projection(64, 24, 0, 16, backend="c")

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            from cetagostini.utils.pytensor.gemma3n_pytensor import _get_mode
            _get_mode("unknown")

    @pytest.mark.parametrize("sparsity", [0.0, 1.0])
    def test_sparse_decoder_rejects_boundary_sparsity(self, sparsity):
        from dataclasses import replace

        with pytest.raises(ValueError, match="activation_sparsity"):
            compile_decoder_layer(
                replace(SMALL_CONFIG, activation_sparsity=sparsity),
                1,
                2,
                has_sparsity=True,
                backend="c",
            )

    @pytest.mark.parametrize(
        "vocab_size,hidden_size,chunk_size,match",
        [(0, 24, 16, "vocab_size"), (64, 0, 16, "hidden_size"), (64, 24, 0, "chunk_size")],
    )
    def test_logit_projection_rejects_invalid_dimensions(
        self, vocab_size, hidden_size, chunk_size, match
    ):
        with pytest.raises(ValueError, match=match):
            compile_logit_projection(
                vocab_size, hidden_size, 1, chunk_size, backend="c"
            )


# ---------------------------------------------------------------------------
# Tests: Mode helpers
# ---------------------------------------------------------------------------


class TestModeHelpers:
    def test_c_mode_creation(self):
        m = make_c_mode()
        assert m is not None

    def test_numba_mode_creation(self):
        m = make_numba_mode()
        assert m is not None


# ---------------------------------------------------------------------------
# Tests: Gemma3nConfig.from_text_config
# ---------------------------------------------------------------------------


class TestConfigAdapter:
    def test_from_text_config(self):
        """Test that Gemma3nConfig.from_text_config correctly bridges fields."""
        from dataclasses import dataclass

        @dataclass
        class MockTextConfig:
            hidden_size: int = 2048
            num_hidden_layers: int = 35
            intermediate_size: int = 16384
            num_attention_heads: int = 8
            head_dim: int = 256
            rms_norm_eps: float = 1e-6
            vocab_size: int = 262400
            num_key_value_heads: int = 2
            sliding_window: int = 512
            rope_local_base_freq: float = 10000.0
            rope_theta: float = 1000000.0
            final_logit_softcapping: float = 30.0
            hidden_size_per_layer_input: int = 256
            altup_num_inputs: int = 4
            altup_coef_clip: float = 120.0
            altup_correct_scale: bool = True
            altup_active_idx: int = 0
            laurel_rank: int = 64
            vocab_size_per_layer_input: int = 262144

        text_cfg = MockTextConfig()
        cfg = Gemma3nConfig.from_text_config(text_cfg)

        assert cfg.hidden_size == 2048
        assert cfg.num_hidden_layers == 35
        assert cfg.intermediate_size == 16384
        assert cfg.num_attention_heads == 8
        assert cfg.head_dim == 256
        assert cfg.rms_norm_eps == 1e-6
        assert cfg.vocab_size == 262400
        assert cfg.num_key_value_heads == 2
        assert cfg.sliding_window == 512
        assert cfg.rope_local_base_freq == 10000.0
        assert cfg.rope_theta == 1000000.0
        assert cfg.final_logit_softcapping == 30.0
        assert cfg.activation_sparsity == 0.0  # Always 0.0 per bridge
        assert cfg.hidden_size_per_layer_input == 256
        assert cfg.altup_num_inputs == 4
        assert cfg.altup_coef_clip == 120.0
        assert cfg.altup_correct_scale is True
        assert cfg.altup_active_idx == 0
        assert cfg.laurel_rank == 64
        assert cfg.vocab_size_per_layer_input == 262144


# ---------------------------------------------------------------------------
# Tests: Per-chunk logits
# ---------------------------------------------------------------------------


class TestPerChunkLogits:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_per_chunk_logits_no_softcap(self, backend):
        from cetagostini.utils.pytensor.gemma3n_pytensor import compile_per_chunk_logits

        H, B, T, C = 24, 1, 3, 16
        fn = compile_per_chunk_logits(H, B, T, C, backend=backend)

        hidden = det_array((B, T, H), seed=1000)
        chunk_emb = det_array((C, H), seed=1001)

        result = fn(hidden, chunk_emb)
        assert result.shape == (B, T, C)
        assert result.dtype == np.float32

        # Verify against manual computation
        expected = np.dot(hidden.reshape(B * T, H), chunk_emb.T).reshape(B, T, C)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_per_chunk_logits_with_softcap(self, backend):
        from cetagostini.utils.pytensor.gemma3n_pytensor import compile_per_chunk_logits

        H, B, T, C = 24, 1, 3, 16
        softcap = 30.0
        fn = compile_per_chunk_logits(H, B, T, C, softcap=softcap, backend=backend)

        hidden = det_array((B, T, H), seed=1010)
        chunk_emb = det_array((C, H), seed=1011)

        result = fn(hidden, chunk_emb)
        assert result.shape == (B, T, C)

        # Verify softcap is applied
        raw_logits = np.dot(hidden.reshape(B * T, H), chunk_emb.T).reshape(B, T, C)
        expected = softcap * np.tanh(raw_logits / softcap)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_per_chunk_logits_validation(self):
        from cetagostini.utils.pytensor.gemma3n_pytensor import compile_per_chunk_logits

        with pytest.raises(ValueError, match="must be >= 1"):
            compile_per_chunk_logits(24, 0, 3, 16, backend="c")

        with pytest.raises(ValueError, match="must be >= 1"):
            compile_per_chunk_logits(24, 1, 0, 16, backend="c")

        with pytest.raises(ValueError, match="must be >= 1"):
            compile_per_chunk_logits(24, 1, 3, 0, backend="c")

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_tail_sized_chunk(self, backend):
        from cetagostini.utils.pytensor.gemma3n_pytensor import compile_per_chunk_logits

        H, B, T, tail = 24, 1, 3, 7
        fn = compile_per_chunk_logits(H, B, T, tail, backend=backend)
        hidden = det_array((B, T, H), seed=1020)
        chunk = det_array((tail, H), seed=1021)

        result = fn(hidden, chunk)
        expected = (hidden.reshape(B * T, H) @ chunk.T).reshape(B, T, tail)

        assert result.shape == (B, T, tail)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Helper to resolve backend string
# ---------------------------------------------------------------------------


def _get_mode(backend):
    from cetagostini.utils.pytensor.gemma3n_pytensor import _get_mode as _gm
    return _gm(backend)
