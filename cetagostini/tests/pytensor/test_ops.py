"""Tests for model-independent symbolic PyTensor operations."""

from __future__ import annotations

import numpy as np
import pytest
import pytensor
import pytensor.tensor as pt

from cetagostini.utils.pytensor.backends import get_mode
from cetagostini.utils.pytensor.ops import (
    apply_rope_symbolic,
    build_rope_table,
    causal_mask,
    gqa_attention,
    linear_proj,
    rms_norm_no_scale,
    rmsnorm_symbolic,
    sliding_window_mask,
)


def test_linear_projection_matches_numpy():
    x = pt.tensor3("x", dtype="float32")
    weight = pt.matrix("weight", dtype="float32")
    output = linear_proj(x, weight, 1, 2, 3, 4)
    fn = pytensor.function([x, weight], output, mode="FAST_COMPILE")

    np_x = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    np_weight = np.arange(12, dtype=np.float32).reshape(3, 4)

    expected = (np_x.reshape(2, 3) @ np_weight).reshape(1, 2, 4)
    np.testing.assert_allclose(fn(np_x, np_weight), expected)


def test_rms_normalization_variants_match_numpy():
    x = pt.matrix("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")

    learned = rmsnorm_symbolic(x, gamma)
    unscaled = rms_norm_no_scale(x)
    fn = pytensor.function([x, gamma], [learned, unscaled], mode="FAST_COMPILE")

    np_x = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]], dtype=np.float32)
    np_gamma = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    expected = np_x / np.sqrt(np.mean(np_x**2, axis=-1, keepdims=True) + 1e-5)

    actual_learned, actual_unscaled = fn(np_x, np_gamma)
    np.testing.assert_allclose(actual_learned, expected * np_gamma, rtol=1e-6)
    np.testing.assert_allclose(actual_unscaled, expected, rtol=1e-6)


def test_rope_table_and_symbolic_rotation():
    cos, sin = build_rope_table(base=10000.0, head_dim=4, sequence_length=2)

    assert cos.shape == (2, 2)
    assert sin.shape == (2, 2)
    assert cos.dtype == np.float32
    assert sin.dtype == np.float32

    x = pt.tensor4("x", dtype="float32")
    pt_cos = pt.tensor("cos", dtype="float32", shape=(None, None))
    pt_sin = pt.tensor("sin", dtype="float32", shape=(None, None))
    output = apply_rope_symbolic(x, pt_cos, pt_sin, head_dim=4)
    fn = pytensor.function([x, pt_cos, pt_sin], output, mode="FAST_COMPILE")

    np_x = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32)

    identity_cos = np.ones((1, 2), dtype=np.float32)
    zero_sin = np.zeros((1, 2), dtype=np.float32)

    np.testing.assert_allclose(fn(np_x, identity_cos, zero_sin), np_x)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base": 0.0, "head_dim": 4, "sequence_length": 2},
        {"base": 10000.0, "head_dim": 3, "sequence_length": 2},
        {"base": 10000.0, "head_dim": 4, "sequence_length": 0},
    ],
)
def test_rope_table_validates_dimensions(kwargs):
    with pytest.raises(ValueError):
        build_rope_table(**kwargs)


def test_causal_and_sliding_masks():
    expected_causal = np.array(
        [
            [0.0, -np.inf, -np.inf, -np.inf],
            [0.0, 0.0, -np.inf, -np.inf],
            [0.0, 0.0, 0.0, -np.inf],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    # Sliding window with window=2: each position can attend to itself and 1 prior
    expected_sliding = np.array(
        [
            [0.0, -np.inf, -np.inf, -np.inf],
            [0.0, 0.0, -np.inf, -np.inf],
            [-np.inf, 0.0, 0.0, -np.inf],
            [-np.inf, -np.inf, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(causal_mask(4), expected_causal)
    np.testing.assert_array_equal(sliding_window_mask(4, 2), expected_sliding)


def test_gqa_attention_repeats_key_value_heads():
    q = pt.tensor4("q", dtype="float32")
    k = pt.tensor4("k", dtype="float32")
    v = pt.tensor4("v", dtype="float32")
    mask = pt.matrix("mask", dtype="float32")

    output = gqa_attention(
        q, k, v, mask,
        n_heads=2, n_kv_heads=1, head_dim=2,
        batch_size=1, sequence_length=2, scale=1.0,
    )
    fn = pytensor.function([q, k, v, mask], output, mode="FAST_COMPILE")

    np_q = np.zeros((1, 2, 2, 2), dtype=np.float32)
    np_k = np.zeros((1, 2, 1, 2), dtype=np.float32)
    np_v = np.array([[[[2.0, 3.0]]]], dtype=np.float32)
    np_v = np.broadcast_to(np_v, (1, 2, 1, 2)).copy()
    np_mask = np.zeros((2, 2), dtype=np.float32)

    expected = np.broadcast_to(np_v[:, :, [0, 0], :], (1, 2, 2, 2)).copy()
    np.testing.assert_allclose(fn(np_q, np_k, np_v, np_mask), expected)


def test_gqa_attention_validates_head_ratio():
    x = pt.tensor4("x", dtype="float32")
    mask = pt.matrix("mask", dtype="float32")

    with pytest.raises(ValueError, match="divisible"):
        gqa_attention(x, x, x, mask, n_heads=3, n_kv_heads=2, head_dim=1, batch_size=1, sequence_length=1, scale=1.0)


def test_backend_resolver_keeps_model_code_separate():
    assert get_mode("FAST_COMPILE") == "FAST_COMPILE"

    c_mode = get_mode("c")
    assert c_mode.linker is not c_mode
    assert c_mode.optimizer is not c_mode

    with pytest.raises(ValueError, match="Unknown backend"):
        get_mode("not-a-backend")
