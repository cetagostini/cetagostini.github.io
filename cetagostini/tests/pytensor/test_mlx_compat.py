"""Tests for clip_symbolic, MLX backend mode, and Gemma MLX integration.

Focused unit tests covering:
- Lazy import hygiene (root package does not import mlx or mlx_compat)
- clip_symbolic vs independent NumPy ordered-where reference
- clip_symbolic vs ordinary pt.clip under FAST_COMPILE
- Built-in Clip dispatch registry untouched by make_mlx_mode
- MLX mode/linker correctness
- Toy Gemma dense/sparse/return/shared decoder layers MLX vs FAST_COMPILE
- All compile helper stages MLX vs FAST_COMPILE
- Runner sync ordering using mocks
- API/CLI mlx routing
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt

from cetagostini.utils.pytensor.gemma3n_pytensor import (
    Gemma3nConfig,
    build_rope_table,
    causal_mask,
    compile_decoder_layer,
    compile_final_unembed,
    compile_initial_projections,
    compile_per_chunk_logits,
    compile_per_layer_projection,
    sliding_window_mask,
)
from cetagostini.utils.pytensor.mlx_compat import clip_symbolic


# ---------------------------------------------------------------------------
# Test configuration — same small config as test_gemma3n_pytensor
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


def det_array(shape, seed, low=0.1, high=1.7):
    """Deterministic, asymmetric, nonzero float32 array."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(low, high, size=shape).astype(np.float32)
    ramp = np.arange(x.size, dtype=np.float32).reshape(shape) * 1e-3
    return x + ramp


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


def _resolve_mode(backend):
    """Resolve a backend name to a PyTensor mode for tests."""
    if backend == "mlx":
        from cetagostini.utils.pytensor.backends import make_mlx_mode
        return make_mlx_mode()
    if backend in ("c", "cvm"):
        from cetagostini.utils.pytensor.backends import make_c_mode
        return make_c_mode()
    if backend == "numba":
        from cetagostini.utils.pytensor.backends import make_numba_mode
        return make_numba_mode()
    return backend  # "FAST_COMPILE"


# ---------------------------------------------------------------------------
# Independent NumPy oracle (NOT np.clip — reversed-bound semantics differ)
# ---------------------------------------------------------------------------


def np_clip_ordered_where(x, lower, upper):
    """Independent NumPy reference matching PyTensor ordered-where clip.

    Implements the same nested-where logic as ``clip_symbolic``::

        clamped_hi = where(x > upper, upper, x)
        result     = where(x < lower, lower, clamped_hi)

    This is **not** ``np.clip`` — reversed-bound semantics differ.
    """
    clamped_hi = np.where(x > upper, upper, x)
    return np.where(x < lower, lower, clamped_hi)


# ---------------------------------------------------------------------------
# Tests: Lazy import hygiene
# ---------------------------------------------------------------------------


class TestLazyImport:
    """Root package import must not import mlx, pytensor.link.mlx, or mlx_compat."""

    def test_root_package_does_not_import_mlx(self):
        """Importing cetagostini.utils.pytensor should not pull in mlx."""
        mods_to_remove = [
            k for k in sys.modules
            if "mlx_compat" in k
        ]
        saved = {}
        for k in mods_to_remove:
            saved[k] = sys.modules.pop(k)

        try:
            import cetagostini.utils.pytensor as pkg
            importlib.reload(pkg)

            assert "cetagostini.utils.pytensor.mlx_compat" not in sys.modules, (
                "Root package import should not import mlx_compat"
            )
        finally:
            sys.modules.update(saved)

    def test_mlx_compat_module_importable_without_mlx(self):
        """mlx_compat module should be importable without importing mlx."""
        from cetagostini.utils.pytensor import mlx_compat

        # Module should expose clip_symbolic
        assert hasattr(mlx_compat, "clip_symbolic")
        assert callable(mlx_compat.clip_symbolic)

        # Module should NOT have ensure_mlx_lowerings (removed)
        assert not hasattr(mlx_compat, "ensure_mlx_lowerings"), (
            "ensure_mlx_lowerings should be removed from mlx_compat"
        )


# ---------------------------------------------------------------------------
# Tests: clip_symbolic vs independent NumPy ordered-where oracle
# ---------------------------------------------------------------------------


class TestClipSymbolicVsNumpyOracle:
    """Verify clip_symbolic matches independent NumPy ordered-where reference."""

    @pytest.mark.parametrize("backend", ["FAST_COMPILE", "c", "numba"])
    @pytest.mark.parametrize(
        "x_val,lo_val,hi_val",
        [
            (-5.0, 0.0, 10.0),
            (0.0, 0.0, 10.0),
            (5.0, 0.0, 10.0),
            (10.0, 0.0, 10.0),
            (15.0, 0.0, 10.0),
            (1.0, 3.0, 3.0),
            (3.0, 3.0, 3.0),
            (5.0, 3.0, 3.0),
            (-5.0, 10.0, 0.0),
            (5.0, 10.0, 0.0),
            (15.0, 10.0, 0.0),
        ],
        ids=[
            "below", "equal_lower", "between", "equal_upper", "above",
            "equal_bounds_below", "equal_bounds_equal", "equal_bounds_above",
            "reversed_below", "reversed_between", "reversed_above",
        ],
    )
    def test_scalar_regions(self, backend, x_val, lo_val, hi_val):
        """Scalar float32 across all regions and backends."""
        x_s = pt.scalar("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")
        out = clip_symbolic(x_s, lo_s, hi_s)

        fn = pytensor.function([x_s, lo_s, hi_s], out, mode=_resolve_mode(backend))

        x_np = np.float32(x_val)
        lo_np = np.float32(lo_val)
        hi_np = np.float32(hi_val)

        expected = np_clip_ordered_where(x_np, lo_np, hi_np)
        actual = fn(x_np, lo_np, hi_np)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    @pytest.mark.parametrize("backend", ["FAST_COMPILE", "c", "numba"])
    def test_nan_value(self, backend):
        """NaN x should propagate through clip_symbolic."""
        x_s = pt.scalar("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")
        out = clip_symbolic(x_s, lo_s, hi_s)

        fn = pytensor.function([x_s, lo_s, hi_s], out, mode=_resolve_mode(backend))

        x_np = np.float32(np.nan)
        lo_np = np.float32(0.0)
        hi_np = np.float32(10.0)

        expected = np_clip_ordered_where(x_np, lo_np, hi_np)
        actual = fn(x_np, lo_np, hi_np)
        assert np.isnan(expected)
        assert np.isnan(actual)

    @pytest.mark.parametrize("backend", ["FAST_COMPILE", "c", "numba"])
    def test_nan_lower(self, backend):
        """NaN lower should propagate per ordered-where semantics."""
        x_s = pt.scalar("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")
        out = clip_symbolic(x_s, lo_s, hi_s)

        fn = pytensor.function([x_s, lo_s, hi_s], out, mode=_resolve_mode(backend))

        x_np = np.float32(5.0)
        lo_np = np.float32(np.nan)
        hi_np = np.float32(10.0)

        expected = np_clip_ordered_where(x_np, lo_np, hi_np)
        actual = fn(x_np, lo_np, hi_np)
        # NaN comparisons are False, so x passes through
        np.testing.assert_allclose(actual, expected, atol=1e-6, equal_nan=True)

    @pytest.mark.parametrize("backend", ["FAST_COMPILE", "c", "numba"])
    def test_nan_upper(self, backend):
        """NaN upper should propagate per ordered-where semantics."""
        x_s = pt.scalar("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")
        out = clip_symbolic(x_s, lo_s, hi_s)

        fn = pytensor.function([x_s, lo_s, hi_s], out, mode=_resolve_mode(backend))

        x_np = np.float32(5.0)
        lo_np = np.float32(0.0)
        hi_np = np.float32(np.nan)

        expected = np_clip_ordered_where(x_np, lo_np, hi_np)
        actual = fn(x_np, lo_np, hi_np)
        np.testing.assert_allclose(actual, expected, atol=1e-6, equal_nan=True)

    @pytest.mark.parametrize("backend", ["FAST_COMPILE", "c", "numba"])
    def test_pos_inf_value(self, backend):
        """+inf x should be clamped to upper."""
        x_s = pt.scalar("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")
        out = clip_symbolic(x_s, lo_s, hi_s)

        fn = pytensor.function([x_s, lo_s, hi_s], out, mode=_resolve_mode(backend))

        x_np = np.float32(np.inf)
        lo_np = np.float32(0.0)
        hi_np = np.float32(10.0)

        expected = np_clip_ordered_where(x_np, lo_np, hi_np)
        actual = fn(x_np, lo_np, hi_np)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    @pytest.mark.parametrize("backend", ["FAST_COMPILE", "c", "numba"])
    def test_neg_inf_value(self, backend):
        """-inf x should be clamped to lower."""
        x_s = pt.scalar("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")
        out = clip_symbolic(x_s, lo_s, hi_s)

        fn = pytensor.function([x_s, lo_s, hi_s], out, mode=_resolve_mode(backend))

        x_np = np.float32(-np.inf)
        lo_np = np.float32(0.0)
        hi_np = np.float32(10.0)

        expected = np_clip_ordered_where(x_np, lo_np, hi_np)
        actual = fn(x_np, lo_np, hi_np)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    @pytest.mark.parametrize("backend", ["FAST_COMPILE", "c", "numba"])
    def test_pos_inf_upper(self, backend):
        """+inf upper means no upper clamp."""
        x_s = pt.scalar("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")
        out = clip_symbolic(x_s, lo_s, hi_s)

        fn = pytensor.function([x_s, lo_s, hi_s], out, mode=_resolve_mode(backend))

        x_np = np.float32(1e30)
        lo_np = np.float32(0.0)
        hi_np = np.float32(np.inf)

        expected = np_clip_ordered_where(x_np, lo_np, hi_np)
        actual = fn(x_np, lo_np, hi_np)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    @pytest.mark.parametrize("backend", ["FAST_COMPILE", "c", "numba"])
    def test_neg_inf_lower(self, backend):
        """-inf lower means no lower clamp."""
        x_s = pt.scalar("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")
        out = clip_symbolic(x_s, lo_s, hi_s)

        fn = pytensor.function([x_s, lo_s, hi_s], out, mode=_resolve_mode(backend))

        x_np = np.float32(-1e30)
        lo_np = np.float32(-np.inf)
        hi_np = np.float32(10.0)

        expected = np_clip_ordered_where(x_np, lo_np, hi_np)
        actual = fn(x_np, lo_np, hi_np)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    @pytest.mark.parametrize("backend", ["FAST_COMPILE", "c", "numba"])
    def test_vector_float32(self, backend):
        """Vector float32 across all regions."""
        x_s = pt.vector("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")
        out = clip_symbolic(x_s, lo_s, hi_s)

        fn = pytensor.function([x_s, lo_s, hi_s], out, mode=_resolve_mode(backend))

        x_np = np.array([-5.0, 0.0, 5.0, 10.0, 15.0], dtype=np.float32)
        lo_np = np.float32(0.0)
        hi_np = np.float32(10.0)

        expected = np_clip_ordered_where(x_np, lo_np, hi_np)
        actual = fn(x_np, lo_np, hi_np)
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    @pytest.mark.parametrize("backend", ["FAST_COMPILE", "c", "numba"])
    def test_matrix_float32(self, backend):
        """Matrix float32 with reversed bounds."""
        x_s = pt.matrix("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")
        out = clip_symbolic(x_s, lo_s, hi_s)

        fn = pytensor.function([x_s, lo_s, hi_s], out, mode=_resolve_mode(backend))

        rng = np.random.default_rng(42)
        x_np = rng.uniform(-10.0, 20.0, size=(3, 4)).astype(np.float32)
        lo_np = np.float32(10.0)  # reversed: lo > hi
        hi_np = np.float32(0.0)

        expected = np_clip_ordered_where(x_np, lo_np, hi_np)
        actual = fn(x_np, lo_np, hi_np)
        np.testing.assert_allclose(actual, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: clip_symbolic vs ordinary pt.clip under FAST_COMPILE
# ---------------------------------------------------------------------------


class TestClipSymbolicVsPtClip:
    """Verify clip_symbolic matches ordinary pt.clip for normal bounds."""

    @pytest.mark.parametrize("backend", ["FAST_COMPILE", "c", "numba"])
    def test_parity_scalar_normal_bounds(self, backend):
        """clip_symbolic matches pt.clip for lo <= hi."""
        x_s = pt.scalar("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")

        out_sym = clip_symbolic(x_s, lo_s, hi_s)
        out_clip = pt.clip(x_s, lo_s, hi_s)

        fn_sym = pytensor.function([x_s, lo_s, hi_s], out_sym, mode=_resolve_mode(backend))
        fn_clip = pytensor.function([x_s, lo_s, hi_s], out_clip, mode="FAST_COMPILE")

        for x_val in [-5.0, 0.0, 5.0, 10.0, 15.0]:
            x_np = np.float32(x_val)
            lo_np = np.float32(0.0)
            hi_np = np.float32(10.0)
            expected = fn_clip(x_np, lo_np, hi_np)
            actual = fn_sym(x_np, lo_np, hi_np)
            np.testing.assert_allclose(actual, expected, atol=1e-6)

    @pytest.mark.parametrize("backend", ["FAST_COMPILE", "c", "numba"])
    def test_parity_vector(self, backend):
        """clip_symbolic matches pt.clip for vector inputs."""
        x_s = pt.vector("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")

        out_sym = clip_symbolic(x_s, lo_s, hi_s)
        out_clip = pt.clip(x_s, lo_s, hi_s)

        fn_sym = pytensor.function([x_s, lo_s, hi_s], out_sym, mode=_resolve_mode(backend))
        fn_clip = pytensor.function([x_s, lo_s, hi_s], out_clip, mode="FAST_COMPILE")

        x_np = np.array([-5.0, 0.0, 5.0, 10.0, 15.0], dtype=np.float32)
        lo_np = np.float32(0.0)
        hi_np = np.float32(10.0)

        expected = fn_clip(x_np, lo_np, hi_np)
        actual = fn_sym(x_np, lo_np, hi_np)
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    @pytest.mark.parametrize("backend", ["FAST_COMPILE", "c", "numba"])
    def test_parity_reversed_bounds(self, backend):
        """clip_symbolic matches pt.clip for reversed bounds (PyTensor authority)."""
        x_s = pt.scalar("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")

        out_sym = clip_symbolic(x_s, lo_s, hi_s)
        out_clip = pt.clip(x_s, lo_s, hi_s)

        fn_sym = pytensor.function([x_s, lo_s, hi_s], out_sym, mode=_resolve_mode(backend))
        fn_clip = pytensor.function([x_s, lo_s, hi_s], out_clip, mode="FAST_COMPILE")

        for x_val in [-5.0, 5.0, 15.0]:
            x_np = np.float32(x_val)
            lo_np = np.float32(10.0)
            hi_np = np.float32(0.0)
            expected = fn_clip(x_np, lo_np, hi_np)
            actual = fn_sym(x_np, lo_np, hi_np)
            np.testing.assert_allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: Built-in Clip dispatch registry untouched
# ---------------------------------------------------------------------------


class TestBuiltinClipRegistryUntouched:
    """Verify make_mlx_mode does not mutate the built-in Clip dispatch registry."""

    def test_clip_registry_unchanged_after_make_mlx_mode(self):
        """make_mlx_mode must not add or modify entries in mlx_funcify."""
        from pytensor.link.mlx.dispatch import mlx_funcify

        registry_before = dict(mlx_funcify.registry)

        from cetagostini.utils.pytensor.backends import make_mlx_mode
        make_mlx_mode()

        # Same number of entries
        assert len(mlx_funcify.registry) == len(registry_before), (
            "make_mlx_mode added new entries to mlx_funcify registry"
        )

        # All existing entries unchanged
        for key, val in registry_before.items():
            assert mlx_funcify.registry[key] is val, (
                f"Existing handler for {key} was mutated"
            )

    def test_no_ensure_mlx_lowerings_in_module(self):
        """mlx_compat should not expose ensure_mlx_lowerings."""
        from cetagostini.utils.pytensor import mlx_compat

        assert not hasattr(mlx_compat, "ensure_mlx_lowerings")
        assert not hasattr(mlx_compat, "_registered")
        assert not hasattr(mlx_compat, "is_registered")


# ---------------------------------------------------------------------------
# Tests: MLX mode tags and linker
# ---------------------------------------------------------------------------


class TestMLXMode:
    """Verify make_mlx_mode returns the built-in MLX mode with correct tags."""

    def test_make_mlx_mode_returns_mode(self):
        from cetagostini.utils.pytensor.backends import make_mlx_mode
        from pytensor.compile.mode import Mode

        mode = make_mlx_mode()
        assert isinstance(mode, Mode)

    def test_make_mlx_mode_is_builtin_mlx(self):
        from cetagostini.utils.pytensor.backends import make_mlx_mode
        from pytensor.compile.mode import MLX

        mode = make_mlx_mode()
        assert mode is MLX

    def test_mlx_mode_include_tags(self):
        from cetagostini.utils.pytensor.backends import make_mlx_mode

        mode = make_mlx_mode()
        opt = mode._optimizer
        include = list(opt.include)
        assert "fast_run" in include
        assert "mlx" in include

    def test_mlx_mode_excludes_fusion(self):
        from cetagostini.utils.pytensor.backends import make_mlx_mode

        mode = make_mlx_mode()
        opt = mode._optimizer
        exclude = list(opt.exclude)
        assert "fusion" in exclude

    def test_mlx_mode_linker_is_mlx(self):
        from cetagostini.utils.pytensor.backends import make_mlx_mode
        from pytensor.link.mlx.linker import MLXLinker

        mode = make_mlx_mode()
        assert isinstance(mode.linker, MLXLinker)

    def test_make_mlx_mode_idempotent(self):
        from cetagostini.utils.pytensor.backends import make_mlx_mode

        mode1 = make_mlx_mode()
        mode2 = make_mlx_mode()
        assert mode1 is mode2


# ---------------------------------------------------------------------------
# Tests: clip_symbolic under MLX backend
# ---------------------------------------------------------------------------


class TestClipSymbolicMLX:
    """Verify clip_symbolic works correctly under the MLX backend."""

    def test_mlx_scalar_normal(self):
        """clip_symbolic under MLX matches FAST_COMPILE for normal bounds."""
        x_s = pt.scalar("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")
        out = clip_symbolic(x_s, lo_s, hi_s)

        fn_fast = pytensor.function([x_s, lo_s, hi_s], out, mode="FAST_COMPILE")
        fn_mlx = pytensor.function(
            [x_s, lo_s, hi_s], out, mode=_resolve_mode("mlx"),
        )

        x_val = np.float32(5.0)
        lo_val = np.float32(0.0)
        hi_val = np.float32(10.0)

        expected = fn_fast(x_val, lo_val, hi_val)
        actual = fn_mlx(x_val, lo_val, hi_val)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_mlx_reversed_bounds(self):
        """clip_symbolic under MLX handles reversed bounds correctly."""
        x_s = pt.scalar("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")
        out = clip_symbolic(x_s, lo_s, hi_s)

        fn_fast = pytensor.function([x_s, lo_s, hi_s], out, mode="FAST_COMPILE")
        fn_mlx = pytensor.function(
            [x_s, lo_s, hi_s], out, mode=_resolve_mode("mlx"),
        )

        x_val = np.float32(5.0)
        lo_val = np.float32(10.0)
        hi_val = np.float32(0.0)

        expected = fn_fast(x_val, lo_val, hi_val)
        actual = fn_mlx(x_val, lo_val, hi_val)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_mlx_vector(self):
        """clip_symbolic under MLX works for vector inputs."""
        x_s = pt.vector("x", dtype="float32")
        lo_s = pt.scalar("lo", dtype="float32")
        hi_s = pt.scalar("hi", dtype="float32")
        out = clip_symbolic(x_s, lo_s, hi_s)

        fn_fast = pytensor.function([x_s, lo_s, hi_s], out, mode="FAST_COMPILE")
        fn_mlx = pytensor.function(
            [x_s, lo_s, hi_s], out, mode=_resolve_mode("mlx"),
        )

        x_val = np.array([-5.0, 0.0, 5.0, 10.0, 15.0], dtype=np.float32)
        lo_val = np.float32(0.0)
        hi_val = np.float32(10.0)

        expected = fn_fast(x_val, lo_val, hi_val)
        actual = fn_mlx(x_val, lo_val, hi_val)
        np.testing.assert_allclose(actual, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Toy Gemma decoder layer MLX vs FAST_COMPILE
# ---------------------------------------------------------------------------


class TestGemmaDecoderLayerMLX:
    """Toy Gemma dense/sparse/return/shared decoder layers MLX vs FAST_COMPILE."""

    @pytest.mark.parametrize(
        "is_sliding,has_sparsity",
        [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ],
        ids=["full_dense", "full_sparse", "sliding_dense", "sliding_sparse"],
    )
    def test_decoder_layer_mlx_vs_fast_compile(self, is_sliding, has_sparsity):
        cfg = SMALL_CONFIG
        B, T = 1, 4
        n = cfg.altup_num_inputs
        H = cfg.hidden_size
        H_pl = cfg.hidden_size_per_layer_input
        hd = cfg.head_dim

        fn_fast = compile_decoder_layer(
            cfg, B, T, has_sparsity=has_sparsity, backend="FAST_COMPILE",
        )
        fn_mlx = compile_decoder_layer(
            cfg, B, T, has_sparsity=has_sparsity, backend="mlx",
        )

        w = make_layer_weights(cfg, seed_base=200)
        hidden = det_array((n, B, T, H), seed=210)
        pli = det_array((B, T, H_pl), seed=211)

        if is_sliding:
            mask = sliding_window_mask(T, cfg.sliding_window)
        else:
            mask = causal_mask(T)

        cos, sin = build_rope_table(cfg.rope_theta, hd, T)

        args = (hidden, mask, pli, cos, sin, *layer_weight_args(w))
        result_fast = fn_fast(*args)
        result_mlx = fn_mlx(*args)

        np.testing.assert_allclose(result_mlx, result_fast, atol=1e-2, rtol=1e-2)

    def test_decoder_kv_mode_return_mlx(self):
        """compile_decoder_layer with kv_mode='return' works under MLX."""
        cfg = SMALL_CONFIG
        B, T = 1, 4
        fn = compile_decoder_layer(cfg, B, T, backend="mlx", kv_mode="return")
        assert len(fn.maker.fgraph.outputs) == 3

        n, H = cfg.altup_num_inputs, cfg.hidden_size
        H_pl = cfg.hidden_size_per_layer_input
        hd = cfg.head_dim

        w = make_layer_weights(cfg, seed_base=300)
        hidden = det_array((n, B, T, H), seed=310)
        pli = det_array((B, T, H_pl), seed=311)
        mask = causal_mask(T)
        cos, sin = build_rope_table(cfg.rope_theta, hd, T)

        result = fn(hidden, mask, pli, cos, sin, *layer_weight_args(w))
        assert len(result) == 3
        corrected, k_out, v_out = result
        assert corrected.shape == (n, B, T, H)
        assert k_out.shape == (B, cfg.num_key_value_heads, T, hd)
        assert v_out.shape == (B, cfg.num_key_value_heads, T, hd)

    def test_decoder_kv_mode_shared_mlx(self):
        """compile_decoder_layer with kv_mode='shared' works under MLX."""
        cfg = SMALL_CONFIG
        B, T = 1, 4
        fn = compile_decoder_layer(cfg, B, T, backend="mlx", kv_mode="shared")
        n_inputs = len(fn.maker.fgraph.inputs)
        # 5 base + 2 shared + 24 weights = 31
        assert n_inputs == 31


# ---------------------------------------------------------------------------
# Tests: All compile helper stages MLX vs FAST_COMPILE
# ---------------------------------------------------------------------------


class TestCompileHelpersMLX:
    """All compile helper stages should produce matching results under MLX."""

    def test_initial_projections_mlx(self):
        cfg = SMALL_CONFIG
        B, T, H = 1, 3, cfg.hidden_size
        n = cfg.altup_num_inputs

        fn_fast = compile_initial_projections(cfg, B, T, backend="FAST_COMPILE")
        fn_mlx = compile_initial_projections(cfg, B, T, backend="mlx")

        h0 = det_array((B, T, H), seed=500)
        proj_ws = [det_array((H, H), seed=510 + i) for i in range(n - 1)]

        result_fast = fn_fast(h0, *proj_ws)
        result_mlx = fn_mlx(h0, *proj_ws)
        np.testing.assert_allclose(result_mlx, result_fast, atol=1e-4, rtol=1e-4)

    def test_per_layer_projection_mlx(self):
        cfg = SMALL_CONFIG
        B, T = 1, 3
        H = cfg.hidden_size
        L = cfg.num_hidden_layers
        H_pl = cfg.hidden_size_per_layer_input

        fn_fast = compile_per_layer_projection(cfg, B, T, backend="FAST_COMPILE")
        fn_mlx = compile_per_layer_projection(cfg, B, T, backend="mlx")

        embeds = det_array((B, T, H), seed=600)
        pw = det_array((H, L * H_pl), seed=601)
        ng = det_array((H_pl,), seed=602)
        ple = det_array((B, T, L, H_pl), seed=603)

        result_fast = fn_fast(embeds, pw, ng, ple)
        result_mlx = fn_mlx(embeds, pw, ng, ple)
        np.testing.assert_allclose(result_mlx, result_fast, atol=1e-4, rtol=1e-4)

    def test_final_unembed_mlx(self):
        cfg = SMALL_CONFIG
        B, T, H = 1, 3, cfg.hidden_size
        n = cfg.altup_num_inputs

        fn_fast = compile_final_unembed(cfg, B, T, backend="FAST_COMPILE")
        fn_mlx = compile_final_unembed(cfg, B, T, backend="mlx")

        h = det_array((n, B, T, H), seed=700)
        unembed_ws = [det_array((H, H), seed=710 + i) for i in range(n - 1)]
        fn_g = det_array((H,), seed=720)

        result_fast = fn_fast(h, *unembed_ws, fn_g)
        result_mlx = fn_mlx(h, *unembed_ws, fn_g)
        np.testing.assert_allclose(result_mlx, result_fast, atol=1e-4, rtol=1e-4)

    def test_per_chunk_logits_mlx(self):
        H, B, T, C = 24, 1, 3, 16
        fn_fast = compile_per_chunk_logits(H, B, T, C, backend="FAST_COMPILE")
        fn_mlx = compile_per_chunk_logits(H, B, T, C, backend="mlx")

        hidden = det_array((B, T, H), seed=1000)
        chunk_emb = det_array((C, H), seed=1001)

        result_fast = fn_fast(hidden, chunk_emb)
        result_mlx = fn_mlx(hidden, chunk_emb)
        np.testing.assert_allclose(result_mlx, result_fast, atol=1e-4, rtol=1e-4)

    def test_per_chunk_logits_with_softcap_mlx(self):
        H, B, T, C = 24, 1, 3, 16
        softcap = 30.0
        fn_fast = compile_per_chunk_logits(
            H, B, T, C, softcap=softcap, backend="FAST_COMPILE",
        )
        fn_mlx = compile_per_chunk_logits(
            H, B, T, C, softcap=softcap, backend="mlx",
        )

        hidden = det_array((B, T, H), seed=1010)
        chunk_emb = det_array((C, H), seed=1011)

        result_fast = fn_fast(hidden, chunk_emb)
        result_mlx = fn_mlx(hidden, chunk_emb)
        np.testing.assert_allclose(result_mlx, result_fast, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: Runner sync ordering using mocks
# ---------------------------------------------------------------------------


class TestRunnerSyncOrdering:
    """Verify MLX sync is called at the right stages in run_pytensor_forward."""

    def test_maybe_sync_noop_for_c_backend(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import _maybe_sync

        value = np.array([1.0, 2.0], dtype=np.float32)
        sync_dt, result = _maybe_sync(value, "c", label="test")
        assert sync_dt == 0.0
        np.testing.assert_array_equal(result, value)

    def test_maybe_sync_noop_for_numba_backend(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import _maybe_sync

        value = np.array([1.0, 2.0], dtype=np.float32)
        sync_dt, result = _maybe_sync(value, "numba", label="test")
        assert sync_dt == 0.0
        np.testing.assert_array_equal(result, value)

    def test_maybe_sync_calls_mlx_for_mlx_backend(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import _maybe_sync

        mock_mx = MagicMock()
        mock_mx.eval = MagicMock()

        value = np.array([1.0, 2.0], dtype=np.float32)

        with patch.dict(sys.modules, {"mlx.core": mock_mx}):
            with patch(
                "cetagostini.utils.pytensor.run_gemma3n_pytensor._mlx_sync",
                return_value=(0.001, value),
            ) as mock_sync:
                sync_dt, result = _maybe_sync(value, "mlx", label="test")
                mock_sync.assert_called_once_with(value, label="test")
                assert sync_dt == 0.001

    def test_mlx_sync_converts_to_numpy(self):
        """_mlx_sync should call mx.eval and convert to numpy."""
        import mlx.core as mx

        from cetagostini.utils.pytensor.run_gemma3n_pytensor import _mlx_sync

        mlx_arr = mx.array([1.0, 2.0, 3.0])
        sync_dt, result = _mlx_sync(mlx_arr, label="test")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])
        assert sync_dt >= 0.0


# ---------------------------------------------------------------------------
# Tests: API/CLI mlx routing
# ---------------------------------------------------------------------------


class TestAPIRouting:
    """Verify Gemma3n accepts mlx backend and CLI routes correctly."""

    def test_gemma3n_accepts_mlx_backend(self, tmp_path):
        from cetagostini.utils.pytensor import Gemma3n

        model = Gemma3n.from_snapshot(tmp_path / "snapshot", backend="mlx")
        assert model.backend == "mlx"

    def test_gemma3n_rejects_invalid_backend(self, tmp_path):
        from cetagostini.utils.pytensor import Gemma3n

        with pytest.raises(ValueError, match="'c', 'numba', or 'mlx'"):
            Gemma3n.from_snapshot(tmp_path / "snapshot", backend="jax")

    def test_cli_accepts_mlx_backend(self, tmp_path):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import parse_args

        snap = tmp_path / "snap"
        snap.mkdir()
        ref_report = tmp_path / "ref_report.json"
        ref_report.write_text("{}", encoding="utf-8")
        ref_logits = tmp_path / "ref_logits.npy"
        ref_logits.write_bytes(b"\x00" * 64)
        args = parse_args([
            "run", "--snapshot", str(snap),
            "--run-id", "test",
            "--reference-report", str(ref_report),
            "--reference-logits", str(ref_logits),
            "--backend", "mlx",
        ])
        assert args.backend == "mlx"

    def test_valid_backends_includes_mlx(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import VALID_BACKENDS

        assert "mlx" in VALID_BACKENDS
        assert "c" in VALID_BACKENDS
        assert "numba" in VALID_BACKENDS

    def test_get_backend_info_mlx(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import get_backend_info

        info = get_backend_info("mlx")
        assert info["name"] == "mlx"
        assert info["linker"] == "mlx"
        assert "mlx" in info["mode"]


# ---------------------------------------------------------------------------
# Tests: _get_mode routes mlx correctly
# ---------------------------------------------------------------------------


class TestGetModeMLX:
    """Verify _get_mode in gemma3n_pytensor routes 'mlx' correctly."""

    def test_get_mode_mlx(self):
        from cetagostini.utils.pytensor.gemma3n_pytensor import _get_mode
        from pytensor.compile.mode import MLX

        mode = _get_mode("mlx")
        assert mode is MLX

    def test_get_mode_unknown_raises(self):
        from cetagostini.utils.pytensor.gemma3n_pytensor import _get_mode

        with pytest.raises(ValueError, match="Unknown backend"):
            _get_mode("unknown")


# ---------------------------------------------------------------------------
# Tests: Production-shape one-layer compile/run (bounded)
# ---------------------------------------------------------------------------


class TestProductionShapeOneLayer:
    """Compile and run a single decoder layer with production-like shapes.

    Uses a small but realistic config to keep compilation time bounded.
    """

    def test_one_layer_production_shape(self):
        """Single decoder layer with moderate dimensions compiles and runs."""
        cfg = Gemma3nConfig(
            hidden_size=64,
            num_hidden_layers=1,
            intermediate_size=128,
            num_attention_heads=4,
            head_dim=16,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            sliding_window=8,
            rope_local_base_freq=10_000.0,
            rope_theta=100_000.0,
            final_logit_softcapping=30.0,
            activation_sparsity=0.0,
            hidden_size_per_layer_input=16,
            altup_num_inputs=4,
            altup_coef_clip=1.0,
            altup_correct_scale=True,
            altup_active_idx=0,
            laurel_rank=8,
            vocab_size_per_layer_input=64,
        )
        B, T = 1, 8

        fn_mlx = compile_decoder_layer(cfg, B, T, backend="mlx")
        fn_fast = compile_decoder_layer(cfg, B, T, backend="FAST_COMPILE")

        n, H = cfg.altup_num_inputs, cfg.hidden_size
        H_pl = cfg.hidden_size_per_layer_input
        hd = cfg.head_dim

        w = make_layer_weights(cfg, seed_base=400)
        hidden = det_array((n, B, T, H), seed=410) * 0.1
        pli = det_array((B, T, H_pl), seed=411) * 0.1
        mask = causal_mask(T)
        cos, sin = build_rope_table(cfg.rope_theta, hd, T)

        args = (hidden, mask, pli, cos, sin, *layer_weight_args(w))
        result_fast = fn_fast(*args)
        result_mlx = fn_mlx(*args)

        assert result_mlx.shape == result_fast.shape
        assert np.all(np.isfinite(result_mlx))
        np.testing.assert_allclose(result_mlx, result_fast, atol=1e-2, rtol=1e-2)
