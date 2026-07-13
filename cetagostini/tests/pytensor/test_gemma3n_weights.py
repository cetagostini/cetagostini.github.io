"""Tests for gemma3n_weights module.

Synthetic unit tests run without the real artifact (temporary safetensors
files and fixtures), plus gated real-snapshot tests that require the
``GEMMA3N_SNAPSHOT`` environment variable pointing to the snapshot directory.

Covers:
- BF16 → float32 round-trip
- Affine-4 dequantization cross-checked against ``mx.dequantize``
- Row gather (duplicates, order, boundaries)
- Per-layer ID substitution (boundary, above, below)
- Vocab chunk iteration (tail handling)
- Config parsing and assertions
- Layer manifest generation (35 rows, key counts)
- Key completeness validation
- Per-layer weight loading (synthetic + real)
- Embedding row loading (synthetic + real)
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from cetagostini.utils.pytensor.gemma3n_weights import (
    BITS,
    GLOBAL_MODULE_SPECS,
    GROUP_SIZE,
    Gemma3nTextConfig,
    Gemma3nWeightLoader,
    LAYER_MODULE_SPECS,
    LayerSignature,
    ModuleSignature,
    NUM_LAYERS,
    PER_LAYER_VOCAB_BOUNDARY,
    PREFIX,
    TensorInfo,
    bf16_to_float32,
    build_layer_manifest,
    dequantize_affine4,
    gather_rows,
    parse_safetensors_header,
    parse_text_config,
    substitute_per_layer_ids,
    vocab_chunks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SNAPSHOT_DIR = os.environ.get("GEMMA3N_SNAPSHOT", "")


def _float32_to_bf16(arr: np.ndarray) -> bytes:
    """Convert float32 array to raw BF16 bytes (truncate lower 16 bits)."""
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


def _write_safetensors(path: Path, tensors: dict[str, tuple[np.ndarray, str]]) -> None:
    """Write a minimal safetensors file.

    Parameters
    ----------
    path : Path
        Output file path.
    tensors : dict
        Mapping from key to ``(array, dtype_str)`` where dtype_str is
        ``"BF16"`` or ``"U32"``.
    """
    header: dict[str, Any] = {}
    data_chunks: list[bytes] = []
    offset = 0
    for key in sorted(tensors.keys()):
        arr, dtype_str = tensors[key]
        if dtype_str == "BF16":
            raw = _float32_to_bf16(arr.astype(np.float32))
        elif dtype_str == "U32":
            raw = np.ascontiguousarray(arr, dtype=np.uint32).tobytes()
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        header[key] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data_chunks.append(raw)
        offset += len(raw)
    header_json = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for chunk in data_chunks:
            f.write(chunk)


def _make_quantized_triplet(
    out_features: int,
    in_features: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic quantized triplet.

    Returns ``(weight_u32, scales_f32, biases_f32)`` where weight is
    already in U32 packed form and scales/biases are float32 (will be
    stored as BF16 in the safetensors).
    """
    assert in_features % GROUP_SIZE == 0
    assert in_features % 8 == 0
    n_groups = in_features // GROUP_SIZE

    # Generate random nibbles (0-15)
    nibbles = rng.integers(0, 16, size=(out_features, in_features), dtype=np.uint8)
    scales = rng.uniform(0.5, 2.0, size=(out_features, n_groups)).astype(np.float32)
    biases = rng.uniform(-1.0, 1.0, size=(out_features, n_groups)).astype(np.float32)

    # Pack nibbles into U32 (low nibble first)
    in_packed = in_features // 8
    weight_u32 = np.zeros((out_features, in_packed), dtype=np.uint32)
    for w_idx in range(in_packed):
        base = w_idx * 8
        word = np.zeros(out_features, dtype=np.uint32)
        for n in range(8):
            word |= nibbles[:, base + n].astype(np.uint32) << (4 * n)
        weight_u32[:, w_idx] = word

    return weight_u32, scales, biases


class ByteCountingMemmap:
    """Wrapper around ``np.memmap`` that counts bytes read via ``__getitem__``.

    Used in tests to verify that row-sliced reads are bounded and do not
    materialize the full embedding table.

    Attributes
    ----------
    total_bytes_read : int
        Cumulative bytes returned through ``__getitem__``.
    read_count : int
        Number of ``__getitem__`` calls.
    """

    def __init__(self, mmap: np.memmap) -> None:
        self._mmap = mmap
        self.total_bytes_read: int = 0
        self.read_count: int = 0

    def __getitem__(self, key: Any) -> np.ndarray:
        result = self._mmap[key]
        self.total_bytes_read += result.nbytes
        self.read_count += 1
        return result


# ---------------------------------------------------------------------------
# BF16 → float32 tests
# ---------------------------------------------------------------------------


class TestBf16ToFloat32:
    """Tests for :func:`bf16_to_float32`."""

    def test_zero(self) -> None:
        raw = np.zeros(4, dtype=np.uint16).tobytes()
        result = bf16_to_float32(raw, (4,))
        np.testing.assert_array_equal(result, np.zeros(4, dtype=np.float32))

    def test_one(self) -> None:
        # BF16 for 1.0 is 0x3F80
        u16 = np.array([0x3F80], dtype=np.uint16)
        result = bf16_to_float32(u16.tobytes(), (1,))
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_round_trip(self) -> None:
        vals = np.array([0.0, 1.0, -1.0, 0.5, 100.0], dtype=np.float32)
        raw = _float32_to_bf16(vals)
        result = bf16_to_float32(raw, vals.shape)
        # BF16 has ~3 decimal digits of precision
        np.testing.assert_allclose(result, vals, atol=0.5)

    def test_shape_2d(self) -> None:
        vals = np.ones((3, 4), dtype=np.float32) * 2.0
        raw = _float32_to_bf16(vals)
        result = bf16_to_float32(raw, (3, 4))
        assert result.shape == (3, 4)
        assert result.flags["C_CONTIGUOUS"]
        np.testing.assert_allclose(result, 2.0, atol=0.02)

    def test_c_contiguous(self) -> None:
        raw = np.zeros(8, dtype=np.uint16).tobytes()
        result = bf16_to_float32(raw, (2, 4))
        assert result.flags["C_CONTIGUOUS"]
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Affine-4 dequantization tests
# ---------------------------------------------------------------------------


class TestDequantizeAffine4:
    """Tests for :func:`dequantize_affine4`."""

    def test_zeros(self) -> None:
        """All-zero weights should dequantize to biases."""
        out, in_ = 2, 64
        w = np.zeros((out, in_ // 8), dtype=np.uint32)
        s = np.ones((out, in_ // GROUP_SIZE), dtype=np.float32)
        b = np.full((out, in_ // GROUP_SIZE), 0.5, dtype=np.float32)
        # Store scales/biases as BF16 uint16
        s_bf16 = np.frombuffer(_float32_to_bf16(s), dtype=np.uint16).reshape(s.shape)
        b_bf16 = np.frombuffer(_float32_to_bf16(b), dtype=np.uint16).reshape(b.shape)
        result = dequantize_affine4(w, s_bf16, b_bf16)
        assert result.shape == (out, in_)
        # nibble=0, result = 0 * 1.0 + 0.5 = 0.5
        np.testing.assert_allclose(result, 0.5, atol=0.02)

    def test_nibble_ordering(self) -> None:
        """Verify low-nibble-first ordering matches mx.dequantize."""
        # 0x76543210 → nibbles [0, 1, 2, 3, 4, 5, 6, 7]
        w = np.zeros((1, 8), dtype=np.uint32)
        w[0, 0] = 0x76543210
        s = np.ones((1, 1), dtype=np.float32)
        b = np.zeros((1, 1), dtype=np.float32)
        s_bf16 = np.frombuffer(_float32_to_bf16(s), dtype=np.uint16).reshape(s.shape)
        b_bf16 = np.frombuffer(_float32_to_bf16(b), dtype=np.uint16).reshape(b.shape)
        result = dequantize_affine4(w, s_bf16, b_bf16)
        expected_first8 = np.arange(8, dtype=np.float32)
        np.testing.assert_array_equal(result[0, :8], expected_first8)

    def test_scale_and_bias(self) -> None:
        """Verify dequant formula: result = nibble * scale + bias."""
        w = np.zeros((1, 8), dtype=np.uint32)
        w[0, 0] = 0x76543210  # nibbles 0..7
        s = np.full((1, 1), 2.0, dtype=np.float32)
        b = np.full((1, 1), 1.0, dtype=np.float32)
        s_bf16 = np.frombuffer(_float32_to_bf16(s), dtype=np.uint16).reshape(s.shape)
        b_bf16 = np.frombuffer(_float32_to_bf16(b), dtype=np.uint16).reshape(b.shape)
        result = dequantize_affine4(w, s_bf16, b_bf16)
        expected = np.arange(8, dtype=np.float32) * 2.0 + 1.0
        np.testing.assert_allclose(result[0, :8], expected, atol=0.02)

    def test_multi_group(self) -> None:
        """Test with multiple groups (in_features > group_size)."""
        out, in_ = 2, 128  # 2 groups
        rng = np.random.default_rng(42)
        w, s, b = _make_quantized_triplet(out, in_, rng)
        s_bf16 = np.frombuffer(_float32_to_bf16(s), dtype=np.uint16).reshape(s.shape)
        b_bf16 = np.frombuffer(_float32_to_bf16(b), dtype=np.uint16).reshape(b.shape)
        result = dequantize_affine4(w, s_bf16, b_bf16)
        assert result.shape == (out, in_)
        assert result.flags["C_CONTIGUOUS"]
        assert np.all(np.isfinite(result))

    def test_c_contiguous_output(self) -> None:
        w = np.zeros((3, 8), dtype=np.uint32)
        s = np.ones((3, 1), dtype=np.float32)
        b = np.zeros((3, 1), dtype=np.float32)
        s_bf16 = np.frombuffer(_float32_to_bf16(s), dtype=np.uint16).reshape(s.shape)
        b_bf16 = np.frombuffer(_float32_to_bf16(b), dtype=np.uint16).reshape(b.shape)
        result = dequantize_affine4(w, s_bf16, b_bf16)
        assert result.flags["C_CONTIGUOUS"]
        assert result.dtype == np.float32

    def test_group_size_larger_than_input_raises(self) -> None:
        weight = np.zeros((1, 8), dtype=np.uint32)
        scales = np.zeros((1, 1), dtype=np.uint16)
        biases = np.zeros((1, 1), dtype=np.uint16)

        with pytest.raises(ValueError, match="divide"):
            dequantize_affine4(weight, scales, biases, group_size=128)

    def test_scale_group_shape_mismatch_raises(self) -> None:
        weight = np.zeros((2, 16), dtype=np.uint32)
        scales = np.zeros((2, 1), dtype=np.uint16)
        biases = np.zeros((2, 2), dtype=np.uint16)

        with pytest.raises(ValueError, match="scales shape"):
            dequantize_affine4(weight, scales, biases)

    def test_bias_group_shape_mismatch_raises(self) -> None:
        weight = np.zeros((2, 16), dtype=np.uint32)
        scales = np.zeros((2, 2), dtype=np.uint16)
        biases = np.zeros((2, 1), dtype=np.uint16)

        with pytest.raises(ValueError, match="biases shape"):
            dequantize_affine4(weight, scales, biases)


class TestDequantizeVsMlx:
    """Cross-check dequantize_affine4 against mx.dequantize."""

    @pytest.fixture(autouse=True)
    def _skip_no_mlx(self) -> None:
        pytest.importorskip("mlx.core")

    def test_synthetic_small(self) -> None:
        """Small synthetic matrix: compare numpy dequant vs mx.dequantize."""
        import mlx.core as mx

        rng = np.random.default_rng(123)
        out, in_ = 4, 64
        w_np, s_np, b_np = _make_quantized_triplet(out, in_, rng)
        s_bf16 = np.frombuffer(_float32_to_bf16(s_np), dtype=np.uint16).reshape(s_np.shape)
        b_bf16 = np.frombuffer(_float32_to_bf16(b_np), dtype=np.uint16).reshape(b_np.shape)

        # Our numpy dequant
        np_result = dequantize_affine4(w_np, s_bf16, b_bf16)

        # MLX dequant (uses BF16 scales/biases internally)
        w_mx = mx.array(w_np.tolist(), dtype=mx.uint32)
        # MLX expects BF16 scales/biases — convert through float32
        s_mx = mx.array(s_np.tolist()).astype(mx.bfloat16)
        b_mx = mx.array(b_np.tolist()).astype(mx.bfloat16)
        mx_result = np.array(
            mx.dequantize(w_mx, s_mx, b_mx, group_size=64, bits=4).astype(mx.float32)
        )

        np.testing.assert_allclose(np_result, mx_result, atol=1e-2, rtol=1e-2)

    def test_synthetic_medium(self) -> None:
        """Medium matrix with multiple groups."""
        import mlx.core as mx

        rng = np.random.default_rng(456)
        out, in_ = 8, 256  # 4 groups
        w_np, s_np, b_np = _make_quantized_triplet(out, in_, rng)
        s_bf16 = np.frombuffer(_float32_to_bf16(s_np), dtype=np.uint16).reshape(s_np.shape)
        b_bf16 = np.frombuffer(_float32_to_bf16(b_np), dtype=np.uint16).reshape(b_np.shape)

        np_result = dequantize_affine4(w_np, s_bf16, b_bf16)

        w_mx = mx.array(w_np.tolist(), dtype=mx.uint32)
        s_mx = mx.array(s_np.tolist()).astype(mx.bfloat16)
        b_mx = mx.array(b_np.tolist()).astype(mx.bfloat16)
        mx_result = np.array(
            mx.dequantize(w_mx, s_mx, b_mx, group_size=64, bits=4).astype(mx.float32)
        )

        np.testing.assert_allclose(np_result, mx_result, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Row gather tests
# ---------------------------------------------------------------------------


class TestGatherRows:
    """Tests for :func:`gather_rows`."""

    def test_basic(self) -> None:
        arr = np.arange(20, dtype=np.float32).reshape(5, 4)
        result = gather_rows(arr, [0, 2, 4])
        expected = arr[[0, 2, 4]]
        np.testing.assert_array_equal(result, expected)

    def test_duplicates(self) -> None:
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = gather_rows(arr, [1, 1, 0])
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result[0], arr[1])
        np.testing.assert_array_equal(result[1], arr[1])
        np.testing.assert_array_equal(result[2], arr[0])

    def test_order_preserved(self) -> None:
        arr = np.arange(20, dtype=np.float32).reshape(5, 4)
        result = gather_rows(arr, [4, 0, 3, 1])
        np.testing.assert_array_equal(result[0], arr[4])
        np.testing.assert_array_equal(result[1], arr[0])
        np.testing.assert_array_equal(result[2], arr[3])
        np.testing.assert_array_equal(result[3], arr[1])

    def test_c_contiguous(self) -> None:
        arr = np.arange(20, dtype=np.float32).reshape(5, 4)
        result = gather_rows(arr, [3, 1])
        assert result.flags["C_CONTIGUOUS"]
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Per-layer ID substitution tests
# ---------------------------------------------------------------------------


class TestSubstitutePerLayerIds:
    """Tests for :func:`substitute_per_layer_ids`."""

    def test_below_boundary(self) -> None:
        ids = np.array([0, 100, 262_143], dtype=np.int64)
        result = substitute_per_layer_ids(ids)
        np.testing.assert_array_equal(result, ids)

    def test_at_boundary(self) -> None:
        ids = np.array([262_144], dtype=np.int64)
        result = substitute_per_layer_ids(ids)
        np.testing.assert_array_equal(result, [0])

    def test_above_boundary(self) -> None:
        ids = np.array([262_145, 262_272, 262_400], dtype=np.int64)
        result = substitute_per_layer_ids(ids)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_mixed(self) -> None:
        ids = np.array([5, 262_144, 100, 262_273], dtype=np.int64)
        result = substitute_per_layer_ids(ids)
        np.testing.assert_array_equal(result, [5, 0, 100, 0])

    def test_does_not_mutate_input(self) -> None:
        ids = np.array([262_144, 5], dtype=np.int64)
        original = ids.copy()
        substitute_per_layer_ids(ids)
        np.testing.assert_array_equal(ids, original)


# ---------------------------------------------------------------------------
# Vocab chunk iteration tests
# ---------------------------------------------------------------------------


class TestVocabChunks:
    """Tests for :func:`vocab_chunks`."""

    def test_exact_division(self) -> None:
        chunks = list(vocab_chunks(100, 25))
        assert chunks == [(0, 25), (25, 50), (50, 75), (75, 100)]

    def test_remainder(self) -> None:
        chunks = list(vocab_chunks(10, 3))
        assert chunks == [(0, 3), (3, 6), (6, 9), (9, 10)]

    def test_single_chunk(self) -> None:
        chunks = list(vocab_chunks(5, 100))
        assert chunks == [(0, 5)]

    def test_tail_chunk(self) -> None:
        """Last chunk should be smaller when vocab_size is not divisible."""
        chunks = list(vocab_chunks(262_400, 4096))
        assert chunks[0] == (0, 4096)
        assert chunks[-1][1] == 262_400
        # Verify total coverage
        total = sum(end - start for start, end in chunks)
        assert total == 262_400

    @pytest.mark.parametrize("chunk_size", [0, -1])
    def test_invalid_chunk_size_raises(self, chunk_size: int) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            list(vocab_chunks(10, chunk_size))

    def test_negative_vocab_size_raises(self) -> None:
        with pytest.raises(ValueError, match="vocab_size"):
            list(vocab_chunks(-1, 4))


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestGemma3nTextConfig:
    """Tests for :class:`Gemma3nTextConfig` and :func:`parse_text_config`."""

    def test_default_values(self) -> None:
        cfg = Gemma3nTextConfig()
        assert cfg.vocab_size == 262_400
        assert cfg.hidden_size == 2048
        assert cfg.num_hidden_layers == 35
        assert cfg.head_dim == 256
        assert len(cfg.layer_types) == 35
        assert len(cfg.activation_sparsity_pattern) == 35

    def test_frozen(self) -> None:
        cfg = Gemma3nTextConfig()
        with pytest.raises(AttributeError):
            cfg.vocab_size = 100  # type: ignore[misc]

    def test_layer_types_pattern(self) -> None:
        cfg = Gemma3nTextConfig()
        # Every 5th layer (4, 9, 14, ...) should be full_attention
        for i, lt in enumerate(cfg.layer_types):
            if (i + 1) % 5 == 0:
                assert lt == "full_attention", f"Layer {i} should be full_attention"
            else:
                assert lt == "sliding_attention", f"Layer {i} should be sliding_attention"

    def test_parse_from_file(self, tmp_path: Path) -> None:
        """Parse a minimal config.json and verify the result."""
        config = {
            "text_config": {
                "vocab_size": 262400,
                "vocab_size_per_layer_input": 262144,
                "hidden_size": 2048,
                "hidden_size_per_layer_input": 256,
                "intermediate_size": 16384,
                "num_hidden_layers": 35,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "max_position_embeddings": 32768,
                "sliding_window": 512,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "rope_local_base_freq": 10000.0,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "hidden_activation": "gelu_pytorch_tanh",
                "final_logit_softcapping": 30.0,
                "altup_active_idx": 0,
                "altup_coef_clip": 120.0,
                "altup_correct_scale": True,
                "altup_lr_multiplier": 1.0,
                "altup_num_inputs": 4,
                "laurel_rank": 64,
                "num_kv_shared_layers": 15,
                "query_pre_attn_scalar": 256,
                "layer_types": list(Gemma3nTextConfig().layer_types),
                "activation_sparsity_pattern": list(Gemma3nTextConfig().activation_sparsity_pattern),
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        cfg = parse_text_config(config_path)
        assert cfg.vocab_size == 262_400
        assert cfg.num_hidden_layers == 35
        assert cfg.hidden_size == 2048

    def test_parse_requires_text_config_object(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model_type": "gemma3n"}))

        with pytest.raises(ValueError, match="text_config object"):
            parse_text_config(config_path)

    def test_parse_rejects_non_integer_layer_count(self, tmp_path: Path) -> None:
        config = {
            "text_config": {
                "vocab_size": 262400,
                "vocab_size_per_layer_input": 262144,
                "hidden_size": 2048,
                "hidden_size_per_layer_input": 256,
                "intermediate_size": 16384,
                "num_hidden_layers": 35.0,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "max_position_embeddings": 32768,
                "sliding_window": 512,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "rope_local_base_freq": 10000.0,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "hidden_activation": "gelu_pytorch_tanh",
                "final_logit_softcapping": 30.0,
                "altup_active_idx": 0,
                "altup_coef_clip": 120.0,
                "altup_correct_scale": True,
                "altup_lr_multiplier": 1.0,
                "altup_num_inputs": 4,
                "laurel_rank": 64,
                "num_kv_shared_layers": 15,
                "query_pre_attn_scalar": 256,
                "layer_types": list(Gemma3nTextConfig().layer_types),
                "activation_sparsity_pattern": list(
                    Gemma3nTextConfig().activation_sparsity_pattern
                ),
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="integer text_config fields"):
            parse_text_config(config_path)


# ---------------------------------------------------------------------------
# Layer manifest tests
# ---------------------------------------------------------------------------


class TestLayerManifest:
    """Tests for :func:`build_layer_manifest`."""

    def test_35_rows(self) -> None:
        cfg = Gemma3nTextConfig()
        manifest = build_layer_manifest(cfg)
        assert len(manifest) == 35

    def test_layer_indices(self) -> None:
        cfg = Gemma3nTextConfig()
        manifest = build_layer_manifest(cfg)
        for i, sig in enumerate(manifest):
            assert sig.layer_idx == i

    def test_layer_types(self) -> None:
        cfg = Gemma3nTextConfig()
        manifest = build_layer_manifest(cfg)
        assert manifest[0].layer_type == "sliding_attention"
        assert manifest[4].layer_type == "full_attention"
        assert manifest[34].layer_type == "full_attention"

    def test_module_count(self) -> None:
        cfg = Gemma3nTextConfig()
        manifest = build_layer_manifest(cfg)
        for sig in manifest:
            assert len(sig.modules) == len(LAYER_MODULE_SPECS)

    def test_key_count_per_layer(self) -> None:
        """Each layer should have exactly 48 safetensor keys."""
        cfg = Gemma3nTextConfig()
        manifest = build_layer_manifest(cfg)
        for sig in manifest:
            assert sig.total_keys == 48, f"Layer {sig.layer_idx} has {sig.total_keys} keys"

    def test_quantized_modules_have_triplet(self) -> None:
        cfg = Gemma3nTextConfig()
        manifest = build_layer_manifest(cfg)
        for sig in manifest:
            for mod in sig.modules:
                if mod.is_quantized:
                    assert len(mod.logical_keys) == 1

    def test_all_suffixes_present(self) -> None:
        cfg = Gemma3nTextConfig()
        manifest = build_layer_manifest(cfg)
        expected_suffixes = {s for s, *_ in LAYER_MODULE_SPECS}
        for sig in manifest:
            actual_suffixes = {m.suffix for m in sig.modules}
            assert actual_suffixes == expected_suffixes

    def test_exactly_four_template_keys(self) -> None:
        """Manifest must produce exactly 4 distinct template keys."""
        cfg = Gemma3nTextConfig()
        manifest = build_layer_manifest(cfg)
        keys = {sig.template_key for sig in manifest}
        assert keys == {
            "sliding_sparse",
            "full_sparse",
            "sliding_dense",
            "full_dense",
        }

    def test_attention_kind(self) -> None:
        cfg = Gemma3nTextConfig()
        manifest = build_layer_manifest(cfg)
        assert manifest[0].attention_kind == "sliding"
        assert manifest[4].attention_kind == "full"

    def test_sparsity_kind(self) -> None:
        cfg = Gemma3nTextConfig()
        manifest = build_layer_manifest(cfg)
        # Layers 0–9 are sparse (0.95), layers 10+ are dense (0.0)
        assert manifest[0].sparsity_kind == "sparse"
        assert manifest[9].sparsity_kind == "sparse"
        assert manifest[10].sparsity_kind == "dense"
        assert manifest[34].sparsity_kind == "dense"

    def test_rope_base(self) -> None:
        cfg = Gemma3nTextConfig()
        manifest = build_layer_manifest(cfg)
        # Sliding layers use rope_local_base_freq
        assert manifest[0].rope_base == cfg.rope_local_base_freq
        # Full layers use rope_theta
        assert manifest[4].rope_base == cfg.rope_theta


# ---------------------------------------------------------------------------
# Safetensors header parsing tests
# ---------------------------------------------------------------------------


class TestParseSafetensorsHeader:
    """Tests for :func:`parse_safetensors_header`."""

    def test_synthetic_file(self, tmp_path: Path) -> None:
        st_path = tmp_path / "test.safetensors"
        arr = np.ones((4, 8), dtype=np.float32)
        _write_safetensors(st_path, {"test.tensor": (arr, "BF16")})
        info = parse_safetensors_header(st_path)
        assert "test.tensor" in info
        assert info["test.tensor"].shape == (4, 8)
        assert info["test.tensor"].dtype == "BF16"

    def test_u32_tensor(self, tmp_path: Path) -> None:
        st_path = tmp_path / "test.safetensors"
        arr = np.array([[1, 2, 3]], dtype=np.uint32)
        _write_safetensors(st_path, {"quant.weight": (arr, "U32")})
        info = parse_safetensors_header(st_path)
        assert info["quant.weight"].dtype == "U32"
        assert info["quant.weight"].shape == (1, 3)

    def test_truncated_head_length_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "truncated.safetensors"
        path.write_bytes(b"short")

        with pytest.raises(ValueError, match="8-byte"):
            parse_safetensors_header(path)

    def test_shape_byte_mismatch_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "mismatch.safetensors"
        header = json.dumps({
            "tensor": {
                "dtype": "BF16",
                "shape": [3],
                "data_offsets": [0, 4],
            }
        }).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)

        with pytest.raises(ValueError, match="byte count"):
            parse_safetensors_header(path)


# ---------------------------------------------------------------------------
# Synthetic integration tests (no real artifact)
# ---------------------------------------------------------------------------


class TestSyntheticLoader:
    """Integration tests using a synthetic safetensors file."""

    @pytest.fixture()
    def synthetic_snapshot(self, tmp_path: Path) -> Path:
        """Create a minimal synthetic snapshot directory."""
        # Config
        config = {
            "text_config": {
                "vocab_size": 262400,
                "vocab_size_per_layer_input": 262144,
                "hidden_size": 2048,
                "hidden_size_per_layer_input": 256,
                "intermediate_size": 16384,
                "num_hidden_layers": 35,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "max_position_embeddings": 32768,
                "sliding_window": 512,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "rope_local_base_freq": 10000.0,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "hidden_activation": "gelu_pytorch_tanh",
                "final_logit_softcapping": 30.0,
                "altup_active_idx": 0,
                "altup_coef_clip": 120.0,
                "altup_correct_scale": True,
                "altup_lr_multiplier": 1.0,
                "altup_num_inputs": 4,
                "laurel_rank": 64,
                "num_kv_shared_layers": 15,
                "query_pre_attn_scalar": 256,
                "layer_types": list(Gemma3nTextConfig().layer_types),
                "activation_sparsity_pattern": list(Gemma3nTextConfig().activation_sparsity_pattern),
            }
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        # Build minimal safetensors with just layer 0 + global keys
        rng = np.random.default_rng(42)
        tensors: dict[str, tuple[np.ndarray, str]] = {}

        # Layer 0 quantized modules
        quant_modules_l0 = [
            ("self_attn.q_proj", 2048, 2048),
            ("self_attn.k_proj", 512, 2048),
            ("self_attn.v_proj", 512, 2048),
            ("self_attn.o_proj", 2048, 2048),
            ("mlp.gate_proj", 16384, 2048),
            ("mlp.up_proj", 16384, 2048),
            ("mlp.down_proj", 2048, 16384),
            ("altup.modality_router", 4, 2048),
            ("laurel.linear_left", 64, 2048),
            ("laurel.linear_right", 2048, 64),
            ("per_layer_input_gate", 256, 2048),
            ("per_layer_projection", 2048, 256),
        ]
        for suffix, out_f, in_f in quant_modules_l0:
            prefix = f"model.language_model.layers.0.{suffix}"
            w, s, b = _make_quantized_triplet(out_f, in_f, rng)
            tensors[f"{prefix}.weight"] = (w, "U32")
            tensors[f"{prefix}.scales"] = (s, "BF16")
            tensors[f"{prefix}.biases"] = (b, "BF16")

        # Layer 0 unquantized modules
        unquant_l0 = [
            ("self_attn.q_norm", (256,), False),
            ("self_attn.k_norm", (256,), False),
            ("input_layernorm", (2048,), False),
            ("post_attention_layernorm", (2048,), False),
            ("pre_feedforward_layernorm", (2048,), False),
            ("post_feedforward_layernorm", (2048,), False),
            ("altup.correct_output_scale", (2048,), True),
            ("altup.correction_coefs", (4, 4), False),
            ("altup.prediction_coefs", (16, 4), False),
            ("altup.router_norm", (2048,), False),
            ("laurel.post_laurel_norm", (2048,), False),
            ("post_per_layer_input_norm", (2048,), False),
        ]
        for suffix, shape, bare_key in unquant_l0:
            prefix = f"model.language_model.layers.0.{suffix}"
            arr = rng.standard_normal(shape).astype(np.float32)
            key = prefix if bare_key else f"{prefix}.weight"
            tensors[key] = (arr, "BF16")

        # Global modules
        # embed_tokens
        w, s, b = _make_quantized_triplet(256, 2048, rng)
        tensors["model.language_model.embed_tokens.weight"] = (w, "U32")
        tensors["model.language_model.embed_tokens.scales"] = (s, "BF16")
        tensors["model.language_model.embed_tokens.biases"] = (b, "BF16")

        # embed_tokens_per_layer
        w, s, b = _make_quantized_triplet(256, 8960, rng)
        tensors["model.language_model.embed_tokens_per_layer.weight"] = (w, "U32")
        tensors["model.language_model.embed_tokens_per_layer.scales"] = (s, "BF16")
        tensors["model.language_model.embed_tokens_per_layer.biases"] = (b, "BF16")

        # norm
        tensors["model.language_model.norm.weight"] = (
            rng.standard_normal((2048,)).astype(np.float32), "BF16"
        )

        # per_layer_model_projection
        w, s, b = _make_quantized_triplet(8960, 2048, rng)
        tensors["model.language_model.per_layer_model_projection.weight"] = (w, "U32")
        tensors["model.language_model.per_layer_model_projection.scales"] = (s, "BF16")
        tensors["model.language_model.per_layer_model_projection.biases"] = (b, "BF16")

        # per_layer_projection_norm
        tensors["model.language_model.per_layer_projection_norm.weight"] = (
            rng.standard_normal((256,)).astype(np.float32), "BF16"
        )

        # altup_projections
        for i in range(3):
            w, s, b = _make_quantized_triplet(2048, 2048, rng)
            tensors[f"model.language_model.altup_projections.{i}.weight"] = (w, "U32")
            tensors[f"model.language_model.altup_projections.{i}.scales"] = (s, "BF16")
            tensors[f"model.language_model.altup_projections.{i}.biases"] = (b, "BF16")

        # altup_unembed_projections
        for i in range(3):
            w, s, b = _make_quantized_triplet(2048, 2048, rng)
            tensors[f"model.language_model.altup_unembed_projections.{i}.weight"] = (w, "U32")
            tensors[f"model.language_model.altup_unembed_projections.{i}.scales"] = (s, "BF16")
            tensors[f"model.language_model.altup_unembed_projections.{i}.biases"] = (b, "BF16")

        _write_safetensors(tmp_path / "model.safetensors", tensors)
        return tmp_path

    def test_loader_creation(self, synthetic_snapshot: Path) -> None:
        with Gemma3nWeightLoader.from_snapshot(synthetic_snapshot) as loader:
            assert loader.config.num_hidden_layers == 35
            assert len(loader.manifest) == 35
            assert len(loader.tensor_info) > 0

    def test_load_layer_0(self, synthetic_snapshot: Path) -> None:
        with Gemma3nWeightLoader.from_snapshot(synthetic_snapshot) as loader:
            layer = loader.load_layer(0)
            assert isinstance(layer, dict)
            assert len(layer) == len(LAYER_MODULE_SPECS)
            for key, arr in layer.items():
                assert arr.dtype == np.float32
                assert arr.flags["C_CONTIGUOUS"]
                assert np.all(np.isfinite(arr))

    def test_layer_0_shapes(self, synthetic_snapshot: Path) -> None:
        with Gemma3nWeightLoader.from_snapshot(synthetic_snapshot) as loader:
            layer = loader.load_layer(0)
            assert layer["self_attn.q_proj"].shape == (2048, 2048)
            assert layer["self_attn.k_proj"].shape == (512, 2048)
            assert layer["mlp.gate_proj"].shape == (16384, 2048)
            assert layer["mlp.down_proj"].shape == (2048, 16384)
            assert layer["input_layernorm"].shape == (2048,)
            assert layer["self_attn.q_norm"].shape == (256,)

    def test_load_global_norm(self, synthetic_snapshot: Path) -> None:
        with Gemma3nWeightLoader.from_snapshot(synthetic_snapshot) as loader:
            norm = loader.load_global("norm")
            assert norm.shape == (2048,)
            assert norm.dtype == np.float32

    def test_load_global_quantized(self, synthetic_snapshot: Path) -> None:
        with Gemma3nWeightLoader.from_snapshot(synthetic_snapshot) as loader:
            proj = loader.load_global("per_layer_model_projection")
            assert proj.shape == (8960, 2048)
            assert proj.dtype == np.float32

    def test_context_manager(self, synthetic_snapshot: Path) -> None:
        loader = Gemma3nWeightLoader.from_snapshot(synthetic_snapshot)
        assert loader._mmap is not None
        loader.close()
        assert loader._mmap is None

    def test_input_embedding_no_full_materialization(
        self, synthetic_snapshot: Path
    ) -> None:
        """load_input_embedding_rows must NOT call _dequantize_module."""
        with Gemma3nWeightLoader.from_snapshot(synthetic_snapshot) as loader:
            with patch.object(
                loader, "_dequantize_module",
                side_effect=AssertionError("_dequantize_module must not be called"),
            ):
                rows = loader.load_input_embedding_rows([0, 1, 2])
                assert rows.shape == (3, 2048)
                assert rows.dtype == np.float32
                assert np.all(np.isfinite(rows))

    def test_input_embedding_negative_id_raises(
        self, synthetic_snapshot: Path
    ) -> None:
        with Gemma3nWeightLoader.from_snapshot(synthetic_snapshot) as loader:
            with pytest.raises(ValueError, match="Negative"):
                loader.load_input_embedding_rows([-1, 0, 1])

    def test_input_embedding_oob_id_raises(
        self, synthetic_snapshot: Path
    ) -> None:
        with Gemma3nWeightLoader.from_snapshot(synthetic_snapshot) as loader:
            with pytest.raises(ValueError, match="out of bounds"):
                loader.load_input_embedding_rows([262400])

    def test_input_embedding_empty(self, synthetic_snapshot: Path) -> None:
        with Gemma3nWeightLoader.from_snapshot(synthetic_snapshot) as loader:
            rows = loader.load_input_embedding_rows([])
            assert rows.shape == (0, 2048)
            assert rows.dtype == np.float32

    def test_input_embedding_duplicates_row_sliced(
        self, synthetic_snapshot: Path
    ) -> None:
        """Duplicates must produce identical rows after row-sliced dequant."""
        with Gemma3nWeightLoader.from_snapshot(synthetic_snapshot) as loader:
            rows = loader.load_input_embedding_rows([5, 5, 0, 5])
            assert rows.shape == (4, 2048)
            np.testing.assert_array_equal(rows[0], rows[1])
            np.testing.assert_array_equal(rows[0], rows[3])


# ---------------------------------------------------------------------------
# Real snapshot tests (gated by GEMMA3N_SNAPSHOT env var)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SNAPSHOT_DIR,
    reason="Set GEMMA3N_SNAPSHOT to the snapshot directory to run real tests",
)
class TestRealSnapshot:
    """Tests against the real Gemma3n snapshot."""

    @pytest.fixture()
    def loader(self) -> Gemma3nWeightLoader:
        loader = Gemma3nWeightLoader.from_snapshot(SNAPSHOT_DIR)
        yield loader
        loader.close()

    def test_config(self, loader: Gemma3nWeightLoader) -> None:
        cfg = loader.config
        assert cfg.vocab_size == 262_400
        assert cfg.hidden_size == 2048
        assert cfg.num_hidden_layers == 35
        assert cfg.head_dim == 256

    def test_key_completeness(self, loader: Gemma3nWeightLoader) -> None:
        loader.validate_keys()

    def test_tensor_count(self, loader: Gemma3nWeightLoader) -> None:
        assert len(loader.tensor_info) == 1709

    def test_load_layer_0(self, loader: Gemma3nWeightLoader) -> None:
        layer = loader.load_layer(0)
        assert len(layer) == len(LAYER_MODULE_SPECS)
        for key, arr in layer.items():
            assert arr.dtype == np.float32
            assert np.all(np.isfinite(arr))

    def test_load_layer_34(self, loader: Gemma3nWeightLoader) -> None:
        layer = loader.load_layer(34)
        assert len(layer) == len(LAYER_MODULE_SPECS)

    def test_layer_0_q_proj_shape(self, loader: Gemma3nWeightLoader) -> None:
        layer = loader.load_layer(0)
        assert layer["self_attn.q_proj"].shape == (2048, 2048)

    def test_input_embedding_rows(self, loader: Gemma3nWeightLoader) -> None:
        rows = loader.load_input_embedding_rows([0, 1, 2, 100])
        assert rows.shape == (4, 2048)
        assert rows.dtype == np.float32
        assert np.all(np.isfinite(rows))
