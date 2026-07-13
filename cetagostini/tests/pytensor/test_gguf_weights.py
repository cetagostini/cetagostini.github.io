"""Tests for gguf_weights module.

Synthetic unit tests run without the real artifact (mocked GGUF reader),
plus gated real-artifact tests that require the ``SMOLLM2_GGUF`` environment
variable pointing to the GGUF file.

Covers:
- Artifact validation (size, hash, metadata)
- Tensor inventory validation
- RoPE permutation undo
- Tensor transformation and dequantization
- Weight loading (synthetic + real)
- Sanitized report generation
- CLI argument parsing
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from cetagostini.utils.pytensor.gguf_weights import (
    EXPECTED_ARCHITECTURE,
    EXPECTED_FILENAME,
    EXPECTED_GGUF_VERSION,
    EXPECTED_QUANT_TYPE_COUNTS,
    EXPECTED_REPO,
    EXPECTED_REVISION,
    EXPECTED_SHA256,
    EXPECTED_SIZE,
    EXPECTED_TENSOR_COUNT,
    SmolLM2Config,
    SmolLM2Weights,
    _build_expected_tensor_names,
    _extract_layer_index,
    _transform_tensor,
    _undo_rope_permutation,
    _validate_finite,
    atomic_write_json,
    build_inventory,
    build_manifest,
    compute_sha256,
    main,
    parse_args,
    sanitize_weights_report,
    validate_artifact,
)


def test_transform_linear_weight_to_graph_orientation(monkeypatch):
    """GGUF [out, in] matrices are returned as graph-ready [in, out]."""
    import sys
    from types import SimpleNamespace

    data = np.arange(8, dtype=np.float32).reshape(2, 4)
    monkeypatch.setitem(
        sys.modules,
        "gguf",
        SimpleNamespace(dequantize=lambda tensor_data, _tensor_type: tensor_data),
    )
    tensor = SimpleNamespace(
        name="blk.0.attn_v.weight",
        data=data,
        tensor_type=SimpleNamespace(name="F32"),
        shape=(4, 2),
    )
    config = SimpleNamespace(n_heads=1, n_kv_heads=1, head_dim=2)

    transformed = _transform_tensor(tensor, config, 0)

    assert transformed.shape == (4, 2)
    np.testing.assert_array_equal(transformed, data.T)
    assert transformed.flags.c_contiguous

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SMOLLM2_GGUF = os.environ.get("SMOLLM2_GGUF", "")


def _make_mock_reader(
    n_layers: int = 30,
    n_heads: int = 9,
    n_kv_heads: int = 3,
    head_dim: int = 64,
    hidden_size: int = 576,
    vocab_size: int = 49152,
    intermediate_size: int = 1536,
) -> Mock:
    """Create a mock GGUF reader with the expected tensor inventory."""
    reader = Mock()

    # Mock metadata fields
    version_field = Mock()
    version_field.parts = [EXPECTED_GGUF_VERSION]
    version_field.data = [0]

    tensor_count_field = Mock()
    tensor_count_field.parts = [EXPECTED_TENSOR_COUNT]
    tensor_count_field.data = [0]

    arch_field = Mock()
    arch_field.parts = [EXPECTED_ARCHITECTURE]
    arch_field.data = [0]

    def get_field(name: str) -> Mock | None:
        if name == "GGUF.version":
            return version_field
        elif name == "GGUF.tensor_count":
            return tensor_count_field
        elif name == "general.architecture":
            return arch_field
        return None

    reader.get_field = get_field

    # Mock tensors
    tensors = []

    # token_embd.weight
    token_embd = Mock()
    token_embd.name = "token_embd.weight"
    token_embd.shape = [hidden_size, vocab_size]
    token_embd.tensor_type = Mock()
    token_embd.tensor_type.name = "F32"
    token_embd.n_bytes = hidden_size * vocab_size * 4
    token_embd.n_elements = hidden_size * vocab_size
    token_embd.data = np.zeros((hidden_size, vocab_size), dtype=np.float32)
    tensors.append(token_embd)

    # output_norm.weight
    output_norm = Mock()
    output_norm.name = "output_norm.weight"
    output_norm.shape = [hidden_size]
    output_norm.tensor_type = Mock()
    output_norm.tensor_type.name = "F32"
    output_norm.n_bytes = hidden_size * 4
    output_norm.n_elements = hidden_size
    output_norm.data = np.ones(hidden_size, dtype=np.float32)
    tensors.append(output_norm)

    # Layer tensors
    for i in range(n_layers):
        # attn_norm
        attn_norm = Mock()
        attn_norm.name = f"blk.{i}.attn_norm.weight"
        attn_norm.shape = [hidden_size]
        attn_norm.tensor_type = Mock()
        attn_norm.tensor_type.name = "F32"
        attn_norm.n_bytes = hidden_size * 4
        attn_norm.n_elements = hidden_size
        attn_norm.data = np.ones(hidden_size, dtype=np.float32)
        tensors.append(attn_norm)

        # attn_q
        attn_q = Mock()
        attn_q.name = f"blk.{i}.attn_q.weight"
        attn_q.shape = [hidden_size, n_heads * head_dim]
        attn_q.tensor_type = Mock()
        attn_q.tensor_type.name = "Q5_0"
        attn_q.n_bytes = hidden_size * n_heads * head_dim // 2
        attn_q.n_elements = hidden_size * n_heads * head_dim
        attn_q.data = np.zeros((hidden_size, n_heads * head_dim), dtype=np.uint8)
        tensors.append(attn_q)

        # attn_k
        attn_k = Mock()
        attn_k.name = f"blk.{i}.attn_k.weight"
        attn_k.shape = [hidden_size, n_kv_heads * head_dim]
        attn_k.tensor_type = Mock()
        attn_k.tensor_type.name = "Q5_0"
        attn_k.n_bytes = hidden_size * n_kv_heads * head_dim // 2
        attn_k.n_elements = hidden_size * n_kv_heads * head_dim
        attn_k.data = np.zeros((hidden_size, n_kv_heads * head_dim), dtype=np.uint8)
        tensors.append(attn_k)

        # attn_v
        attn_v = Mock()
        attn_v.name = f"blk.{i}.attn_v.weight"
        attn_v.shape = [hidden_size, n_kv_heads * head_dim]
        attn_v.tensor_type = Mock()
        attn_v.tensor_type.name = "Q5_0"
        attn_v.n_bytes = hidden_size * n_kv_heads * head_dim // 2
        attn_v.n_elements = hidden_size * n_kv_heads * head_dim
        attn_v.data = np.zeros((hidden_size, n_kv_heads * head_dim), dtype=np.uint8)
        tensors.append(attn_v)

        # attn_output
        attn_output = Mock()
        attn_output.name = f"blk.{i}.attn_output.weight"
        attn_output.shape = [n_heads * head_dim, hidden_size]
        attn_output.tensor_type = Mock()
        attn_output.tensor_type.name = "Q5_0"
        attn_output.n_bytes = n_heads * head_dim * hidden_size // 2
        attn_output.n_elements = n_heads * head_dim * hidden_size
        attn_output.data = np.zeros((n_heads * head_dim, hidden_size), dtype=np.uint8)
        tensors.append(attn_output)

        # ffn_norm
        ffn_norm = Mock()
        ffn_norm.name = f"blk.{i}.ffn_norm.weight"
        ffn_norm.shape = [hidden_size]
        ffn_norm.tensor_type = Mock()
        ffn_norm.tensor_type.name = "F32"
        ffn_norm.n_bytes = hidden_size * 4
        ffn_norm.n_elements = hidden_size
        ffn_norm.data = np.ones(hidden_size, dtype=np.float32)
        tensors.append(ffn_norm)

        # ffn_gate
        ffn_gate = Mock()
        ffn_gate.name = f"blk.{i}.ffn_gate.weight"
        ffn_gate.shape = [hidden_size, intermediate_size]
        ffn_gate.tensor_type = Mock()
        ffn_gate.tensor_type.name = "Q5_0"
        ffn_gate.n_bytes = hidden_size * intermediate_size // 2
        ffn_gate.n_elements = hidden_size * intermediate_size
        ffn_gate.data = np.zeros((hidden_size, intermediate_size), dtype=np.uint8)
        tensors.append(ffn_gate)

        # ffn_up
        ffn_up = Mock()
        ffn_up.name = f"blk.{i}.ffn_up.weight"
        ffn_up.shape = [hidden_size, intermediate_size]
        ffn_up.tensor_type = Mock()
        ffn_up.tensor_type.name = "Q5_0"
        ffn_up.n_bytes = hidden_size * intermediate_size // 2
        ffn_up.n_elements = hidden_size * intermediate_size
        ffn_up.data = np.zeros((hidden_size, intermediate_size), dtype=np.uint8)
        tensors.append(ffn_up)

        # ffn_down
        ffn_down = Mock()
        ffn_down.name = f"blk.{i}.ffn_down.weight"
        ffn_down.shape = [intermediate_size, hidden_size]
        ffn_down.tensor_type = Mock()
        ffn_down.tensor_type.name = "Q5_0"
        ffn_down.n_bytes = intermediate_size * hidden_size // 2
        ffn_down.n_elements = intermediate_size * hidden_size
        ffn_down.data = np.zeros((intermediate_size, hidden_size), dtype=np.uint8)
        tensors.append(ffn_down)

    reader.tensors = tensors
    return reader


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestSmolLM2Config:
    """Tests for :class:`SmolLM2Config`."""

    def test_default_values(self) -> None:
        cfg = SmolLM2Config()
        assert cfg.vocab_size == 49152
        assert cfg.hidden_size == 576
        assert cfg.n_layers == 30
        assert cfg.n_heads == 9
        assert cfg.n_kv_heads == 3
        assert cfg.head_dim == 64
        assert cfg.intermediate_size == 1536
        assert cfg.context_length == 8192
        assert cfg.rms_eps == 1e-5
        assert cfg.rope_theta == 100000.0

    def test_frozen(self) -> None:
        cfg = SmolLM2Config()
        with pytest.raises(AttributeError):
            cfg.vocab_size = 100  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Artifact validation tests
# ---------------------------------------------------------------------------


class TestValidateArtifact:
    """Tests for :func:`validate_artifact`."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.gguf"
        with pytest.raises(FileNotFoundError, match="GGUF file not found"):
            validate_artifact(path, verify_hash=False)

    def test_wrong_size(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong_size.gguf"
        path.write_bytes(b"x" * 100)
        with pytest.raises(ValueError, match="Expected file size"):
            validate_artifact(path, verify_hash=False)

    def test_correct_size_no_hash(self, tmp_path: Path) -> None:
        path = tmp_path / "correct_size.gguf"
        path.write_bytes(b"x" * EXPECTED_SIZE)
        # Should not raise
        validate_artifact(path, verify_hash=False)


# ---------------------------------------------------------------------------
# RoPE permutation tests
# ---------------------------------------------------------------------------


class TestUndoRopePermutation:
    """Tests for :func:`_undo_rope_permutation`."""

    def test_identity_permutation(self) -> None:
        """Test that undoing a permutation restores the original order."""
        n_heads = 2
        head_dim = 4
        in_features = 8

        # Create a test array with distinct values
        original = np.arange(n_heads * head_dim * in_features, dtype=np.float32)
        original = original.reshape(n_heads * head_dim, in_features)

        # Apply the permutation (even indices first, then odd)
        permuted = np.empty_like(original)
        half_dim = head_dim // 2
        reshaped = original.reshape(n_heads, head_dim, in_features)
        permuted_reshaped = np.empty_like(reshaped)
        permuted_reshaped[:, 0::2, :] = reshaped[:, :half_dim, :]
        permuted_reshaped[:, 1::2, :] = reshaped[:, half_dim:, :]
        permuted = permuted_reshaped.reshape(n_heads * head_dim, in_features)

        # Undo the permutation
        result = _undo_rope_permutation(permuted, n_heads, head_dim)

        np.testing.assert_array_equal(result, original)

    def test_shape_preserved(self) -> None:
        n_heads = 4
        head_dim = 8
        in_features = 16
        arr = np.random.randn(n_heads * head_dim, in_features).astype(np.float32)
        result = _undo_rope_permutation(arr, n_heads, head_dim)
        assert result.shape == arr.shape

    def test_wrong_out_dimension_raises(self) -> None:
        n_heads = 2
        head_dim = 4
        arr = np.zeros((10, 8), dtype=np.float32)  # Wrong: should be 8
        with pytest.raises(ValueError, match="Expected out="):
            _undo_rope_permutation(arr, n_heads, head_dim)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestExtractLayerIndex:
    """Tests for :func:`_extract_layer_index`."""

    def test_layer_tensor(self) -> None:
        assert _extract_layer_index("blk.5.attn_q.weight") == 5
        assert _extract_layer_index("blk.0.ffn_norm.weight") == 0
        assert _extract_layer_index("blk.29.attn_output.weight") == 29

    def test_non_layer_tensor(self) -> None:
        assert _extract_layer_index("token_embd.weight") is None
        assert _extract_layer_index("output_norm.weight") is None

    def test_malformed_name(self) -> None:
        assert _extract_layer_index("blk.abc.attn_q.weight") is None
        assert _extract_layer_index("blk") is None


class TestBuildExpectedTensorNames:
    """Tests for :func:`_build_expected_tensor_names`."""

    def test_includes_global_tensors(self) -> None:
        cfg = SmolLM2Config()
        names = _build_expected_tensor_names(cfg)
        assert "token_embd.weight" in names
        assert "output_norm.weight" in names

    def test_includes_layer_tensors(self) -> None:
        cfg = SmolLM2Config()
        names = _build_expected_tensor_names(cfg)
        for i in range(cfg.n_layers):
            assert f"blk.{i}.attn_norm.weight" in names
            assert f"blk.{i}.attn_q.weight" in names
            assert f"blk.{i}.ffn_down.weight" in names

    def test_correct_count(self) -> None:
        cfg = SmolLM2Config()
        names = _build_expected_tensor_names(cfg)
        # 2 global + 9 per layer * 30 layers = 272
        assert len(names) == 272


class TestValidateFinite:
    """Tests for :func:`_validate_finite`."""

    def test_finite_array(self) -> None:
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # Should not raise
        _validate_finite(arr, "test")

    def test_nan_raises(self) -> None:
        arr = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite values"):
            _validate_finite(arr, "test")

    def test_inf_raises(self) -> None:
        arr = np.array([1.0, np.inf, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite values"):
            _validate_finite(arr, "test")


# ---------------------------------------------------------------------------
# Manifest building tests
# ---------------------------------------------------------------------------


class TestBuildInventory:
    """Tests for :func:`build_inventory`."""

    def test_inventory_structure(self) -> None:
        reader = _make_mock_reader(n_layers=2)
        inventory = build_inventory(reader)
        assert isinstance(inventory, list)
        assert len(inventory) > 0
        for item in inventory:
            assert "name" in item
            assert "shape" in item
            assert "dtype" in item
            assert "n_bytes" in item
            assert "n_elements" in item

    def test_inventory_includes_all_tensors(self) -> None:
        reader = _make_mock_reader(n_layers=2)
        inventory = build_inventory(reader)
        names = {item["name"] for item in inventory}
        assert "token_embd.weight" in names
        assert "output_norm.weight" in names
        assert "blk.0.attn_q.weight" in names
        assert "blk.1.ffn_down.weight" in names


class TestBuildManifest:
    """Tests for :func:`build_manifest`."""

    def test_manifest_structure(self) -> None:
        reader = _make_mock_reader(n_layers=2)
        manifest = build_manifest(reader)
        assert "architecture" in manifest
        assert "tensor_count" in manifest
        assert "quant_type_counts" in manifest
        assert "total_bytes" in manifest
        assert "total_elements" in manifest

    def test_manifest_values(self) -> None:
        reader = _make_mock_reader(n_layers=2)
        manifest = build_manifest(reader)
        assert manifest["architecture"] == EXPECTED_ARCHITECTURE
        assert manifest["tensor_count"] == len(reader.tensors)
        assert isinstance(manifest["quant_type_counts"], dict)
        assert manifest["total_bytes"] > 0
        assert manifest["total_elements"] > 0


# ---------------------------------------------------------------------------
# Sanitized report tests
# ---------------------------------------------------------------------------


class TestSanitizeWeightsReport:
    """Tests for :func:`sanitize_weights_report`."""

    def test_report_structure(self) -> None:
        reader = _make_mock_reader(n_layers=2)
        config = SmolLM2Config()
        token_embedding = np.zeros((config.vocab_size, config.hidden_size), dtype=np.float32)
        final_norm = np.ones(config.hidden_size, dtype=np.float32)
        layers = [
            {
                "attn_norm": np.ones(config.hidden_size, dtype=np.float32),
                "wq": np.zeros((config.n_heads * config.head_dim, config.hidden_size), dtype=np.float32),
                "wk": np.zeros((config.n_kv_heads * config.head_dim, config.hidden_size), dtype=np.float32),
                "wv": np.zeros((config.n_kv_heads * config.head_dim, config.hidden_size), dtype=np.float32),
                "wo": np.zeros((config.hidden_size, config.n_heads * config.head_dim), dtype=np.float32),
                "ffn_norm": np.ones(config.hidden_size, dtype=np.float32),
                "w_gate": np.zeros((config.intermediate_size, config.hidden_size), dtype=np.float32),
                "w_up": np.zeros((config.intermediate_size, config.hidden_size), dtype=np.float32),
                "w_down": np.zeros((config.hidden_size, config.intermediate_size), dtype=np.float32),
            }
            for _ in range(2)
        ]
        weights = SmolLM2Weights(
            config=config,
            token_embedding=token_embedding,
            layers=layers,
            final_norm=final_norm,
            reader=reader,
        )

        report = sanitize_weights_report(weights, Path("test.gguf"))

        assert "model_repo" in report
        assert "model_revision" in report
        assert "filename" in report
        assert "architecture" in report
        assert "gguf_version" in report
        assert "config" in report
        assert "manifest" in report
        assert "token_embedding" in report
        assert "final_norm" in report
        assert "layer_stats" in report

    def test_report_no_absolute_paths(self) -> None:
        reader = _make_mock_reader(n_layers=1)
        config = SmolLM2Config()
        token_embedding = np.zeros((config.vocab_size, config.hidden_size), dtype=np.float32)
        final_norm = np.ones(config.hidden_size, dtype=np.float32)
        layers = [
            {
                "attn_norm": np.ones(config.hidden_size, dtype=np.float32),
                "wq": np.zeros((config.n_heads * config.head_dim, config.hidden_size), dtype=np.float32),
                "wk": np.zeros((config.n_kv_heads * config.head_dim, config.hidden_size), dtype=np.float32),
                "wv": np.zeros((config.n_kv_heads * config.head_dim, config.hidden_size), dtype=np.float32),
                "wo": np.zeros((config.hidden_size, config.n_heads * config.head_dim), dtype=np.float32),
                "ffn_norm": np.ones(config.hidden_size, dtype=np.float32),
                "w_gate": np.zeros((config.intermediate_size, config.hidden_size), dtype=np.float32),
                "w_up": np.zeros((config.intermediate_size, config.hidden_size), dtype=np.float32),
                "w_down": np.zeros((config.hidden_size, config.intermediate_size), dtype=np.float32),
            }
        ]
        weights = SmolLM2Weights(
            config=config,
            token_embedding=token_embedding,
            layers=layers,
            final_norm=final_norm,
            reader=reader,
        )

        report = sanitize_weights_report(weights, Path("/absolute/path/to/test.gguf"))

        # Filename should be relative
        assert report["filename"] == "test.gguf"
        # No absolute paths in the report
        report_str = json.dumps(report)
        assert "/absolute" not in report_str


# ---------------------------------------------------------------------------
# Atomic write tests
# ---------------------------------------------------------------------------


class TestAtomicWriteJson:
    """Tests for :func:`atomic_write_json`."""

    def test_writes_json(self, tmp_path: Path) -> None:
        dest = tmp_path / "output.json"
        data = {"key": "value", "number": 42}
        atomic_write_json(data, dest)

        assert dest.exists()
        with open(dest) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        dest = tmp_path / "subdir" / "nested" / "output.json"
        data = {"key": "value"}
        atomic_write_json(data, dest)

        assert dest.exists()

    def test_no_nan_values(self, tmp_path: Path) -> None:
        dest = tmp_path / "output.json"
        data = {"key": float("nan")}
        with pytest.raises(ValueError):
            atomic_write_json(data, dest)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for :func:`parse_args`."""

    def test_required_model(self) -> None:
        args = parse_args(["--model", "test.gguf"])
        assert args.model == Path("test.gguf")
        assert args.output is None
        assert args.no_verify_hash is False

    def test_optional_output(self) -> None:
        args = parse_args(["--model", "test.gguf", "--output", "out.json"])
        assert args.output == Path("out.json")

    def test_no_verify_hash_flag(self) -> None:
        args = parse_args(["--model", "test.gguf", "--no-verify-hash"])
        assert args.no_verify_hash is True

    def test_missing_model_raises(self) -> None:
        with pytest.raises(SystemExit):
            parse_args([])


class TestMain:
    """Tests for :func:`main`."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nonexistent.gguf"
        exit_code = main(["--model", str(nonexistent)])
        assert exit_code == 1


# ---------------------------------------------------------------------------
# Real artifact tests (gated by SMOLLM2_GGUF env var)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SMOLLM2_GGUF,
    reason="Set SMOLLM2_GGUF to the GGUF file path to run real tests",
)
class TestRealArtifact:
    """Tests against the real SmolLM2-135M GGUF artifact."""

    @pytest.fixture()
    def weights(self) -> SmolLM2Weights:
        from cetagostini.utils.pytensor.gguf_weights import load_smollm2_weights

        return load_smollm2_weights(SMOLLM2_GGUF, verify_hash=True)

    def test_config(self, weights: SmolLM2Weights) -> None:
        cfg = weights.config
        assert cfg.vocab_size == 49152
        assert cfg.hidden_size == 576
        assert cfg.n_layers == 30
        assert cfg.n_heads == 9
        assert cfg.n_kv_heads == 3
        assert cfg.head_dim == 64

    def test_token_embedding_shape(self, weights: SmolLM2Weights) -> None:
        assert weights.token_embedding.shape == (49152, 576)
        assert weights.token_embedding.dtype == np.float32
        assert np.all(np.isfinite(weights.token_embedding))

    def test_final_norm_shape(self, weights: SmolLM2Weights) -> None:
        assert weights.final_norm.shape == (576,)
        assert weights.final_norm.dtype == np.float32
        assert np.all(np.isfinite(weights.final_norm))

    def test_layer_count(self, weights: SmolLM2Weights) -> None:
        assert len(weights.layers) == 30

    def test_layer_0_shapes(self, weights: SmolLM2Weights) -> None:
        layer = weights.layers[0]
        assert layer["attn_norm"].shape == (576,)
        assert layer["wq"].shape == (576, 576)
        assert layer["wk"].shape == (576, 192)
        assert layer["wv"].shape == (576, 192)
        assert layer["wo"].shape == (576, 576)
        assert layer["ffn_norm"].shape == (576,)
        assert layer["w_gate"].shape == (576, 1536)
        assert layer["w_up"].shape == (576, 1536)
        assert layer["w_down"].shape == (1536, 576)

    def test_all_layers_finite(self, weights: SmolLM2Weights) -> None:
        for i, layer in enumerate(weights.layers):
            for key, arr in layer.items():
                assert np.all(np.isfinite(arr)), f"Layer {i} {key} has non-finite values"
                assert arr.dtype == np.float32
                assert arr.flags["C_CONTIGUOUS"]
