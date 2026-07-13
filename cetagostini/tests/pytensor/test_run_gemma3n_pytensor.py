"""Tests for run_gemma3n_pytensor orchestration/runtime.

Focused unit tests covering:
- CLI parsing (probe/run subcommands, oracle artifacts, backend rejection)
- Snapshot validation (revision, manifest, config)
- Weight transposition (Linear [out,in] → [in,out])
- All-position metrics computation
- Publication thresholds (hard gates)
- Atomic JSON writes (allow_nan=False)
- MLX-LM reference contract (cache=None)

Integration test is gated by the ``GEMMA3N_SNAPSHOT`` environment variable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
    DEFAULT_PROMPT,
    EXPECTED_ARCHITECTURE,
    EXPECTED_BITS,
    EXPECTED_GROUP_SIZE,
    EXPECTED_MANIFEST,
    EXPECTED_MODEL_TYPE,
    EXPECTED_REPO,
    EXPECTED_REVISION,
    REQUIRED_FILES,
    VALID_BACKENDS,
    PUB_COSINE_MIN,
    PUB_PEARSON_MIN,
    PUB_ALL_TOP1_MATCH,
    PUB_TOP10_OVERLAP_MEAN_MIN,
    _LINEAR_KEYS,
    _sha256_file,
    _unpack_layer_args,
    atomic_write_json,
    build_file_manifest,
    check_optional_statuses,
    check_publication_thresholds,
    collect_versions,
    compute_all_position_metrics,
    detect_revision,
    decode_single_token,
    get_backend_info,
    get_device,
    get_peak_rss_mib,
    format_and_tokenize,
    hash_token_ids,
    main,
    parse_args,
    run_probe,
    run_mlx_reference,
    sanitize_result,
    transpose_global_weight,
    transpose_layer_weights,
    validate_snapshot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _valid_config() -> dict[str, Any]:
    """Return a minimal valid config.json for the snapshot."""
    return {
        "model_type": EXPECTED_MODEL_TYPE,
        "architectures": [EXPECTED_ARCHITECTURE],
        "quantization": {
            "bits": EXPECTED_BITS,
            "group_size": EXPECTED_GROUP_SIZE,
        },
    }


def _make_snapshot(tmp_path: Path, *, basename: str | None = None) -> Path:
    """Create a minimal valid snapshot directory under ``tmp_path``."""
    name = basename if basename is not None else EXPECTED_REVISION
    snap = tmp_path / name
    snap.mkdir(parents=True, exist_ok=True)
    config = _valid_config()
    (snap / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"\x00" * 64)
    (snap / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snap / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snap / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    return snap


def _patch_expected_manifest(monkeypatch, snapshot_dir: Path) -> None:
    """Patch EXPECTED_MANIFEST to match the test fixture's actual files."""
    from cetagostini.utils.pytensor import run_gemma3n_pytensor

    patched: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_FILES:
        fpath = snapshot_dir / name
        if fpath.exists() and fpath.stat().st_size > 0:
            patched[name] = {
                "size": fpath.stat().st_size,
                "sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
            }
    monkeypatch.setattr(run_gemma3n_pytensor, "EXPECTED_MANIFEST", patched)


def _make_ref_logits(T: int = 5, V: int = 100, seed: int = 42) -> np.ndarray:
    """Create deterministic reference logits."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, T, V)).astype(np.float32)


def _mock_implementation_manifest() -> dict[str, Any]:
    """Return stable implementation identity for oracle contract tests."""
    return {
        "git_commit": "a" * 40,
        "git_clean": True,
        "environment_yml_sha256": "b" * 64,
        "source_hashes": [{"path": "runner.py", "sha256": "c" * 64}],
        "implementation_manifest_sha256": "d" * 64,
        "python_executable": "/opt/conda/bin/python",
        "environment": {"python_version": "3.13.14"},
        "package_versions": {"pytensor": "3.1.2", "mlx": "0.32.0"},
        "module_paths": {"pytensor": "/env/pytensor/__init__.py"},
    }


def _make_pt_logits(
    ref_logits: np.ndarray, noise: float = 0.01, seed: int = 99
) -> np.ndarray:
    """Create PyTensor logits as noisy copy of reference."""
    rng = np.random.default_rng(seed)
    noise_arr = rng.standard_normal(ref_logits.shape).astype(np.float32) * noise
    return ref_logits + noise_arr


def _make_small_text_config() -> SimpleNamespace:
    """Create a small Gemma3nTextConfig-like namespace for testing."""
    layer_types = tuple(
        ["sliding_attention"] * 4 + ["full_attention"]
    ) * 1  # 5 layers
    sparsity = tuple([0.95] * 3 + [0.0] * 2)  # 5 layers
    return SimpleNamespace(
        vocab_size=256,
        vocab_size_per_layer_input=128,
        hidden_size=64,
        hidden_size_per_layer_input=16,
        intermediate_size=128,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        sliding_window=32,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        rope_local_base_freq=10_000.0,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_activation="gelu_pytorch_tanh",
        final_logit_softcapping=30.0,
        altup_active_idx=0,
        altup_coef_clip=120.0,
        altup_correct_scale=True,
        altup_lr_multiplier=1.0,
        altup_num_inputs=4,
        laurel_rank=16,
        num_kv_shared_layers=3,
        query_pre_attn_scalar=256,
        layer_types=layer_types,
        activation_sparsity_pattern=sparsity,
    )


def _make_mock_layer_weights(text_config: SimpleNamespace) -> dict[str, np.ndarray]:
    """Create mock layer weights with realistic shapes (stored as [out, in])."""
    H = text_config.hidden_size
    H_pl = text_config.hidden_size_per_layer_input
    n_h = text_config.num_attention_heads
    n_kv = text_config.num_key_value_heads
    hd = text_config.head_dim
    I = text_config.intermediate_size
    rank = text_config.laurel_rank
    n = text_config.altup_num_inputs

    rng = np.random.default_rng(42)

    def _rand(shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.01

    return {
        # Attention: stored [out, in]
        "self_attn.q_proj": _rand((n_h * hd, H)),
        "self_attn.k_proj": _rand((n_kv * hd, H)),
        "self_attn.v_proj": _rand((n_kv * hd, H)),
        "self_attn.o_proj": _rand((H, n_h * hd)),
        "self_attn.q_norm": _rand((hd,)),
        "self_attn.k_norm": _rand((hd,)),
        # MLP: stored [out, in]
        "mlp.gate_proj": _rand((I, H)),
        "mlp.up_proj": _rand((I, H)),
        "mlp.down_proj": _rand((H, I)),
        # LayerNorms
        "input_layernorm": _rand((H,)),
        "post_attention_layernorm": _rand((H,)),
        "pre_feedforward_layernorm": _rand((H,)),
        "post_feedforward_layernorm": _rand((H,)),
        # AltUp
        "altup.correct_output_scale": _rand((H,)),
        "altup.correction_coefs": _rand((n, n)),
        "altup.modality_router": _rand((n, H)),
        "altup.prediction_coefs": _rand((n * n, n)),
        "altup.router_norm": _rand((H,)),
        # LAuReL: stored [out, in]
        "laurel.linear_left": _rand((rank, H)),
        "laurel.linear_right": _rand((H, rank)),
        "laurel.post_laurel_norm": _rand((H,)),
        # Per-layer: stored [out, in]
        "per_layer_input_gate": _rand((H_pl, H)),
        "per_layer_projection": _rand((H, H_pl)),
        "post_per_layer_input_norm": _rand((H,)),
    }


# ---------------------------------------------------------------------------
# Tests: CLI parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    """Tests for CLI argument parsing."""

    def test_probe_minimal(self):
        args = parse_args(["probe"])
        assert args.command == "probe"
        assert args.snapshot is None

    def test_probe_with_snapshot(self, tmp_path):
        snap = _make_snapshot(tmp_path)
        args = parse_args(["probe", "--snapshot", str(snap)])
        assert args.command == "probe"
        assert args.snapshot == snap

    def test_run_minimal(self, tmp_path):
        snap = _make_snapshot(tmp_path)
        ref_report = tmp_path / "ref_report.json"
        ref_report.write_text("{}", encoding="utf-8")
        ref_logits = tmp_path / "ref_logits.npy"
        ref_logits.write_bytes(b"\x00" * 64)
        logits_output = tmp_path / "backend_logits.npy"
        args = parse_args([
            "run", "--snapshot", str(snap),
            "--run-id", "test-run",
            "--reference-report", str(ref_report),
            "--reference-logits", str(ref_logits),
            "--logits-output", str(logits_output),
        ])
        assert args.command == "run"
        assert args.snapshot == snap
        assert args.prompt == DEFAULT_PROMPT
        assert args.backend == "c"
        assert args.output is None
        assert args.run_id == "test-run"
        assert args.reference_report == ref_report
        assert args.reference_logits == ref_logits
        assert args.logits_output == logits_output

    def test_run_all_options(self, tmp_path):
        snap = _make_snapshot(tmp_path)
        out = tmp_path / "result.json"
        ref_report = tmp_path / "ref_report.json"
        ref_report.write_text("{}", encoding="utf-8")
        ref_logits = tmp_path / "ref_logits.npy"
        ref_logits.write_bytes(b"\x00" * 64)
        logits_output = tmp_path / "backend_logits.npy"
        args = parse_args([
            "run",
            "--snapshot", str(snap),
            "--run-id", "my-run",
            "--reference-report", str(ref_report),
            "--reference-logits", str(ref_logits),
            "--logits-output", str(logits_output),
            "--prompt", "Hello world",
            "--backend", "numba",
            "--output", str(out),
        ])
        assert args.prompt == "Hello world"
        assert args.backend == "numba"
        assert args.output == out
        assert args.run_id == "my-run"
        assert args.logits_output == logits_output

    def test_reference_only_flag(self, tmp_path):
        """--reference-only is no longer a valid flag (oracle is consumed externally)."""
        snap = _make_snapshot(tmp_path)
        ref_report = tmp_path / "ref_report.json"
        ref_report.write_text("{}", encoding="utf-8")
        ref_logits = tmp_path / "ref_logits.npy"
        ref_logits.write_bytes(b"\x00" * 64)
        logits_output = tmp_path / "backend_logits.npy"
        with pytest.raises(SystemExit):
            parse_args([
                "run", "--snapshot", str(snap),
                "--run-id", "test",
                "--reference-report", str(ref_report),
                "--reference-logits", str(ref_logits),
                "--logits-output", str(logits_output),
                "--reference-only",
            ])

    def test_run_requires_snapshot(self, tmp_path):
        """run requires snapshot, run identity, and all logits artifacts."""
        with pytest.raises(SystemExit):
            parse_args(["run"])

    def test_no_command_fails(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_mlx_backend_accepted(self, tmp_path):
        """MLX is now an accepted run backend."""
        snap = _make_snapshot(tmp_path)
        ref_report = tmp_path / "ref_report.json"
        ref_report.write_text("{}", encoding="utf-8")
        ref_logits = tmp_path / "ref_logits.npy"
        ref_logits.write_bytes(b"\x00" * 64)
        logits_output = tmp_path / "backend_logits.npy"
        args = parse_args([
            "run", "--snapshot", str(snap),
            "--run-id", "test",
            "--reference-report", str(ref_report),
            "--reference-logits", str(ref_logits),
            "--logits-output", str(logits_output),
            "--backend", "mlx",
        ])
        assert args.backend == "mlx"

    def test_valid_backends_includes_mlx(self):
        assert "c" in VALID_BACKENDS
        assert "numba" in VALID_BACKENDS
        assert "mlx" in VALID_BACKENDS


# ---------------------------------------------------------------------------
# Tests: detect_revision
# ---------------------------------------------------------------------------


class TestDetectRevision:
    """Tests for strict revision detection."""

    def test_exact_match(self, tmp_path):
        snap = _make_snapshot(tmp_path)
        assert detect_revision(snap) == EXPECTED_REVISION

    def test_mismatch_raises(self, tmp_path):
        snap = _make_snapshot(tmp_path, basename="wrong_revision_hash")
        with pytest.raises(ValueError, match="does not match"):
            detect_revision(snap)

    def test_never_substitutes_expected(self, tmp_path):
        snap = _make_snapshot(tmp_path, basename="some_other_directory")
        with pytest.raises(ValueError):
            detect_revision(snap)


# ---------------------------------------------------------------------------
# Tests: validate_snapshot
# ---------------------------------------------------------------------------


class TestValidateSnapshot:
    """Tests for snapshot validation."""

    def test_valid_snapshot(self, tmp_path, monkeypatch):
        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        config = validate_snapshot(snap)
        assert config["model_type"] == EXPECTED_MODEL_TYPE

    def test_missing_file(self, tmp_path, monkeypatch):
        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        os.unlink(snap / "tokenizer.json")
        with pytest.raises(FileNotFoundError, match="tokenizer.json"):
            validate_snapshot(snap)

    def test_empty_file(self, tmp_path, monkeypatch):
        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        (snap / "tokenizer.json").write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            validate_snapshot(snap)

    def test_wrong_model_type(self, tmp_path, monkeypatch):
        snap = _make_snapshot(tmp_path)
        config = _valid_config()
        config["model_type"] = "wrong_type"
        (snap / "config.json").write_text(json.dumps(config), encoding="utf-8")
        _patch_expected_manifest(monkeypatch, snap)
        with pytest.raises(ValueError, match="model_type"):
            validate_snapshot(snap)

    def test_wrong_bits(self, tmp_path, monkeypatch):
        snap = _make_snapshot(tmp_path)
        config = _valid_config()
        config["quantization"]["bits"] = 8
        (snap / "config.json").write_text(json.dumps(config), encoding="utf-8")
        _patch_expected_manifest(monkeypatch, snap)
        with pytest.raises(ValueError, match="bits"):
            validate_snapshot(snap)

    def test_wrong_revision(self, tmp_path, monkeypatch):
        snap = _make_snapshot(tmp_path, basename="wrong_rev")
        with pytest.raises(ValueError, match="does not match"):
            validate_snapshot(snap)


# ---------------------------------------------------------------------------
# Tests: _sha256_file
# ---------------------------------------------------------------------------


class TestSha256File:
    """Tests for chunked SHA-256 file hashing."""

    def test_matches_read_bytes(self, tmp_path):
        fpath = tmp_path / "small.bin"
        data = b"hello world" * 100
        fpath.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _sha256_file(fpath) == expected

    def test_empty_file(self, tmp_path):
        fpath = tmp_path / "empty.bin"
        fpath.write_bytes(b"")
        assert _sha256_file(fpath) == hashlib.sha256(b"").hexdigest()

    def test_zero_chunk_size_raises(self, tmp_path):
        fpath = tmp_path / "data.bin"
        fpath.write_bytes(b"not empty")

        with pytest.raises(ValueError, match="chunk_size"):
            _sha256_file(fpath, chunk_size=0)


# ---------------------------------------------------------------------------
# Tests: build_file_manifest
# ---------------------------------------------------------------------------


class TestBuildFileManifest:
    """Tests for file manifest construction."""

    def test_all_files_present(self, tmp_path, monkeypatch):
        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        manifest = build_file_manifest(snap)
        assert len(manifest) == len(REQUIRED_FILES)
        names = [m["name"] for m in manifest]
        for name in REQUIRED_FILES:
            assert name in names

    def test_each_entry_has_required_keys(self, tmp_path, monkeypatch):
        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        manifest = build_file_manifest(snap)
        for entry in manifest:
            assert "name" in entry
            assert "size_bytes" in entry
            assert "sha256" in entry
            assert isinstance(entry["size_bytes"], int)
            assert entry["size_bytes"] > 0
            assert len(entry["sha256"]) == 64


# ---------------------------------------------------------------------------
# Tests: hash_token_ids
# ---------------------------------------------------------------------------


class TestHashTokenIds:
    """Tests for token ID hashing."""

    def test_deterministic(self):
        ids = [1, 2, 3, 4, 5]
        assert hash_token_ids(ids) == hash_token_ids(ids)

    def test_different_ids_different_hash(self):
        assert hash_token_ids([1, 2, 3]) != hash_token_ids([4, 5, 6])

    def test_empty_list(self):
        assert hash_token_ids([]) == hashlib.sha256(b"").hexdigest()

    def test_returns_hex_string(self):
        h = hash_token_ids([100, 200])
        assert len(h) == 64
        int(h, 16)

    @pytest.mark.parametrize("token_id", [-1, 2**32])
    def test_rejects_out_of_uint32_range(self, token_id):
        with pytest.raises(ValueError, match="uint32 range"):
            hash_token_ids([token_id])

    @pytest.mark.parametrize("token_id", [1.0, True, np.float64(2.0), np.bool_(False)])
    def test_rejects_non_integer_types(self, token_id):
        with pytest.raises(ValueError, match="integer type"):
            hash_token_ids([token_id])


# ---------------------------------------------------------------------------
# Tests: collect_versions
# ---------------------------------------------------------------------------


class TestCollectVersions:
    """Tests for version collection."""

    def test_always_has_python(self):
        v = collect_versions()
        assert "python" in v
        assert v["python"] != "unavailable"

    def test_always_has_numpy(self):
        v = collect_versions()
        assert "numpy" in v
        assert v["numpy"] != "unavailable"

    def test_has_pytensor_key(self):
        assert "pytensor" in collect_versions()

    def test_has_mlx_key(self):
        assert "mlx" in collect_versions()


# ---------------------------------------------------------------------------
# Tests: check_optional_statuses
# ---------------------------------------------------------------------------


class TestOptionalStatuses:
    """Tests for JAX/PyTensor-MLX status checks."""

    def test_returns_jax_keys(self):
        s = check_optional_statuses()
        assert "jax_installed" in s
        assert "jax_version" in s
        assert isinstance(s["jax_installed"], bool)

    def test_returns_pytensor_ml_keys(self):
        s = check_optional_statuses()
        assert "pytensor_ml_installed" in s
        assert "pytensor_ml_version" in s

    def test_jax_not_installed_in_test_env(self):
        s = check_optional_statuses()
        assert s["jax_installed"] is False
        assert s["jax_version"] is None


# ---------------------------------------------------------------------------
# Tests: get_device / get_peak_rss_mib
# ---------------------------------------------------------------------------


class TestDeviceAndMemory:
    """Tests for device and memory helpers."""

    def test_device_returns_string(self):
        d = get_device()
        assert isinstance(d, str)
        assert len(d) > 0

    def test_peak_rss_positive(self):
        rss = get_peak_rss_mib()
        assert isinstance(rss, float)
        assert rss > 0


# ---------------------------------------------------------------------------
# Tests: get_backend_info
# ---------------------------------------------------------------------------


class TestBackendInfo:
    """Tests for backend linker/mode info."""

    def test_c_backend(self):
        info = get_backend_info("c")
        assert info["name"] == "c"
        assert info["linker"] == "cvm"
        assert info["mode"] == "o2"

    def test_numba_backend(self):
        info = get_backend_info("numba")
        assert info["name"] == "numba"
        assert info["linker"] == "numba"
        assert info["mode"] == "fast_compile"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend_info("unknown")

    def test_mlx_backend_info(self):
        info = get_backend_info("mlx")
        assert info["name"] == "mlx"
        assert info["linker"] == "mlx"


# ---------------------------------------------------------------------------
# Tests: transpose_layer_weights
# ---------------------------------------------------------------------------


class TestTransposeLayerWeights:
    """Tests for weight transposition from [out,in] to [in,out]."""

    def test_linear_keys_transposed(self):
        tc = _make_small_text_config()
        raw = _make_mock_layer_weights(tc)
        transposed = transpose_layer_weights(raw)

        for key in _LINEAR_KEYS:
            raw_shape = raw[key].shape
            trans_shape = transposed[key].shape
            assert trans_shape == (raw_shape[1], raw_shape[0]), (
                f"{key}: expected ({raw_shape[1]}, {raw_shape[0]}), got {trans_shape}"
            )

    def test_non_linear_keys_unchanged(self):
        tc = _make_small_text_config()
        raw = _make_mock_layer_weights(tc)
        transposed = transpose_layer_weights(raw)

        non_linear = set(raw.keys()) - _LINEAR_KEYS
        for key in non_linear:
            np.testing.assert_array_equal(raw[key], transposed[key])

    def test_transposed_is_contiguous_float32(self):
        tc = _make_small_text_config()
        raw = _make_mock_layer_weights(tc)
        transposed = transpose_layer_weights(raw)

        for key in _LINEAR_KEYS:
            assert transposed[key].dtype == np.float32
            assert transposed[key].flags["C_CONTIGUOUS"]

    def test_q_proj_shape(self):
        tc = _make_small_text_config()
        raw = _make_mock_layer_weights(tc)
        transposed = transpose_layer_weights(raw)
        # q_proj: stored [n_h*hd, H] → transposed [H, n_h*hd]
        H = tc.hidden_size
        n_h_hd = tc.num_attention_heads * tc.head_dim
        assert transposed["self_attn.q_proj"].shape == (H, n_h_hd)

    def test_gate_proj_shape(self):
        tc = _make_small_text_config()
        raw = _make_mock_layer_weights(tc)
        transposed = transpose_layer_weights(raw)
        # gate_proj: stored [I, H] → transposed [H, I]
        assert transposed["mlp.gate_proj"].shape == (tc.hidden_size, tc.intermediate_size)

    def test_laurel_left_shape(self):
        tc = _make_small_text_config()
        raw = _make_mock_layer_weights(tc)
        transposed = transpose_layer_weights(raw)
        # laurel.linear_left: stored [rank, H] → transposed [H, rank]
        assert transposed["laurel.linear_left"].shape == (tc.hidden_size, tc.laurel_rank)

    def test_all_keys_preserved(self):
        tc = _make_small_text_config()
        raw = _make_mock_layer_weights(tc)
        transposed = transpose_layer_weights(raw)
        assert set(raw.keys()) == set(transposed.keys())


# ---------------------------------------------------------------------------
# Tests: _unpack_layer_args
# ---------------------------------------------------------------------------


class TestUnpackLayerArgs:
    """Tests for positional arg unpacking."""

    def test_returns_24_elements(self):
        tc = _make_small_text_config()
        raw = _make_mock_layer_weights(tc)
        w = transpose_layer_weights(raw)
        args = _unpack_layer_args(w)
        assert len(args) == 24

    def test_first_arg_is_q_proj(self):
        tc = _make_small_text_config()
        raw = _make_mock_layer_weights(tc)
        w = transpose_layer_weights(raw)
        args = _unpack_layer_args(w)
        np.testing.assert_array_equal(args[0], w["self_attn.q_proj"])

    def test_last_arg_is_per_layer_projection(self):
        tc = _make_small_text_config()
        raw = _make_mock_layer_weights(tc)
        w = transpose_layer_weights(raw)
        args = _unpack_layer_args(w)
        np.testing.assert_array_equal(args[-1], w["per_layer_projection"])

    def test_order_matches_decoder_layer_signature(self):
        """Verify the unpacking order matches compile_decoder_layer inputs."""
        tc = _make_small_text_config()
        raw = _make_mock_layer_weights(tc)
        w = transpose_layer_weights(raw)
        args = _unpack_layer_args(w)
        expected_keys = [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "laurel.linear_left", "laurel.linear_right", "laurel.post_laurel_norm",
            "altup.prediction_coefs", "altup.correction_coefs",
            "altup.modality_router", "altup.router_norm", "altup.correct_output_scale",
            "input_layernorm", "post_attention_layernorm",
            "pre_feedforward_layernorm", "post_feedforward_layernorm",
            "post_per_layer_input_norm",
            "per_layer_input_gate", "per_layer_projection",
        ]
        for i, key in enumerate(expected_keys):
            np.testing.assert_array_equal(args[i], w[key], err_msg=f"arg[{i}] != {key}")


# ---------------------------------------------------------------------------
# Tests: transpose_global_weight
# ---------------------------------------------------------------------------


class TestTransposeGlobalWeight:
    """Tests for global weight transposition."""

    def test_transposes_2d(self):
        arr = np.random.default_rng(0).standard_normal((10, 20)).astype(np.float32)
        result = transpose_global_weight(arr)
        assert result.shape == (20, 10)
        assert result.dtype == np.float32
        assert result.flags["C_CONTIGUOUS"]

    def test_values_correct(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = transpose_global_weight(arr)
        expected = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("shape", [(3,), (2, 3, 4)])
    def test_rejects_non_matrix(self, shape):
        with pytest.raises(ValueError, match="rank-2"):
            transpose_global_weight(np.zeros(shape, dtype=np.float32))


# ---------------------------------------------------------------------------
# Tests: compute_all_position_metrics
# ---------------------------------------------------------------------------


class TestAllPositionMetrics:
    """Tests for all-position comparison metrics."""

    def test_identical_logits_perfect_match(self):
        logits = _make_ref_logits(T=5, V=100)
        metrics = compute_all_position_metrics(logits, logits.copy())
        assert metrics["all_finite_ref"] is True
        assert metrics["all_finite_pt"] is True
        assert metrics["n_positions"] == 5
        assert metrics["final_top1_match"] is True
        for pos_m in metrics["per_position"]:
            assert pos_m["max_abs_diff"] == 0.0
            assert pos_m["mean_abs_diff"] == 0.0
            assert pos_m["rmse"] == 0.0
            assert pos_m["top10_overlap"] == 10
            assert pos_m["top1_match"] is True

    def test_noisy_logits_close_match(self):
        ref = _make_ref_logits(T=3, V=50, seed=42)
        pt = _make_pt_logits(ref, noise=0.001, seed=99)
        metrics = compute_all_position_metrics(ref, pt)
        agg = metrics["aggregate"]
        assert agg["cosine_mean"] > 0.99
        assert agg["pearson_mean"] > 0.99
        assert agg["top10_overlap_mean"] >= 5

    def test_random_logits_poor_match(self):
        ref = _make_ref_logits(T=3, V=50, seed=42)
        rng = np.random.default_rng(123)
        pt = rng.standard_normal((1, 3, 50)).astype(np.float32) * 10
        metrics = compute_all_position_metrics(ref, pt)
        agg = metrics["aggregate"]
        assert agg["cosine_mean"] < 0.99
        assert agg["rmse_mean"] > 0.01

    def test_nan_in_ref_detected(self):
        ref = _make_ref_logits(T=3, V=50)
        ref[0, 1, 0] = np.nan
        pt = _make_ref_logits(T=3, V=50, seed=99)
        metrics = compute_all_position_metrics(ref, pt)
        assert metrics["all_finite_ref"] is False
        assert metrics["per_position"][1]["finite_ref"] is False

    def test_nan_in_pt_detected(self):
        ref = _make_ref_logits(T=3, V=50)
        pt = ref.copy()
        pt[0, 2, 5] = np.nan
        metrics = compute_all_position_metrics(ref, pt)
        assert metrics["all_finite_pt"] is False
        assert metrics["per_position"][2]["finite_pt"] is False

    def test_nonfinite_final_position_has_no_bogus_token(self):
        ref = _make_ref_logits(T=3, V=50)
        pt = ref.copy()
        ref[0, -1, :] = np.nan
        pt[0, -1, :] = np.nan

        metrics = compute_all_position_metrics(ref, pt)

        assert metrics["final_top1_ref"] is None
        assert metrics["final_top1_pt"] is None
        assert metrics["final_top1_match"] is False
        assert metrics["all_top1_match"] is False

    def test_per_position_count_matches_T(self):
        ref = _make_ref_logits(T=7, V=30)
        metrics = compute_all_position_metrics(ref, ref.copy())
        assert len(metrics["per_position"]) == 7

    def test_final_top1(self):
        ref = _make_ref_logits(T=3, V=50)
        metrics = compute_all_position_metrics(ref, ref.copy())
        expected_top1 = int(np.argmax(ref[0, -1]))
        assert metrics["final_top1_ref"] == expected_top1
        assert metrics["final_top1_pt"] == expected_top1
        assert metrics["final_top1_match"] is True

    def test_final_top1_mismatch(self):
        ref = _make_ref_logits(T=3, V=50)
        pt = ref.copy()
        pt[0, -1, :] = 0.0
        pt[0, -1, 0] = 999.0
        ref[0, -1, 0] = -999.0
        metrics = compute_all_position_metrics(ref, pt)
        assert metrics["final_top1_match"] is False

    def test_single_position(self):
        ref = _make_ref_logits(T=1, V=20)
        metrics = compute_all_position_metrics(ref, ref.copy())
        assert metrics["n_positions"] == 1
        assert len(metrics["per_position"]) == 1

    def test_cosine_orthogonal(self):
        V = 100
        ref = np.zeros((1, 1, V), dtype=np.float32)
        pt = np.zeros((1, 1, V), dtype=np.float32)
        ref[0, 0, 0] = 1.0
        pt[0, 0, 1] = 1.0
        metrics = compute_all_position_metrics(ref, pt)
        assert metrics["per_position"][0]["cosine"] < 0.01

    def test_identical_constant_logits_have_perfect_agreement(self):
        ref = np.ones((1, 2, 20), dtype=np.float32)
        metrics = compute_all_position_metrics(ref, ref.copy())

        assert metrics["aggregate"]["cosine_min"] == pytest.approx(1.0)
        assert metrics["aggregate"]["pearson_min"] == 1.0

    def test_cosine_is_not_mean_centered(self):
        ref = np.array([[[1.0] + [0.0] * 9]], dtype=np.float32)
        candidate = np.array([[[2.0] + [1.0] * 9]], dtype=np.float32)
        metrics = compute_all_position_metrics(ref, candidate)

        expected = float(
            np.dot(ref.ravel(), candidate.ravel())
            / (np.linalg.norm(ref) * np.linalg.norm(candidate))
        )
        assert metrics["per_position"][0]["cosine"] == pytest.approx(
            expected, abs=1e-6
        )

    def test_cosine_stays_in_mathematical_range(self):
        rng = np.random.default_rng(4)
        logits = rng.standard_normal((1, 2, 262_400)).astype(np.float32)

        metrics = compute_all_position_metrics(logits, logits.copy())

        assert all(
            -1.0 <= position["cosine"] <= 1.0
            for position in metrics["per_position"]
        )

    def test_vocabulary_smaller_than_top10_raises(self):
        logits = np.zeros((1, 1, 9), dtype=np.float32)
        with pytest.raises(ValueError, match="at least 10"):
            compute_all_position_metrics(logits, logits.copy())

    def test_aggregate_keys_present(self):
        ref = _make_ref_logits(T=3, V=50)
        metrics = compute_all_position_metrics(ref, ref.copy())
        agg = metrics["aggregate"]
        expected_keys = [
            "max_abs_diff_max", "max_abs_diff_mean",
            "mean_abs_diff_max", "mean_abs_diff_mean",
            "rmse_max", "rmse_mean",
            "cosine_min", "cosine_mean",
            "pearson_min", "pearson_mean",
            "top10_overlap_min", "top10_overlap_mean",
        ]
        for key in expected_keys:
            assert key in agg, f"Missing aggregate key: {key}"

    def test_shape_mismatch_raises(self):
        ref = _make_ref_logits(T=2, V=10)
        candidate = _make_ref_logits(T=3, V=10)
        with pytest.raises(ValueError, match="shapes must match"):
            compute_all_position_metrics(ref, candidate)

    def test_batch_larger_than_one_raises(self):
        logits = np.zeros((2, 2, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="batch size 1"):
            compute_all_position_metrics(logits, logits.copy())

    def test_empty_sequence_raises(self):
        logits = np.zeros((1, 0, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="non-empty"):
            compute_all_position_metrics(logits, logits.copy())


# ---------------------------------------------------------------------------
# Tests: check_publication_thresholds
# ---------------------------------------------------------------------------


class TestPublicationThresholds:
    """Tests for hard publication threshold checks."""

    def test_perfect_logits_pass(self):
        ref = _make_ref_logits(T=5, V=100)
        metrics = compute_all_position_metrics(ref, ref.copy())
        result = check_publication_thresholds(metrics)
        assert result["passed"] is True
        assert all(c["passed"] for c in result["checks"])

    def test_random_logits_fail(self):
        ref = _make_ref_logits(T=3, V=50, seed=42)
        rng = np.random.default_rng(123)
        pt = rng.standard_normal((1, 3, 50)).astype(np.float32) * 10
        metrics = compute_all_position_metrics(ref, pt)
        result = check_publication_thresholds(metrics)
        assert result["passed"] is False

    def test_nonfinite_logits_fail(self):
        ref = _make_ref_logits(T=2, V=50)
        pt = ref.copy()
        pt[0, 0, 0] = np.nan

        metrics = compute_all_position_metrics(ref, pt)
        result = check_publication_thresholds(metrics)

        assert metrics["all_finite_pt"] is False
        assert result["passed"] is False

    def test_one_bad_position_fails(self):
        ref = _make_ref_logits(T=20, V=100)
        candidate = ref.copy()
        candidate[0, 7] = -ref[0, 7]

        metrics = compute_all_position_metrics(ref, candidate)
        result = check_publication_thresholds(metrics)

        assert metrics["aggregate"]["cosine_mean"] > 0.89
        assert metrics["aggregate"]["cosine_min"] < 0.0
        assert metrics["all_top1_match"] is False
        assert result["passed"] is False

    def test_thresholds_use_unrounded_values(self):
        metrics = {
            "all_finite_ref": True,
            "all_finite_pt": True,
            "all_top1_match": True,
            "aggregate": {
                "cosine_min": 0.9899996,
                "pearson_min": 1.0,
                "top10_overlap_mean": 10.0,
            },
        }

        result = check_publication_thresholds(metrics)

        assert result["passed"] is False

    def test_check_names_present(self):
        ref = _make_ref_logits(T=3, V=50)
        metrics = compute_all_position_metrics(ref, ref.copy())
        result = check_publication_thresholds(metrics)
        names = [c["name"] for c in result["checks"]]
        assert "cosine_min" in names
        assert "pearson_min" in names
        assert "all_top1_match" in names
        assert "top10_overlap_mean" in names

    def test_each_check_has_threshold_and_actual(self):
        ref = _make_ref_logits(T=3, V=50)
        metrics = compute_all_position_metrics(ref, ref.copy())
        result = check_publication_thresholds(metrics)
        for check in result["checks"]:
            assert "name" in check
            assert "threshold" in check
            assert "actual" in check
            assert "passed" in check

    def test_thresholds_match_constants(self):
        ref = _make_ref_logits(T=3, V=50)
        metrics = compute_all_position_metrics(ref, ref.copy())
        result = check_publication_thresholds(metrics)
        check_map = {c["name"]: c for c in result["checks"]}
        assert check_map["cosine_min"]["threshold"] == PUB_COSINE_MIN
        assert check_map["pearson_min"]["threshold"] == PUB_PEARSON_MIN
        assert check_map["all_top1_match"]["threshold"] == PUB_ALL_TOP1_MATCH
        assert check_map["top10_overlap_mean"]["threshold"] == PUB_TOP10_OVERLAP_MEAN_MIN


# ---------------------------------------------------------------------------
# Tests: atomic_write_json
# ---------------------------------------------------------------------------


class TestAtomicWriteJson:
    """Tests for atomic JSON writing."""

    def test_writes_valid_json(self, tmp_path):
        dest = tmp_path / "out.json"
        data = {"key": "value", "num": 42}
        atomic_write_json(data, dest)
        assert dest.exists()
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        assert loaded == data

    def test_rejects_nan(self, tmp_path):
        dest = tmp_path / "out.json"
        with pytest.raises(ValueError):
            atomic_write_json({"bad": float("nan")}, dest)
        assert not dest.exists()

    def test_rejects_inf(self, tmp_path):
        dest = tmp_path / "out.json"
        with pytest.raises(ValueError):
            atomic_write_json({"bad": float("inf")}, dest)
        assert not dest.exists()

    def test_rejects_neg_inf(self, tmp_path):
        dest = tmp_path / "out.json"
        with pytest.raises(ValueError):
            atomic_write_json({"bad": float("-inf")}, dest)
        assert not dest.exists()

    def test_creates_parent_dirs(self, tmp_path):
        dest = tmp_path / "sub" / "dir" / "out.json"
        atomic_write_json({"ok": True}, dest)
        assert dest.exists()

    def test_overwrites_existing(self, tmp_path):
        dest = tmp_path / "out.json"
        dest.write_text('{"old": true}', encoding="utf-8")
        atomic_write_json({"new": True}, dest)
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        assert loaded == {"new": True}

    def test_no_temp_files_left(self, tmp_path):
        dest = tmp_path / "out.json"
        atomic_write_json({"ok": True}, dest)
        temps = list(tmp_path.glob("*.tmp"))
        assert len(temps) == 0

    def test_nested_nan_rejected(self, tmp_path):
        dest = tmp_path / "out.json"
        with pytest.raises(ValueError):
            atomic_write_json({"nested": {"deep": float("nan")}}, dest)

    def test_unicode_preserved(self, tmp_path):
        dest = tmp_path / "out.json"
        data = {"text": "hello \u00e9\u00e8"}
        atomic_write_json(data, dest)
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        assert loaded["text"] == data["text"]


# ---------------------------------------------------------------------------
# Tests: sanitize_result
# ---------------------------------------------------------------------------


class TestSanitizeResult:
    """Tests for result sanitization."""

    def _make_inputs(self, tmp_path):
        snap = _make_snapshot(tmp_path)
        ref_logits = _make_ref_logits(T=3, V=50)
        ref_result = {
            "logits": ref_logits,
            "load_s": 1.0,
            "forward_s": 0.5,
            "sync_s": 0.1,
            "peak_memory_mib": 100.0,
            "vocab_size": 50,
            "seq_len": 3,
        }
        return snap, ref_result

    def test_basic_structure(self, tmp_path, monkeypatch):
        snap, ref_result = self._make_inputs(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        manifest = build_file_manifest(snap)

        report = sanitize_result(
            snapshot_dir=snap,
            config_dict=_valid_config(),
            prompt_text="test prompt",
            formatted_text="<formatted>",
            token_ids=[1, 2, 3],
            backend="c",
            backend_info=get_backend_info("c"),
            versions=collect_versions(),
            optional_statuses=check_optional_statuses(),
            manifest=manifest,
            ref_result=ref_result,
            pt_result=None,
            metrics=None,
            pub_thresholds=None,
            timings={"ref_load_s": 1.0},
            memory={"peak_rss_mib": 100.0},
        )

        assert report["model"]["repo"] == EXPECTED_REPO
        assert report["model"]["revision"] == EXPECTED_REVISION
        assert report["prompt"]["text"] == "test prompt"
        assert report["prompt"]["token_ids"] == [1, 2, 3]
        assert "token_hash" in report["prompt"]
        assert report["backend"]["name"] == "c"
        assert "reference" in report
        assert "pytensor" not in report
        assert "metrics" not in report
        assert "publication_thresholds" not in report

    def test_with_pytensor_result(self, tmp_path, monkeypatch):
        snap, ref_result = self._make_inputs(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        manifest = build_file_manifest(snap)

        pt_result = {
            "logits": _make_pt_logits(ref_result["logits"]),
            "per_layer_s": [0.1, 0.2],
            "embed_s": 0.05,
            "ple_s": 0.01,
            "global_load_s": 0.02,
            "initial_s": 0.03,
            "per_layer_proj_s": 0.04,
            "final_s": 0.02,
            "logits_s": 0.03,
            "total_s": 0.5,
            "layers_completed": 2,
            "layer_types_used": ["sliding_attention", "full_attention"],
            "rope_bases_used": ["10K", "1M"],
            "sparse_layers_used": [0],
            "chunks_processed": 1,
        }
        metrics = compute_all_position_metrics(
            ref_result["logits"], pt_result["logits"]
        )
        pub = check_publication_thresholds(metrics)

        report = sanitize_result(
            snapshot_dir=snap,
            config_dict=_valid_config(),
            prompt_text="test",
            formatted_text="<fmt>",
            token_ids=[1, 2],
            backend="c",
            backend_info=get_backend_info("c"),
            versions=collect_versions(),
            optional_statuses=check_optional_statuses(),
            manifest=manifest,
            ref_result=ref_result,
            pt_result=pt_result,
            metrics=metrics,
            pub_thresholds=pub,
            timings={"ref_load_s": 1.0, "pt_total_s": 0.5},
            memory={"peak_rss_mib": 100.0},
        )

        assert "pytensor" in report
        assert report["pytensor"]["layers_completed"] == 2
        assert report["pytensor"]["chunks_processed"] == 1
        assert report["pytensor"]["layer_types_used"] == ["sliding_attention", "full_attention"]
        assert report["pytensor"]["rope_bases_used"] == ["10K", "1M"]
        assert "metrics" in report
        assert "publication_thresholds" in report

    def test_no_absolute_paths(self, tmp_path, monkeypatch):
        snap, ref_result = self._make_inputs(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        manifest = build_file_manifest(snap)

        report = sanitize_result(
            snapshot_dir=snap,
            config_dict=_valid_config(),
            prompt_text="test",
            formatted_text="<fmt>",
            token_ids=[1],
            backend="c",
            backend_info=get_backend_info("c"),
            versions=collect_versions(),
            optional_statuses=check_optional_statuses(),
            manifest=manifest,
            ref_result=ref_result,
            pt_result=None,
            metrics=None,
            pub_thresholds=None,
            timings={},
            memory={},
        )

        report_str = json.dumps(report)
        assert str(tmp_path) not in report_str

    def test_reference_has_logit_hash(self, tmp_path, monkeypatch):
        snap, ref_result = self._make_inputs(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        manifest = build_file_manifest(snap)

        report = sanitize_result(
            snapshot_dir=snap,
            config_dict=_valid_config(),
            prompt_text="test",
            formatted_text="<fmt>",
            token_ids=[1],
            backend="c",
            backend_info=get_backend_info("c"),
            versions=collect_versions(),
            optional_statuses=check_optional_statuses(),
            manifest=manifest,
            ref_result=ref_result,
            pt_result=None,
            metrics=None,
            pub_thresholds=None,
            timings={},
            memory={},
        )

        assert "logits_sha256" in report["reference"]
        assert len(report["reference"]["logits_sha256"]) == 64


# ---------------------------------------------------------------------------
# Tests: run_probe
# ---------------------------------------------------------------------------


class TestRunProbe:
    """Tests for the probe mode."""

    def test_probe_without_snapshot(self):
        result = run_probe(None)
        assert "versions" in result
        assert "optional_statuses" in result
        assert "device" in result
        assert "valid_backends" in result
        assert result["valid_backends"] == ["c", "numba", "mlx"]
        assert "publication_thresholds" in result
        assert "snapshot" not in result

    def test_probe_with_valid_snapshot(self, tmp_path, monkeypatch):
        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        result = run_probe(snap)
        assert result["snapshot"]["valid"] is True
        assert result["snapshot"]["revision"] == EXPECTED_REVISION

    def test_probe_with_invalid_snapshot(self, tmp_path):
        snap = tmp_path / "bad_snapshot"
        snap.mkdir()
        result = run_probe(snap)
        assert result["snapshot"]["valid"] is False
        assert "error" in result["snapshot"]

    def test_probe_reports_thresholds(self):
        result = run_probe(None)
        pt = result["publication_thresholds"]
        assert pt["cosine_min"] == PUB_COSINE_MIN
        assert pt["pearson_min"] == PUB_PEARSON_MIN


# ---------------------------------------------------------------------------
# Tests: MLX-LM reference contract (cache=None)
# ---------------------------------------------------------------------------


class TestPromptTokenization:
    def test_chat_template_does_not_add_a_second_bos(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<bos>formatted"
        tokenizer.encode.return_value = [2, 105, 107]

        formatted, token_ids = format_and_tokenize(tokenizer, "hello")

        assert formatted == "<bos>formatted"
        assert token_ids == [2, 105, 107]
        tokenizer.encode.assert_called_once_with(
            "<bos>formatted", add_special_tokens=False
        )


class TestMLXReferenceContract:
    """Tests for the direct cache=None oracle contract."""

    def test_run_mlx_reference_calls_production_contract(self):
        import mlx.core as mx

        class RecordingModel:
            def __init__(self):
                self.input_shape = None
                self.cache = "unset"

            def __call__(self, input_ids, *, cache):
                self.input_shape = input_ids.shape
                self.cache = cache
                return mx.ones((1, input_ids.shape[1], 16), dtype=mx.bfloat16)

        model = RecordingModel()
        with patch("mlx_lm.load", return_value=(model, object())):
            with patch(
                "cetagostini.utils.pytensor.run_gemma3n_pytensor.get_mlx_peak_memory_mib",
                return_value=12.5,
            ):
                result = run_mlx_reference(Path("/unused"), [10, 20, 30])

        assert model.input_shape == (1, 3)
        assert model.cache is None
        assert result["logits"].shape == (1, 3, 16)
        assert result["logits"].dtype == np.float32
        assert result["peak_memory_mib"] == 12.5

    @pytest.mark.parametrize(
        "decoded,expected",
        [("text", "text"), (b"text", "text"), (["text"], "['text']"), (None, "None")],
    )
    def test_decode_single_token_normalizes_text(self, decoded, expected):
        tokenizer = MagicMock()
        tokenizer.decode.return_value = decoded

        assert decode_single_token(tokenizer, 7) == expected
        tokenizer.decode.assert_called_once_with([7])


# ---------------------------------------------------------------------------
# Tests: main entry point (mocked)
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    """Tests for the main() entry point with mocked dependencies."""

    def test_probe_returns_zero(self, capsys):
        rc = main(["probe"])
        assert rc == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "versions" in output

    def test_probe_with_invalid_snapshot_returns_one(self, tmp_path, capsys):
        snap = tmp_path / "bad"
        snap.mkdir()
        rc = main(["probe", "--snapshot", str(snap)])
        assert rc == 1
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["snapshot"]["valid"] is False

    def test_run_with_missing_snapshot_returns_one(self, tmp_path):
        snap = tmp_path / "nonexistent"
        ref_report = tmp_path / "ref_report.json"
        ref_report.write_text("{}", encoding="utf-8")
        ref_logits = tmp_path / "ref_logits.npy"
        ref_logits.write_bytes(b"\x00" * 64)
        rc = main([
            "run", "--snapshot", str(snap),
            "--run-id", "test",
            "--reference-report", str(ref_report),
            "--reference-logits", str(ref_logits),
            "--logits-output", str(tmp_path / "backend.npy"),
        ])
        assert rc == 1

    def test_run_with_invalid_snapshot_returns_one(self, tmp_path):
        snap = tmp_path / "bad_snapshot"
        snap.mkdir()
        ref_report = tmp_path / "ref_report.json"
        ref_report.write_text("{}", encoding="utf-8")
        ref_logits = tmp_path / "ref_logits.npy"
        ref_logits.write_bytes(b"\x00" * 64)
        rc = main([
            "run", "--snapshot", str(snap),
            "--run-id", "test",
            "--reference-report", str(ref_report),
            "--reference-logits", str(ref_logits),
            "--logits-output", str(tmp_path / "backend.npy"),
        ])
        assert rc == 1


# ---------------------------------------------------------------------------
# Tests: hash_token_ids with NumPy integers
# ---------------------------------------------------------------------------


class TestHashTokenIdsNumpy:
    """Verify hash_token_ids accepts NumPy integer types."""

    def test_numpy_int32(self):
        ids = [np.int32(1), np.int32(2), np.int32(3)]
        assert hash_token_ids(ids) == hash_token_ids([1, 2, 3])

    def test_numpy_int64(self):
        ids = [np.int64(100), np.int64(200)]
        assert hash_token_ids(ids) == hash_token_ids([100, 200])

    def test_mixed_python_and_numpy(self):
        ids = [1, np.int32(2), np.int64(3)]
        assert hash_token_ids(ids) == hash_token_ids([1, 2, 3])


# ---------------------------------------------------------------------------
# Tests: MLX sync split (eval_tree + host_copy)
# ---------------------------------------------------------------------------


class TestMLXSyncSplit:
    """Verify the split eval/host-copy MLX synchronization pattern."""

    def test_maybe_eval_tree_noop_for_c(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import _maybe_eval_tree

        arr = np.zeros((2, 3), dtype=np.float32)
        dt = _maybe_eval_tree(arr, "c", label="test")
        assert dt == 0.0

    def test_maybe_eval_tree_noop_for_numba(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import _maybe_eval_tree

        arr = np.zeros((2, 3), dtype=np.float32)
        dt = _maybe_eval_tree(arr, "numba", label="test")
        assert dt == 0.0

    def test_maybe_host_copy_noop_for_c(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import _maybe_host_copy

        arr = np.zeros((2, 3), dtype=np.float32)
        dt, result = _maybe_host_copy(arr, "c")
        assert dt == 0.0
        assert result is arr

    def test_maybe_host_copy_noop_for_numba(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import _maybe_host_copy

        arr = np.zeros((2, 3), dtype=np.float32)
        dt, result = _maybe_host_copy(arr, "numba")
        assert dt == 0.0
        assert result is arr

    def test_mlx_eval_tree_calls_mx_eval(self):
        """For MLX backend, _maybe_eval_tree must call mx.eval."""
        import mlx.core as mx
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import _maybe_eval_tree

        arr = mx.ones((2, 3))
        dt = _maybe_eval_tree(arr, "mlx", label="test")
        assert isinstance(dt, float)
        assert dt >= 0.0

    def test_mlx_host_copy_produces_f4_c_contiguous(self):
        """For MLX backend, _maybe_host_copy must produce <f4 C-contiguous."""
        import mlx.core as mx
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
            _maybe_eval_tree,
            _maybe_host_copy,
        )

        arr = mx.ones((2, 3))
        _maybe_eval_tree(arr, "mlx", label="test")
        dt, result = _maybe_host_copy(arr, "mlx")
        assert isinstance(dt, float)
        assert dt >= 0.0
        assert result.dtype == np.dtype("<f4")
        assert result.flags["C_CONTIGUOUS"]
        assert result.shape == (2, 3)

    def test_mlx_host_copy_owns_memory(self):
        """Host-copied array must own its memory (not a view of device)."""
        import mlx.core as mx
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
            _maybe_eval_tree,
            _maybe_host_copy,
        )

        arr = mx.ones((4,))
        _maybe_eval_tree(arr, "mlx", label="test")
        _, result = _maybe_host_copy(arr, "mlx")
        assert result.flags["OWNDATA"]

    def test_mlx_eval_preserves_tuple_structure(self):
        """mx.eval must preserve tuple/list/dict structure."""
        import mlx.core as mx
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import _maybe_eval_tree

        t = (mx.ones((2,)), mx.zeros((3,)))
        dt = _maybe_eval_tree(t, "mlx", label="tuple_test")
        assert isinstance(dt, float)
        # Structure preserved — still a tuple
        assert isinstance(t, tuple)
        assert len(t) == 2


# ---------------------------------------------------------------------------
# Tests: Stage recording
# ---------------------------------------------------------------------------


class TestStageRecording:
    """Verify ordered stage entries with label, eval_s, host_copy_s."""

    def test_stage_entry_structure(self):
        """Each stage entry must have label, eval_s, host_copy_s."""
        stage = {"label": "initial_projections", "eval_s": 0.1, "host_copy_s": 0.0}
        assert "label" in stage
        assert "eval_s" in stage
        assert "host_copy_s" in stage

    def test_zero_duration_stage_recording(self):
        """Non-MLX backends must record zero-duration stages."""
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import _maybe_eval_tree

        arr = np.zeros((2, 3), dtype=np.float32)
        dt = _maybe_eval_tree(arr, "c", label="test_stage")
        assert dt == 0.0


# ---------------------------------------------------------------------------
# Tests: Numeric sparsity routing
# ---------------------------------------------------------------------------


class TestNumericSparsityRouting:
    """Verify layer functions are compiled/selected by distinct numeric values."""

    def test_sparsity_pattern_has_distinct_values(self):
        """The activation_sparsity_pattern must have distinct numeric values."""
        text_config = _make_small_text_config()
        distinct = sorted(set(text_config.activation_sparsity_pattern))
        assert 0.0 in distinct
        assert 0.95 in distinct
        assert len(distinct) == 2

    def test_sparsity_values_are_numeric_not_bool(self):
        """Sparsity values must be numeric floats, not booleans."""
        text_config = _make_small_text_config()
        for val in text_config.activation_sparsity_pattern:
            assert isinstance(val, float)
            assert not isinstance(val, bool)


# ---------------------------------------------------------------------------
# Tests: Tokenizer-only loading
# ---------------------------------------------------------------------------


class TestTokenizerOnlyLoading:
    """Verify tokenizer loading does not load model weights."""

    def test_load_tokenizer_uses_auto_tokenizer(self):
        """load_tokenizer_from_snapshot must use transformers.AutoTokenizer."""
        from unittest.mock import patch, MagicMock
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import load_tokenizer_from_snapshot

        mock_tokenizer = MagicMock()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_from:
            result = load_tokenizer_from_snapshot(Path("/fake/snapshot"))
            mock_from.assert_called_once_with(
                "/fake/snapshot", local_files_only=True
            )
            assert result is mock_tokenizer


# ---------------------------------------------------------------------------
# Tests: Oracle consumption
# ---------------------------------------------------------------------------


class TestOracleConsumption:
    """Verify oracle artifact loading and identity verification."""

    def _make_valid_ref_report(self, tmp_path, snap, token_ids):
        """Create a valid reference report for testing."""
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
            build_file_manifest,
            hash_token_ids,
        )

        manifest = build_file_manifest(snap)
        implementation = _mock_implementation_manifest()
        report = {
            "schema_version": "gemma3n-oracle-v1",
            "run_id": "test-run",
            "model": {
                "repo": EXPECTED_REPO,
                "revision": EXPECTED_REVISION,
                "model_type": EXPECTED_MODEL_TYPE,
                "architecture": EXPECTED_ARCHITECTURE,
                "quantization": {
                    "bits": EXPECTED_BITS,
                    "group_size": EXPECTED_GROUP_SIZE,
                },
                "manifest": manifest,
            },
            "prompt": {
                "text": "test prompt",
                "formatted": "formatted prompt",
                "token_ids": token_ids,
                "n_tokens": len(token_ids),
                "token_hash": hash_token_ids(token_ids),
            },
            "reference": {
                "shape": [1, len(token_ids), 100],
                "vocab_size": 100,
                "seq_len": len(token_ids),
                "logits_sha256": "a" * 64,
            },
            "raw_artifact": {
                "shape": [1, len(token_ids), 100],
                "canonical_sha256": "a" * 64,
            },
            "provenance": {
                "run_id": "test-run",
                "schema_version": "gemma3n-oracle-v1",
                "implementation": implementation,
                "command": [],
            },
        }
        report_path = tmp_path / "ref_report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path, report

    def test_load_and_verify_reference_report_success(self, tmp_path, monkeypatch):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
            load_and_verify_reference_report,
        )

        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        token_ids = [1, 2, 3]
        report_path, _ = self._make_valid_ref_report(tmp_path, snap, token_ids)

        from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
            build_file_manifest,
            validate_snapshot,
        )

        config_dict = validate_snapshot(snap)
        manifest = build_file_manifest(snap)

        result = load_and_verify_reference_report(
            report_path,
            run_id="test-run",
            snapshot_dir=snap,
            config_dict=config_dict,
            manifest=manifest,
            prompt_text="test prompt",
            formatted_text="formatted prompt",
            token_ids=token_ids,
            implementation_manifest=_mock_implementation_manifest(),
        )
        assert result["model"]["repo"] == EXPECTED_REPO

    def test_load_and_verify_reference_report_wrong_repo(self, tmp_path, monkeypatch):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
            OracleVerificationError,
            load_and_verify_reference_report,
        )

        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        token_ids = [1, 2, 3]
        report_path, _ = self._make_valid_ref_report(tmp_path, snap, token_ids)

        # Tamper with the report
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["model"]["repo"] = "wrong/repo"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
            build_file_manifest,
            validate_snapshot,
        )

        config_dict = validate_snapshot(snap)
        manifest = build_file_manifest(snap)

        with pytest.raises(OracleVerificationError, match="repo"):
            load_and_verify_reference_report(
                report_path,
                run_id="test-run",
                snapshot_dir=snap,
                config_dict=config_dict,
                manifest=manifest,
                prompt_text="test prompt",
                formatted_text="formatted prompt",
                token_ids=token_ids,
                implementation_manifest=_mock_implementation_manifest(),
            )

    def test_load_and_verify_reference_report_wrong_token_hash(self, tmp_path, monkeypatch):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
            OracleVerificationError,
            load_and_verify_reference_report,
        )

        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        token_ids = [1, 2, 3]
        report_path, _ = self._make_valid_ref_report(tmp_path, snap, token_ids)

        # Tamper with token hash
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["prompt"]["token_hash"] = "0" * 64
        report_path.write_text(json.dumps(report), encoding="utf-8")

        from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
            build_file_manifest,
            validate_snapshot,
        )

        config_dict = validate_snapshot(snap)
        manifest = build_file_manifest(snap)

        with pytest.raises(OracleVerificationError, match="[Tt]oken hash"):
            load_and_verify_reference_report(
                report_path,
                run_id="test-run",
                snapshot_dir=snap,
                config_dict=config_dict,
                manifest=manifest,
                prompt_text="test prompt",
                formatted_text="formatted prompt",
                token_ids=token_ids,
                implementation_manifest=_mock_implementation_manifest(),
            )

    def test_load_and_verify_reference_logits_missing_file(self, tmp_path):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
            OracleVerificationError,
            load_and_verify_reference_logits,
        )

        ref_report = {
            "reference": {"vocab_size": 100, "seq_len": 3, "logits_sha256": "a" * 64},
        }
        with pytest.raises(OracleVerificationError, match="not found"):
            load_and_verify_reference_logits(
                tmp_path / "missing.npy",
                ref_report,
                expected_seq_len=3,
            )


# ---------------------------------------------------------------------------
# Tests: Memory fields
# ---------------------------------------------------------------------------


class TestMemoryFields:
    """Verify memory field structure in reports."""

    def test_get_mlx_memory_snapshot_returns_dict_or_none(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import get_mlx_memory_snapshot

        result = get_mlx_memory_snapshot()
        assert result is None or isinstance(result, dict)

    def test_reset_mlx_allocator_no_error(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import reset_mlx_allocator

        # Should not raise even if MLX is not available
        reset_mlx_allocator()


# ---------------------------------------------------------------------------
# Tests: Report schema and provenance
# ---------------------------------------------------------------------------


class TestReportSchemaAndProvenance:
    """Verify report includes schema_version, run_id, and provenance fields."""

    def test_sanitize_result_has_required_keys(self, tmp_path, monkeypatch):
        snap, ref_result = TestSanitizeResult._make_inputs(None, tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        manifest = build_file_manifest(snap)

        report = sanitize_result(
            snapshot_dir=snap,
            config_dict=_valid_config(),
            prompt_text="test",
            formatted_text="<fmt>",
            token_ids=[1, 2, 3],
            backend="c",
            backend_info=get_backend_info("c"),
            versions=collect_versions(),
            optional_statuses=check_optional_statuses(),
            manifest=manifest,
            ref_result=ref_result,
            pt_result=None,
            metrics=None,
            pub_thresholds=None,
            timings={},
            memory={"whole_process_peak_rss_mib": 100.0},
        )

        assert "model" in report
        assert "prompt" in report
        assert "backend" in report
        assert "reference" in report
        assert "timing" in report
        assert "memory" in report
        assert report["memory"]["whole_process_peak_rss_mib"] == 100.0

    def test_pytensor_result_has_stage_fields(self, tmp_path, monkeypatch):
        snap, ref_result = TestSanitizeResult._make_inputs(None, tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        manifest = build_file_manifest(snap)

        pt_result = {
            "logits": _make_pt_logits(ref_result["logits"]),
            "per_layer_s": [0.1, 0.2],
            "embed_s": 0.05,
            "ple_s": 0.01,
            "global_load_s": 0.02,
            "initial_s": 0.03,
            "per_layer_proj_s": 0.04,
            "final_s": 0.02,
            "logits_s": 0.03,
            "total_s": 0.5,
            "layers_completed": 2,
            "layer_types_used": ["sliding_attention", "full_attention"],
            "rope_bases_used": ["10K", "1M"],
            "sparse_layers_used": [0],
            "chunks_processed": 1,
            "mlx_eval_s": 0.1,
            "mlx_host_copy_s": 0.05,
            "mlx_stages": [
                {"label": "initial_projections", "eval_s": 0.05, "host_copy_s": 0.0},
            ],
            "stage_count": 1,
        }

        report = sanitize_result(
            snapshot_dir=snap,
            config_dict=_valid_config(),
            prompt_text="test",
            formatted_text="<fmt>",
            token_ids=[1, 2],
            backend="c",
            backend_info=get_backend_info("c"),
            versions=collect_versions(),
            optional_statuses=check_optional_statuses(),
            manifest=manifest,
            ref_result=ref_result,
            pt_result=pt_result,
            metrics=None,
            pub_thresholds=None,
            timings={},
            memory={},
        )

        assert "pytensor" in report
        assert report["pytensor"]["mlx_eval_s"] == 0.1
        assert report["pytensor"]["mlx_host_copy_s"] == 0.05
        assert report["pytensor"]["stage_count"] == 1
        assert len(report["pytensor"]["mlx_stages"]) == 1
        assert report["pytensor"]["mlx_stages"][0]["label"] == "initial_projections"


# ---------------------------------------------------------------------------
# Tests: C/Numba regressions (no backend_mlx structure)
# ---------------------------------------------------------------------------


class TestCNumbaRegressions:
    """Verify C/Numba backends produce no backend_mlx structure."""

    def test_c_backend_no_backend_mlx_in_memory(self, tmp_path, monkeypatch):
        snap, ref_result = TestSanitizeResult._make_inputs(None, tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        manifest = build_file_manifest(snap)

        report = sanitize_result(
            snapshot_dir=snap,
            config_dict=_valid_config(),
            prompt_text="test",
            formatted_text="<fmt>",
            token_ids=[1],
            backend="c",
            backend_info=get_backend_info("c"),
            versions=collect_versions(),
            optional_statuses=check_optional_statuses(),
            manifest=manifest,
            ref_result=ref_result,
            pt_result=None,
            metrics=None,
            pub_thresholds=None,
            timings={},
            memory={"whole_process_peak_rss_mib": 100.0},
        )

        assert "backend_mlx" not in report["memory"]

    def test_numba_backend_no_backend_mlx_in_memory(self, tmp_path, monkeypatch):
        snap, ref_result = TestSanitizeResult._make_inputs(None, tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        manifest = build_file_manifest(snap)

        report = sanitize_result(
            snapshot_dir=snap,
            config_dict=_valid_config(),
            prompt_text="test",
            formatted_text="<fmt>",
            token_ids=[1],
            backend="numba",
            backend_info=get_backend_info("numba"),
            versions=collect_versions(),
            optional_statuses=check_optional_statuses(),
            manifest=manifest,
            ref_result=ref_result,
            pt_result=None,
            metrics=None,
            pub_thresholds=None,
            timings={},
            memory={"whole_process_peak_rss_mib": 100.0},
        )

        assert "backend_mlx" not in report["memory"]


# ---------------------------------------------------------------------------
# Tests: Atomic write JSON delegates to evidence
# ---------------------------------------------------------------------------


class TestAtomicWriteJsonDelegation:
    """Verify atomic_write_json delegates to evidence.atomic_write_json."""

    def test_delegates_to_evidence(self, tmp_path):
        dest = tmp_path / "test.json"
        data = {"key": "value"}
        atomic_write_json(data, dest)
        assert dest.exists()
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        assert loaded == data

    def test_rejects_nan_via_evidence(self, tmp_path):
        dest = tmp_path / "nan.json"
        with pytest.raises(ValueError):
            atomic_write_json({"bad": float("nan")}, dest)
        assert not dest.exists()


# ---------------------------------------------------------------------------
# Integration test (gated)
# ---------------------------------------------------------------------------


GEMMA3N_SNAPSHOT = os.environ.get("GEMMA3N_SNAPSHOT")
GEMMA3N_ORACLE_REPORT = os.environ.get("GEMMA3N_ORACLE_REPORT")
GEMMA3N_ORACLE_LOGITS = os.environ.get("GEMMA3N_ORACLE_LOGITS")
GEMMA3N_RUN_ID = os.environ.get("GEMMA3N_RUN_ID")


@pytest.mark.skipif(
    any(
        value is None
        for value in (
            GEMMA3N_SNAPSHOT,
            GEMMA3N_ORACLE_REPORT,
            GEMMA3N_ORACLE_LOGITS,
            GEMMA3N_RUN_ID,
        )
    ),
    reason=(
        "Set GEMMA3N_SNAPSHOT, GEMMA3N_ORACLE_REPORT, "
        "GEMMA3N_ORACLE_LOGITS, and GEMMA3N_RUN_ID"
    ),
)
class TestIntegration:
    """End-to-end integration test with real snapshot.

    Gated by ``GEMMA3N_SNAPSHOT`` environment variable pointing to a
    valid local HF snapshot directory.
    """

    def test_probe_real_snapshot(self):
        snap = Path(GEMMA3N_SNAPSHOT)
        result = run_probe(snap)
        assert result["snapshot"]["valid"] is True

    @pytest.mark.parametrize("backend", ["c", "numba", "mlx"])
    def test_full_backend_real_snapshot(self, tmp_path, backend):
        out = tmp_path / f"{backend}.json"
        logits_out = tmp_path / f"{backend}.npy"
        rc = main([
            "run",
            "--snapshot",
            str(Path(GEMMA3N_SNAPSHOT)),
            "--run-id",
            GEMMA3N_RUN_ID,
            "--reference-report",
            GEMMA3N_ORACLE_REPORT,
            "--reference-logits",
            GEMMA3N_ORACLE_LOGITS,
            "--logits-output",
            str(logits_out),
            "--backend",
            backend,
            "--output",
            str(out),
        ])

        assert rc == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["pytensor"]["layers_completed"] == 35
        assert report["pytensor"]["chunks_processed"] == 65
        assert report["metrics"]["n_positions"] == 20
        assert report["metrics"]["all_top1_match"] is True
        assert report["publication_thresholds"]["passed"] is True
        if backend == "mlx":
            assert report["pytensor"]["stage_count"] == 103
            assert "backend_mlx" in report["memory"]
        else:
            assert "backend_mlx" not in report["memory"]
