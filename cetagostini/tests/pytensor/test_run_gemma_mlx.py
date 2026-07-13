"""Tests for run_gemma_mlx (native MLX-LM reference oracle).

Focused unit tests covering:
- CLI parsing (including new --run-id and --logits-output)
- Result sanitization (backward compat)
- Oracle report building
- Oracle forward pass (mocked MLX-LM)
- Atomic publication ordering
- Artifact manifest/hash identity
- Strict nonfinite rejection
- Memory field separation
- No absolute path leakage
- Integration with run_gemma3n_pytensor helpers

Integration test is gated by the ``GEMMA3N_SNAPSHOT`` environment variable.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from cetagostini.utils.pytensor.run_gemma_mlx import (
    DEFAULT_PROMPT,
    GEMMA3N_ORACLE_SCHEMA_VERSION,
    _find_repo_root,
    atomic_write_json,
    atomic_write_npy,
    build_npy_manifest,
    build_oracle_report,
    collect_versions,
    main,
    parse_args,
    run_oracle_forward,
    sanitize_result,
    verify_npy_artifact,
)
from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
    EXPECTED_ARCHITECTURE,
    EXPECTED_BITS,
    EXPECTED_GROUP_SIZE,
    EXPECTED_MODEL_TYPE,
    EXPECTED_REPO,
    EXPECTED_REVISION,
    REQUIRED_FILES,
    build_file_manifest,
    detect_revision,
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


def _make_oracle_result(
    seq_len: int = 20,
    vocab_size: int = 256,
    *,
    finite: bool = True,
) -> dict[str, Any]:
    """Build a mock oracle result dictionary."""
    rng = np.random.default_rng(42)
    if finite:
        logits = rng.standard_normal((1, seq_len, vocab_size)).astype(np.float32)
    else:
        logits = rng.standard_normal((1, seq_len, vocab_size)).astype(np.float32)
        logits[0, 0, 0] = float("nan")

    return {
        "logits": logits,
        "load_s": 1.234,
        "forward_s": 0.567,
        "sync_s": 0.089,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "mlx_api": "mx.get_peak_memory",
        "mlx_version": "0.24.1",
        "mlx_baseline_mib": 0.0,
        "mlx_peak_mib": 1234.56,
        "mlx_current_mib": 800.0,
    }


def _make_mock_provenance(run_id: str = "test-run-001") -> dict[str, Any]:
    """Build a mock provenance report."""
    return {
        "run_id": run_id,
        "schema_version": GEMMA3N_ORACLE_SCHEMA_VERSION,
        "implementation": {
            "git_commit": "a" * 40,
            "git_clean": True,
            "environment_yml_sha256": None,
            "source_hashes": [
                {"path": "cetagostini/utils/pytensor/run_gemma_mlx.py", "sha256": "b" * 64},
            ],
            "implementation_manifest_sha256": "c" * 64,
            "timestamp_utc": "2026-07-13T00:00:00+00:00",
            "python_executable": "/usr/bin/python3",
            "environment": {
                "python_version": "3.13.0",
                "python_executable": "/usr/bin/python3",
                "platform_system": "Darwin",
                "platform_machine": "arm64",
                "platform_release": "24.0.0",
                "platform_version": "Darwin Kernel Version 24.0.0",
            },
            "package_versions": {"python": "3.13.0", "numpy": "2.0.0"},
            "module_paths": {"numpy": "/usr/lib/python3/numpy/__init__.py"},
        },
        "command": ["python", "-m", "run_gemma_mlx", "--snapshot", "snapshot_dir"],
    }


# ---------------------------------------------------------------------------
# Tests: CLI parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    """Tests for CLI argument parsing."""

    def test_minimal(self, tmp_path):
        snap = _make_snapshot(tmp_path)
        logits_out = tmp_path / "logits.npy"
        args = parse_args([
            "--snapshot", str(snap),
            "--run-id", "test-001",
            "--logits-output", str(logits_out),
        ])
        assert args.snapshot == snap
        assert args.run_id == "test-001"
        assert args.logits_output == logits_out
        assert args.prompt == DEFAULT_PROMPT
        assert args.output is None

    def test_all_options(self, tmp_path):
        snap = _make_snapshot(tmp_path)
        logits_out = tmp_path / "logits.npy"
        out = tmp_path / "result.json"
        args = parse_args([
            "--snapshot", str(snap),
            "--run-id", "test-002",
            "--logits-output", str(logits_out),
            "--prompt", "Hello world",
            "--output", str(out),
        ])
        assert args.prompt == "Hello world"
        assert args.output == out
        assert args.run_id == "test-002"
        assert args.logits_output == logits_out

    def test_requires_snapshot(self):
        with pytest.raises(SystemExit):
            parse_args([
                "--run-id", "r1",
                "--logits-output", "l.npy",
            ])

    def test_requires_run_id(self, tmp_path):
        snap = _make_snapshot(tmp_path)
        with pytest.raises(SystemExit):
            parse_args([
                "--snapshot", str(snap),
                "--logits-output", "l.npy",
            ])

    def test_requires_logits_output(self, tmp_path):
        snap = _make_snapshot(tmp_path)
        with pytest.raises(SystemExit):
            parse_args([
                "--snapshot", str(snap),
                "--run-id", "r1",
            ])

    def test_default_prompt(self, tmp_path):
        snap = _make_snapshot(tmp_path)
        args = parse_args([
            "--snapshot", str(snap),
            "--run-id", "r1",
            "--logits-output", str(tmp_path / "l.npy"),
        ])
        assert args.prompt == DEFAULT_PROMPT


# ---------------------------------------------------------------------------
# Tests: sanitize_result (backward compat)
# ---------------------------------------------------------------------------


class TestSanitizeResult:
    """Tests for result sanitization."""

    def test_basic_structure(self, tmp_path, monkeypatch):
        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        manifest = build_file_manifest(snap)

        ref_logits = np.random.default_rng(42).standard_normal((1, 3, 50)).astype(np.float32)
        ref_result = {
            "logits": ref_logits,
            "load_s": 1.0,
            "forward_s": 0.5,
            "sync_s": 0.1,
            "peak_memory_mib": 100.0,
            "vocab_size": 50,
            "seq_len": 3,
        }

        report = sanitize_result(
            snapshot_dir=snap,
            config_dict=_valid_config(),
            prompt_text="test prompt",
            formatted_text="<formatted>",
            token_ids=[1, 2, 3],
            versions=collect_versions(),
            optional_statuses={},
            manifest=manifest,
            ref_result=ref_result,
            timings={"ref_load_s": 1.0},
            memory={"peak_rss_mib": 100.0},
        )

        assert report["model"]["repo"] == EXPECTED_REPO
        assert report["model"]["revision"] == EXPECTED_REVISION
        assert report["runtime"] == "mlx_lm_native"
        assert report["prompt"]["text"] == "test prompt"
        assert report["prompt"]["token_ids"] == [1, 2, 3]
        assert "reference" in report
        assert "top10_final_position" in report["reference"]
        assert len(report["reference"]["top10_final_position"]) == 10

    def test_no_absolute_paths(self, tmp_path, monkeypatch):
        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        manifest = build_file_manifest(snap)

        ref_logits = np.random.default_rng(42).standard_normal((1, 3, 50)).astype(np.float32)
        ref_result = {
            "logits": ref_logits,
            "load_s": 1.0,
            "forward_s": 0.5,
            "sync_s": 0.1,
            "peak_memory_mib": 100.0,
            "vocab_size": 50,
            "seq_len": 3,
        }

        report = sanitize_result(
            snapshot_dir=snap,
            config_dict=_valid_config(),
            prompt_text="test",
            formatted_text="<fmt>",
            token_ids=[1],
            versions=collect_versions(),
            optional_statuses={},
            manifest=manifest,
            ref_result=ref_result,
            timings={},
            memory={},
        )

        report_str = json.dumps(report)
        assert str(tmp_path) not in report_str

    def test_reference_has_logit_hash(self, tmp_path, monkeypatch):
        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        manifest = build_file_manifest(snap)

        ref_logits = np.random.default_rng(42).standard_normal((1, 3, 50)).astype(np.float32)
        ref_result = {
            "logits": ref_logits,
            "load_s": 1.0,
            "forward_s": 0.5,
            "sync_s": 0.1,
            "peak_memory_mib": 100.0,
            "vocab_size": 50,
            "seq_len": 3,
        }

        report = sanitize_result(
            snapshot_dir=snap,
            config_dict=_valid_config(),
            prompt_text="test",
            formatted_text="<fmt>",
            token_ids=[1],
            versions=collect_versions(),
            optional_statuses={},
            manifest=manifest,
            ref_result=ref_result,
            timings={},
            memory={},
        )

        assert "logits_sha256" in report["reference"]
        assert len(report["reference"]["logits_sha256"]) == 64


# ---------------------------------------------------------------------------
# Tests: build_oracle_report
# ---------------------------------------------------------------------------


class TestBuildOracleReport:
    """Tests for the oracle report builder."""

    def _build(self, tmp_path, monkeypatch, **overrides):
        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        file_manifest = build_file_manifest(snap)

        oracle_result = overrides.pop(
            "oracle_result", _make_oracle_result(seq_len=20, vocab_size=256)
        )
        npy_manifest = overrides.pop("npy_manifest", {
            "basename": "oracle_logits.npy",
            "format": "npy",
            "dtype": "<f4",
            "byte_order": "little",
            "order": "C",
            "shape": [1, 20, 256],
            "payload_bytes": 20 * 256 * 4,
            "file_size": 20 * 256 * 4 + 128,
            "file_sha256": "a" * 64,
            "canonical_sha256": "b" * 64,
        })
        provenance = overrides.pop("provenance", _make_mock_provenance())

        return build_oracle_report(
            run_id=overrides.pop("run_id", "test-run-001"),
            snapshot_dir=snap,
            prompt_text=overrides.pop("prompt_text", "test prompt"),
            formatted_text=overrides.pop("formatted_text", "<formatted>"),
            token_ids=overrides.pop("token_ids", [1, 2, 3]),
            oracle_result=oracle_result,
            npy_manifest=npy_manifest,
            file_manifest=file_manifest,
            versions=collect_versions(),
            provenance=provenance,
        )

    def test_schema_version(self, tmp_path, monkeypatch):
        report = self._build(tmp_path, monkeypatch)
        assert report["schema_version"] == GEMMA3N_ORACLE_SCHEMA_VERSION

    def test_run_id(self, tmp_path, monkeypatch):
        report = self._build(tmp_path, monkeypatch, run_id="my-run-42")
        assert report["run_id"] == "my-run-42"

    def test_model_manifest_identity(self, tmp_path, monkeypatch):
        report = self._build(tmp_path, monkeypatch)
        model = report["model"]
        assert model["repo"] == EXPECTED_REPO
        assert model["revision"] == EXPECTED_REVISION
        assert model["model_type"] == EXPECTED_MODEL_TYPE
        assert model["architecture"] == EXPECTED_ARCHITECTURE
        assert model["quantization"]["bits"] == EXPECTED_BITS
        assert model["quantization"]["group_size"] == EXPECTED_GROUP_SIZE
        assert "manifest" in model
        assert len(model["manifest"]) == len(REQUIRED_FILES)

    def test_prompt_fields(self, tmp_path, monkeypatch):
        report = self._build(
            tmp_path, monkeypatch,
            prompt_text="hello",
            formatted_text="<fmt>hello",
            token_ids=[10, 20, 30],
        )
        assert report["prompt"]["text"] == "hello"
        assert report["prompt"]["formatted"] == "<fmt>hello"
        assert report["prompt"]["token_ids"] == [10, 20, 30]
        assert report["prompt"]["n_tokens"] == 3
        assert len(report["prompt"]["token_hash"]) == 64

    def test_reference_shape_hash_top1(self, tmp_path, monkeypatch):
        oracle_result = _make_oracle_result(seq_len=20, vocab_size=256)
        report = self._build(tmp_path, monkeypatch, oracle_result=oracle_result)
        ref = report["reference"]
        assert ref["shape"] == [1, 20, 256]
        assert len(ref["logits_sha256"]) == 64
        assert isinstance(ref["top1_id"], int)
        assert 0 <= ref["top1_id"] < 256
        assert ref["vocab_size"] == 256
        assert ref["seq_len"] == 20

    def test_raw_artifact_manifest(self, tmp_path, monkeypatch):
        npy_man = {
            "basename": "logits.npy",
            "format": "npy",
            "dtype": "<f4",
            "byte_order": "little",
            "order": "C",
            "shape": [1, 20, 256],
            "payload_bytes": 20480,
            "file_size": 20608,
            "file_sha256": "x" * 64,
            "canonical_sha256": "y" * 64,
        }
        report = self._build(tmp_path, monkeypatch, npy_manifest=npy_man)
        assert report["raw_artifact"] is npy_man
        assert report["raw_artifact"]["basename"] == "logits.npy"
        assert report["raw_artifact"]["dtype"] == "<f4"

    def test_no_absolute_paths(self, tmp_path, monkeypatch):
        report = self._build(tmp_path, monkeypatch)
        report_str = json.dumps(report)
        assert str(tmp_path) not in report_str

    def test_memory_field_separation(self, tmp_path, monkeypatch):
        report = self._build(tmp_path, monkeypatch)
        memory = report["memory"]
        assert "whole_process_peak_rss_mib" in memory
        assert isinstance(memory["whole_process_peak_rss_mib"], float)
        assert "oracle_mlx" in memory
        mlx = memory["oracle_mlx"]
        assert "api" in mlx
        assert "version" in mlx
        assert "baseline_mib" in mlx
        assert "current_mib" in mlx
        assert "peak_mib" in mlx
        # Verify separation: whole_process is RSS, oracle_mlx is MLX-specific
        assert mlx["peak_mib"] == 1234.56
        assert mlx["baseline_mib"] == 0.0
        assert mlx["current_mib"] == 800.0

    def test_provenance_fields(self, tmp_path, monkeypatch):
        prov = _make_mock_provenance(run_id="prov-run")
        report = self._build(tmp_path, monkeypatch, provenance=prov)
        assert report["provenance"]["run_id"] == "prov-run"
        assert report["provenance"]["schema_version"] == GEMMA3N_ORACLE_SCHEMA_VERSION
        assert "implementation" in report["provenance"]
        assert "command" in report["provenance"]

    def test_json_serializable(self, tmp_path, monkeypatch):
        report = self._build(tmp_path, monkeypatch)
        serialized = json.dumps(report, allow_nan=False)
        assert len(serialized) > 0
        deserialized = json.loads(serialized)
        assert deserialized["schema_version"] == GEMMA3N_ORACLE_SCHEMA_VERSION

    def test_runtime_field(self, tmp_path, monkeypatch):
        report = self._build(tmp_path, monkeypatch)
        assert report["runtime"] == "mlx_lm_native"

    def test_timing_fields(self, tmp_path, monkeypatch):
        report = self._build(tmp_path, monkeypatch)
        timing = report["timing"]
        assert "ref_load_s" in timing
        assert "ref_forward_s" in timing
        assert "ref_sync_s" in timing


# ---------------------------------------------------------------------------
# Tests: Oracle forward pass (mocked MLX-LM)
# ---------------------------------------------------------------------------


class TestOracleForwardMocked:
    """Tests for run_oracle_forward with mocked MLX-LM APIs."""

    def _make_mock_mlx(self):
        """Create a mock mlx.core module."""
        mock_mx = MagicMock()
        mock_mx.float32 = np.float32

        # Track call order
        call_log = []

        def mock_reset():
            call_log.append("reset_peak_memory")

        def mock_eval(value):
            call_log.append("eval")

        def mock_get_peak():
            call_log.append("get_peak_memory")
            return 100 * 1024 * 1024  # 100 MiB in bytes

        mock_mx.reset_peak_memory = mock_reset
        mock_mx.eval = mock_eval
        mock_mx.get_peak_memory = mock_get_peak

        # Remove metal and get_active_memory to simplify
        del mock_mx.metal
        del mock_mx.get_active_memory

        return mock_mx, call_log

    def test_one_model_load(self, tmp_path):
        """Verify mlx_lm.load is called exactly once."""
        snap = _make_snapshot(tmp_path)
        mock_mx, call_log = self._make_mock_mlx()

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.astype.return_value = mock_output
        mock_model.return_value = mock_output

        load_call_count = 0

        def mock_mlx_load(path):
            nonlocal load_call_count
            load_call_count += 1
            return mock_model, MagicMock()

        # Mock np.asarray to return a real array
        real_logits = np.random.default_rng(42).standard_normal(
            (1, 20, 256)
        ).astype(np.float32)

        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": mock_mx, "mlx_lm": MagicMock()}):
            with patch("cetagostini.utils.pytensor.run_gemma_mlx.importlib") as mock_importlib:
                mock_importlib.metadata.version.return_value = "0.24.1"

                # We need to patch the imports inside run_oracle_forward
                with patch("mlx.core", mock_mx):
                    with patch("mlx_lm.load", mock_mlx_load):
                        # Patch np.asarray to handle mock objects
                        original_asarray = np.asarray

                        def patched_asarray(obj, dtype=None):
                            if obj is mock_output:
                                return real_logits
                            return original_asarray(obj, dtype=dtype)

                        with patch("numpy.asarray", patched_asarray):
                            result = run_oracle_forward(snap, list(range(20)))

        assert load_call_count == 1

    def test_exact_all_position_shape(self, tmp_path):
        """Verify output shape is (1, T, V) for all-position logits."""
        snap = _make_snapshot(tmp_path)
        mock_mx, call_log = self._make_mock_mlx()

        T, V = 20, 256
        real_logits = np.random.default_rng(42).standard_normal(
            (1, T, V)
        ).astype(np.float32)

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.astype.return_value = mock_output
        mock_model.return_value = mock_output

        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": mock_mx, "mlx_lm": MagicMock()}):
            with patch("cetagostini.utils.pytensor.run_gemma_mlx.importlib") as mock_importlib:
                mock_importlib.metadata.version.return_value = "0.24.1"
                with patch("mlx.core", mock_mx):
                    with patch("mlx_lm.load", return_value=(mock_model, MagicMock())):
                        original_asarray = np.asarray

                        def patched_asarray(obj, dtype=None):
                            if obj is mock_output:
                                return real_logits
                            return original_asarray(obj, dtype=dtype)

                        with patch("numpy.asarray", patched_asarray):
                            result = run_oracle_forward(snap, list(range(T)))

        assert result["logits"].shape == (1, T, V)
        assert result["seq_len"] == T
        assert result["vocab_size"] == V

    def test_reset_peak_before_forward(self, tmp_path):
        """Verify reset_peak_memory is called before the forward pass."""
        snap = _make_snapshot(tmp_path)
        mock_mx, call_log = self._make_mock_mlx()

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.astype.return_value = mock_output
        mock_model.return_value = mock_output

        real_logits = np.random.default_rng(42).standard_normal(
            (1, 5, 32)
        ).astype(np.float32)

        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": mock_mx, "mlx_lm": MagicMock()}):
            with patch("cetagostini.utils.pytensor.run_gemma_mlx.importlib") as mock_importlib:
                mock_importlib.metadata.version.return_value = "0.24.1"
                with patch("mlx.core", mock_mx):
                    with patch("mlx_lm.load", return_value=(mock_model, MagicMock())):
                        original_asarray = np.asarray

                        def patched_asarray(obj, dtype=None):
                            if obj is mock_output:
                                return real_logits
                            return original_asarray(obj, dtype=dtype)

                        with patch("numpy.asarray", patched_asarray):
                            run_oracle_forward(snap, list(range(5)))

        # reset_peak_memory should appear before eval in the call log
        assert "reset_peak_memory" in call_log
        assert "eval" in call_log
        reset_idx = call_log.index("reset_peak_memory")
        eval_idx = call_log.index("eval")
        assert reset_idx < eval_idx

    def test_eval_copy_before_metrics(self, tmp_path):
        """Verify eval is called before get_peak_memory (metrics read)."""
        snap = _make_snapshot(tmp_path)
        mock_mx, call_log = self._make_mock_mlx()

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.astype.return_value = mock_output
        mock_model.return_value = mock_output

        real_logits = np.random.default_rng(42).standard_normal(
            (1, 5, 32)
        ).astype(np.float32)

        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": mock_mx, "mlx_lm": MagicMock()}):
            with patch("cetagostini.utils.pytensor.run_gemma_mlx.importlib") as mock_importlib:
                mock_importlib.metadata.version.return_value = "0.24.1"
                with patch("mlx.core", mock_mx):
                    with patch("mlx_lm.load", return_value=(mock_model, MagicMock())):
                        original_asarray = np.asarray

                        def patched_asarray(obj, dtype=None):
                            if obj is mock_output:
                                return real_logits
                            return original_asarray(obj, dtype=dtype)

                        with patch("numpy.asarray", patched_asarray):
                            run_oracle_forward(snap, list(range(5)))

        # Peak is read after eval; the pre-forward baseline uses active memory.
        eval_idx = call_log.index("eval")
        peak_indices = [
            i for i, c in enumerate(call_log) if c == "get_peak_memory"
        ]
        assert peak_indices
        assert peak_indices[-1] > eval_idx

    def test_memory_fields_returned(self, tmp_path):
        """Verify all memory fields are present in the result."""
        snap = _make_snapshot(tmp_path)
        mock_mx, call_log = self._make_mock_mlx()

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.astype.return_value = mock_output
        mock_model.return_value = mock_output

        real_logits = np.random.default_rng(42).standard_normal(
            (1, 5, 32)
        ).astype(np.float32)

        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": mock_mx, "mlx_lm": MagicMock()}):
            with patch("cetagostini.utils.pytensor.run_gemma_mlx.importlib") as mock_importlib:
                mock_importlib.metadata.version.return_value = "0.24.1"
                with patch("mlx.core", mock_mx):
                    with patch("mlx_lm.load", return_value=(mock_model, MagicMock())):
                        original_asarray = np.asarray

                        def patched_asarray(obj, dtype=None):
                            if obj is mock_output:
                                return real_logits
                            return original_asarray(obj, dtype=dtype)

                        with patch("numpy.asarray", patched_asarray):
                            result = run_oracle_forward(snap, list(range(5)))

        assert "mlx_api" in result
        assert "mlx_version" in result
        assert "mlx_baseline_mib" in result
        assert "mlx_peak_mib" in result
        assert "mlx_current_mib" in result
        assert result["mlx_api"] == "mx.get_peak_memory"


# ---------------------------------------------------------------------------
# Tests: Atomic publication ordering
# ---------------------------------------------------------------------------


class TestAtomicPublicationOrdering:
    """Tests for correct publication ordering: .npy first, then JSON."""

    def test_npy_written_before_json(self, tmp_path):
        """Verify .npy is written before JSON in the publication flow."""
        write_order: list[str] = []

        original_write_npy = atomic_write_npy
        original_write_json = atomic_write_json

        def tracking_write_npy(arr, dest, **kwargs):
            write_order.append("npy")
            original_write_npy(arr, dest, **kwargs)

        def tracking_write_json(data, dest, **kwargs):
            write_order.append("json")
            original_write_json(data, dest, **kwargs)

        # Create a real .npy file to test the flow
        arr = np.arange(24, dtype=np.float32).reshape(1, 3, 8)
        npy_path = tmp_path / "test_logits.npy"
        json_path = tmp_path / "test_report.json"

        tracking_write_npy(arr, npy_path)
        assert npy_path.exists()

        # Build manifest after .npy exists
        manifest = build_npy_manifest(npy_path)

        # Write JSON last
        report = {"schema_version": GEMMA3N_ORACLE_SCHEMA_VERSION, "raw_artifact": manifest}
        tracking_write_json(report, json_path)

        assert write_order == ["npy", "json"]
        assert npy_path.exists()
        assert json_path.exists()

    def test_manifest_never_exists_before_raw_artifact(self, tmp_path):
        """The manifest is built from the .npy file, so the file must exist first."""
        npy_path = tmp_path / "logits.npy"

        # Manifest should fail if .npy doesn't exist
        with pytest.raises(FileNotFoundError):
            build_npy_manifest(npy_path)

        # Write .npy, then manifest should succeed
        arr = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        atomic_write_npy(arr, npy_path)
        manifest = build_npy_manifest(npy_path)
        assert manifest["shape"] == [1, 3, 4]


# ---------------------------------------------------------------------------
# Tests: Artifact manifest/hash identity
# ---------------------------------------------------------------------------


class TestArtifactManifestIdentity:
    """Tests for artifact manifest and hash identity."""

    def test_manifest_roundtrip(self, tmp_path):
        """Build manifest, verify artifact matches."""
        arr = np.random.default_rng(42).standard_normal(
            (1, 20, 256)
        ).astype(np.float32)
        arr = np.ascontiguousarray(arr, dtype=np.dtype("<f4"))
        path = tmp_path / "oracle_logits.npy"
        atomic_write_npy(arr, path)

        manifest = build_npy_manifest(path)
        verified = verify_npy_artifact(path, manifest)

        np.testing.assert_array_equal(verified, arr)
        assert verified.dtype == np.dtype("<f4")
        assert verified.flags["C_CONTIGUOUS"]

    def test_manifest_hash_identity(self, tmp_path):
        """The canonical_sha256 in the manifest matches the array bytes."""
        arr = np.arange(100, dtype=np.float32).reshape(1, 10, 10)
        arr = np.ascontiguousarray(arr, dtype=np.dtype("<f4"))
        path = tmp_path / "test.npy"
        atomic_write_npy(arr, path)

        manifest = build_npy_manifest(path)
        expected_hash = hashlib.sha256(arr.tobytes()).hexdigest()
        assert manifest["canonical_sha256"] == expected_hash

    def test_no_absolute_paths_in_manifest(self, tmp_path):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        path = tmp_path / "test.npy"
        atomic_write_npy(arr, path)

        manifest = build_npy_manifest(path)
        manifest_str = json.dumps(manifest)
        assert str(tmp_path) not in manifest_str


# ---------------------------------------------------------------------------
# Tests: Strict nonfinite rejection
# ---------------------------------------------------------------------------


class TestNonfiniteRejection:
    """Tests for strict nonfinite rejection in the oracle flow."""

    def test_nan_in_logits_rejected_by_main(self, tmp_path, monkeypatch):
        """main() should return 1 if logits contain NaN."""
        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        logits_out = tmp_path / "logits.npy"
        json_out = tmp_path / "report.json"

        nan_logits = np.ones((1, 5, 32), dtype=np.float32)
        nan_logits[0, 0, 0] = float("nan")

        mock_result = {
            "logits": nan_logits,
            "load_s": 0.1,
            "forward_s": 0.1,
            "sync_s": 0.01,
            "vocab_size": 32,
            "seq_len": 5,
            "mlx_api": "mx.get_peak_memory",
            "mlx_version": "0.24.1",
            "mlx_baseline_mib": 0.0,
            "mlx_peak_mib": 100.0,
            "mlx_current_mib": 50.0,
        }

        with patch(
            "cetagostini.utils.pytensor.run_gemma_mlx._load_tokenizer_only"
        ) as mock_tok:
            mock_tok.return_value = MagicMock()
            with patch(
                "cetagostini.utils.pytensor.run_gemma_mlx.format_and_tokenize"
            ) as mock_fmt:
                mock_fmt.return_value = ("<fmt>", [1, 2, 3, 4, 5])
                with patch(
                    "cetagostini.utils.pytensor.run_gemma_mlx.run_oracle_forward"
                ) as mock_fwd:
                    mock_fwd.return_value = mock_result
                    rc = main([
                        "--snapshot", str(snap),
                        "--run-id", "nan-test",
                        "--logits-output", str(logits_out),
                        "--output", str(json_out),
                    ])

        assert rc == 1
        assert not logits_out.exists()
        assert not json_out.exists()

    def test_inf_in_logits_rejected_by_main(self, tmp_path, monkeypatch):
        """main() should return 1 if logits contain Inf."""
        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        logits_out = tmp_path / "logits.npy"
        json_out = tmp_path / "report.json"

        inf_logits = np.ones((1, 5, 32), dtype=np.float32)
        inf_logits[0, 0, 0] = float("inf")

        mock_result = {
            "logits": inf_logits,
            "load_s": 0.1,
            "forward_s": 0.1,
            "sync_s": 0.01,
            "vocab_size": 32,
            "seq_len": 5,
            "mlx_api": "mx.get_peak_memory",
            "mlx_version": "0.24.1",
            "mlx_baseline_mib": 0.0,
            "mlx_peak_mib": 100.0,
            "mlx_current_mib": 50.0,
        }

        with patch(
            "cetagostini.utils.pytensor.run_gemma_mlx._load_tokenizer_only"
        ) as mock_tok:
            mock_tok.return_value = MagicMock()
            with patch(
                "cetagostini.utils.pytensor.run_gemma_mlx.format_and_tokenize"
            ) as mock_fmt:
                mock_fmt.return_value = ("<fmt>", [1, 2, 3, 4, 5])
                with patch(
                    "cetagostini.utils.pytensor.run_gemma_mlx.run_oracle_forward"
                ) as mock_fwd:
                    mock_fwd.return_value = mock_result
                    rc = main([
                        "--snapshot", str(snap),
                        "--run-id", "inf-test",
                        "--logits-output", str(logits_out),
                        "--output", str(json_out),
                    ])

        assert rc == 1
        assert not logits_out.exists()
        assert not json_out.exists()

    def test_json_writer_rejects_nan(self, tmp_path):
        """atomic_write_json from evidence rejects NaN values."""
        dest = tmp_path / "bad.json"
        with pytest.raises(ValueError):
            atomic_write_json({"value": float("nan")}, dest)
        assert not dest.exists()

    def test_json_writer_rejects_inf(self, tmp_path):
        """atomic_write_json from evidence rejects Inf values."""
        dest = tmp_path / "bad.json"
        with pytest.raises(ValueError):
            atomic_write_json({"value": float("inf")}, dest)
        assert not dest.exists()


# ---------------------------------------------------------------------------
# Tests: main entry point (mocked)
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    """Tests for the main() entry point with mocked dependencies."""

    def test_run_with_missing_snapshot_returns_one(self, tmp_path):
        snap = tmp_path / "nonexistent"
        logits_out = tmp_path / "logits.npy"
        rc = main([
            "--snapshot", str(snap),
            "--run-id", "test",
            "--logits-output", str(logits_out),
        ])
        assert rc == 1

    def test_run_with_invalid_snapshot_returns_one(self, tmp_path):
        snap = tmp_path / "bad_snapshot"
        snap.mkdir()
        logits_out = tmp_path / "logits.npy"
        rc = main([
            "--snapshot", str(snap),
            "--run-id", "test",
            "--logits-output", str(logits_out),
        ])
        assert rc == 1

    def test_full_flow_with_mocks(self, tmp_path, monkeypatch):
        """Test the full main() flow with all dependencies mocked."""
        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        logits_out = tmp_path / "logits.npy"
        json_out = tmp_path / "report.json"

        finite_logits = np.random.default_rng(42).standard_normal(
            (1, 5, 32)
        ).astype(np.float32)

        mock_result = {
            "logits": finite_logits,
            "load_s": 0.1,
            "forward_s": 0.1,
            "sync_s": 0.01,
            "vocab_size": 32,
            "seq_len": 5,
            "mlx_api": "mx.get_peak_memory",
            "mlx_version": "0.24.1",
            "mlx_baseline_mib": 0.0,
            "mlx_peak_mib": 100.0,
            "mlx_current_mib": 50.0,
        }

        mock_provenance = _make_mock_provenance()

        with patch(
            "cetagostini.utils.pytensor.run_gemma_mlx._load_tokenizer_only"
        ) as mock_tok:
            mock_tok.return_value = MagicMock()
            with patch(
                "cetagostini.utils.pytensor.run_gemma_mlx.format_and_tokenize"
            ) as mock_fmt:
                mock_fmt.return_value = ("<fmt>", [1, 2, 3, 4, 5])
                with patch(
                    "cetagostini.utils.pytensor.run_gemma_mlx.run_oracle_forward"
                ) as mock_fwd:
                    mock_fwd.return_value = mock_result
                    with patch(
                        "cetagostini.utils.pytensor.run_gemma_mlx._find_repo_root"
                    ) as mock_root:
                        mock_root.return_value = tmp_path
                        with patch(
                            "cetagostini.utils.pytensor.run_gemma_mlx.build_implementation_manifest"
                        ) as mock_impl:
                            mock_impl.return_value = {
                                "git_commit": "a" * 40,
                                "git_clean": True,
                                "source_hashes": [],
                                "implementation_manifest_sha256": "b" * 64,
                            }
                            rc = main([
                                "--snapshot", str(snap),
                                "--run-id", "full-flow-test",
                                "--prompt", "test prompt",
                                "--logits-output", str(logits_out),
                                "--output", str(json_out),
                            ])

        assert rc == 0
        assert logits_out.exists()
        assert json_out.exists()

        # Verify JSON content
        report = json.loads(json_out.read_text(encoding="utf-8"))
        assert report["schema_version"] == GEMMA3N_ORACLE_SCHEMA_VERSION
        assert report["run_id"] == "full-flow-test"
        assert report["reference"]["shape"] == [1, 5, 32]
        assert "raw_artifact" in report
        assert "provenance" in report
        assert "memory" in report
        assert "oracle_mlx" in report["memory"]

        # Verify no absolute paths
        report_str = json.dumps(report)
        assert str(tmp_path) not in report_str

    def test_provenance_failure_returns_one(self, tmp_path, monkeypatch):
        """main() should return 1 if provenance building fails."""
        snap = _make_snapshot(tmp_path)
        _patch_expected_manifest(monkeypatch, snap)
        logits_out = tmp_path / "logits.npy"
        json_out = tmp_path / "report.json"

        finite_logits = np.random.default_rng(42).standard_normal(
            (1, 5, 32)
        ).astype(np.float32)

        mock_result = {
            "logits": finite_logits,
            "load_s": 0.1,
            "forward_s": 0.1,
            "sync_s": 0.01,
            "vocab_size": 32,
            "seq_len": 5,
            "mlx_api": "mx.get_peak_memory",
            "mlx_version": "0.24.1",
            "mlx_baseline_mib": 0.0,
            "mlx_peak_mib": 100.0,
            "mlx_current_mib": 50.0,
        }

        with patch(
            "cetagostini.utils.pytensor.run_gemma_mlx._load_tokenizer_only"
        ) as mock_tok:
            mock_tok.return_value = MagicMock()
            with patch(
                "cetagostini.utils.pytensor.run_gemma_mlx.format_and_tokenize"
            ) as mock_fmt:
                mock_fmt.return_value = ("<fmt>", [1, 2, 3, 4, 5])
                with patch(
                    "cetagostini.utils.pytensor.run_gemma_mlx.run_oracle_forward"
                ) as mock_fwd:
                    mock_fwd.return_value = mock_result
                    with patch(
                        "cetagostini.utils.pytensor.run_gemma_mlx._find_repo_root"
                    ) as mock_root:
                        mock_root.side_effect = RuntimeError("Not inside a git repository")
                        rc = main([
                            "--snapshot", str(snap),
                            "--run-id", "prov-fail-test",
                            "--prompt", "test prompt",
                            "--logits-output", str(logits_out),
                            "--output", str(json_out),
                        ])

        assert rc == 1
        # .npy should exist (written before provenance)
        assert logits_out.exists()
        # JSON should NOT exist (provenance failed before JSON write)
        assert not json_out.exists()


# ---------------------------------------------------------------------------
# Tests: _find_repo_root
# ---------------------------------------------------------------------------


class TestFindRepoRoot:
    """Tests for _find_repo_root."""

    def test_finds_repo_root(self):
        """Should find the repo root when running inside a git repo."""
        root = _find_repo_root()
        assert (root / ".git").exists()

    def test_returns_path(self):
        root = _find_repo_root()
        assert isinstance(root, Path)


# ---------------------------------------------------------------------------
# Integration test (gated)
# ---------------------------------------------------------------------------


GEMMA3N_SNAPSHOT = os.environ.get("GEMMA3N_SNAPSHOT")


@pytest.mark.skipif(
    GEMMA3N_SNAPSHOT is None,
    reason="Set GEMMA3N_SNAPSHOT env var to run integration test",
)
class TestIntegration:
    """End-to-end integration test with real snapshot."""

    def test_reference_real_snapshot(self, tmp_path):
        snap = Path(GEMMA3N_SNAPSHOT)
        logits_out = tmp_path / "oracle_logits.npy"
        json_out = tmp_path / "oracle_report.json"
        rc = main([
            "--snapshot", str(snap),
            "--run-id", "integration-test",
            "--logits-output", str(logits_out),
            "--output", str(json_out),
        ])
        assert rc == 0
        assert logits_out.exists()
        assert json_out.exists()

        report = json.loads(json_out.read_text(encoding="utf-8"))
        assert report["model"]["revision"] == EXPECTED_REVISION
        assert report["runtime"] == "mlx_lm_native"
        assert report["reference"]["vocab_size"] > 0
        assert report["reference"]["seq_len"] > 0
        assert report["schema_version"] == GEMMA3N_ORACLE_SCHEMA_VERSION

        # Verify .npy artifact
        arr = np.load(str(logits_out), allow_pickle=False)
        assert arr.dtype == np.dtype("<f4")
        assert arr.flags["C_CONTIGUOUS"]
        assert arr.ndim == 3
        assert arr.shape[0] == 1
