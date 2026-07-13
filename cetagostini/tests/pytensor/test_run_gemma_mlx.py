"""Tests for run_gemma_mlx (native MLX-LM reference oracle).

Focused unit tests covering:
- CLI parsing
- Result sanitization
- Integration with run_gemma3n_pytensor helpers

Integration test is gated by the ``GEMMA3N_SNAPSHOT`` environment variable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cetagostini.utils.pytensor.run_gemma_mlx import (
    DEFAULT_PROMPT,
    atomic_write_json,
    collect_versions,
    main,
    parse_args,
    sanitize_result,
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
    import hashlib

    patched: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_FILES:
        fpath = snapshot_dir / name
        if fpath.exists() and fpath.stat().st_size > 0:
            patched[name] = {
                "size": fpath.stat().st_size,
                "sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
            }
    monkeypatch.setattr(run_gemma3n_pytensor, "EXPECTED_MANIFEST", patched)


# ---------------------------------------------------------------------------
# Tests: CLI parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    """Tests for CLI argument parsing."""

    def test_minimal(self, tmp_path):
        snap = _make_snapshot(tmp_path)
        args = parse_args(["--snapshot", str(snap)])
        assert args.snapshot == snap
        assert args.prompt == DEFAULT_PROMPT
        assert args.output is None

    def test_all_options(self, tmp_path):
        snap = _make_snapshot(tmp_path)
        out = tmp_path / "result.json"
        args = parse_args([
            "--snapshot", str(snap),
            "--prompt", "Hello world",
            "--output", str(out),
        ])
        assert args.prompt == "Hello world"
        assert args.output == out

    def test_requires_snapshot(self):
        with pytest.raises(SystemExit):
            parse_args([])


# ---------------------------------------------------------------------------
# Tests: sanitize_result
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
# Tests: main entry point (mocked)
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    """Tests for the main() entry point with mocked dependencies."""

    def test_run_with_missing_snapshot_returns_one(self, tmp_path):
        snap = tmp_path / "nonexistent"
        rc = main(["--snapshot", str(snap)])
        assert rc == 1

    def test_run_with_invalid_snapshot_returns_one(self, tmp_path):
        snap = tmp_path / "bad_snapshot"
        snap.mkdir()
        rc = main(["--snapshot", str(snap)])
        assert rc == 1


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
        out = tmp_path / "result.json"
        rc = main(["--snapshot", str(snap), "--output", str(out)])
        assert rc == 0
        assert out.exists()
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["model"]["revision"] == EXPECTED_REVISION
        assert report["runtime"] == "mlx_lm_native"
        assert report["reference"]["vocab_size"] > 0
        assert report["reference"]["seq_len"] > 0
        assert len(report["reference"]["top10_final_position"]) == 10
