"""Tests for run_smollm2_mlx (native MLX-LM reference oracle for SmolLM2).

Focused unit tests covering:
- CLI parsing
- Result sanitization

Integration test is gated by the ``SMOLLM2_MLX_SNAPSHOT`` environment variable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cetagostini.utils.pytensor.run_smollm2_mlx import (
    DEFAULT_PROMPT,
    DEFAULT_MAX_TOKENS,
    atomic_write_json,
    collect_versions,
    get_device,
    get_peak_rss_mib,
    main,
    parse_args,
    run_mlx_generation,
    sanitize_result,
)


# ---------------------------------------------------------------------------
# Tests: CLI parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    """Tests for CLI argument parsing."""

    def test_minimal(self, tmp_path):
        model = tmp_path / "snapshot"
        model.mkdir()
        args = parse_args(["--model", str(model)])
        assert args.model == model
        assert args.prompt == DEFAULT_PROMPT
        assert args.max_tokens == DEFAULT_MAX_TOKENS
        assert args.output is None

    def test_all_options(self, tmp_path):
        model = tmp_path / "snapshot"
        model.mkdir()
        out = tmp_path / "result.json"
        args = parse_args([
            "--model", str(model),
            "--prompt", "Hello",
            "--max-tokens", "32",
            "--output", str(out),
        ])
        assert args.prompt == "Hello"
        assert args.max_tokens == 32
        assert args.output == out

    def test_requires_model(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_positive_int_rejects_zero(self):
        with pytest.raises(SystemExit):
            parse_args(["--model", "/tmp/m", "--max-tokens", "0"])


# ---------------------------------------------------------------------------
# Tests: sanitize_result
# ---------------------------------------------------------------------------


class TestSanitizeResult:
    """Tests for result sanitization."""

    def test_basic_structure(self, tmp_path):
        model = tmp_path / "snapshot"
        model.mkdir()

        gen_result = {
            "generated_ids": [10, 20, 30],
            "generated_text": "test output",
            "prompt_ids": [1, 2, 3],
            "prompt_text": "<formatted prompt>",
            "load_s": 1.5,
            "generation_s": 0.3,
            "peak_memory_mib": 100.0,
        }

        report = sanitize_result(
            model_path=model,
            prompt_text="test prompt",
            versions=collect_versions(),
            gen_result=gen_result,
            timings={"load_s": 1.5, "generation_s": 0.3},
            memory={"peak_rss_mib": 200.0, "mlx_peak_memory_mib": 100.0},
        )

        assert report["model"]["path"] == "snapshot"
        assert report["runtime"] == "mlx_lm_native"
        assert report["prompt"]["text"] == "test prompt"
        assert report["prompt"]["formatted"] == "<formatted prompt>"
        assert report["prompt"]["token_ids"] == [1, 2, 3]
        assert report["generation"]["generated_ids"] == [10, 20, 30]
        assert report["generation"]["text"] == "test output"
        assert report["generation"]["n_tokens"] == 3

    def test_no_absolute_paths(self, tmp_path):
        model = tmp_path / "snapshot"
        model.mkdir()

        gen_result = {
            "generated_ids": [10],
            "generated_text": "test",
            "prompt_ids": [1],
            "prompt_text": "<fmt>",
            "load_s": 1.0,
            "generation_s": 0.1,
            "peak_memory_mib": 50.0,
        }

        report = sanitize_result(
            model_path=model,
            prompt_text="test",
            versions=collect_versions(),
            gen_result=gen_result,
            timings={},
            memory={},
        )

        report_str = json.dumps(report)
        assert str(tmp_path) not in report_str


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for helper functions."""

    def test_collect_versions_has_python(self):
        v = collect_versions()
        assert "python" in v
        assert v["python"] != "unavailable"

    def test_collect_versions_has_numpy(self):
        v = collect_versions()
        assert "numpy" in v
        assert v["numpy"] != "unavailable"

    def test_collect_versions_has_mlx(self):
        v = collect_versions()
        assert "mlx" in v

    def test_get_device_returns_string(self):
        d = get_device()
        assert isinstance(d, str)
        assert len(d) > 0

    def test_get_peak_rss_positive(self):
        rss = get_peak_rss_mib()
        assert isinstance(rss, float)
        assert rss > 0


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


# ---------------------------------------------------------------------------
# Tests: main entry point (mocked)
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    """Tests for the main() entry point with mocked dependencies."""

    def test_run_with_missing_model_returns_one(self, tmp_path):
        model = tmp_path / "nonexistent"
        rc = main(["--model", str(model)])
        assert rc == 1


# ---------------------------------------------------------------------------
# Integration test (gated)
# ---------------------------------------------------------------------------


SMOLLM2_MLX_SNAPSHOT = os.environ.get("SMOLLM2_MLX_SNAPSHOT")


@pytest.mark.skipif(
    SMOLLM2_MLX_SNAPSHOT is None,
    reason="Set SMOLLM2_MLX_SNAPSHOT env var to run integration test",
)
class TestIntegration:
    """End-to-end integration test with real snapshot."""

    def test_generation_real_snapshot(self, tmp_path):
        snap = Path(SMOLLM2_MLX_SNAPSHOT)
        out = tmp_path / "result.json"
        rc = main(["--model", str(snap), "--max-tokens", "8", "--output", str(out)])
        assert rc == 0
        assert out.exists()
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["runtime"] == "mlx_lm_native"
        assert report["generation"]["n_tokens"] > 0
        assert len(report["generation"]["generated_ids"]) > 0
