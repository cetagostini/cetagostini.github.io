"""Cross-runner shared tests and contract verification.

Tests that verify shared contracts across all runner modules:
- Atomic JSON writing consistency
- Version collection patterns
- Report schema compatibility
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Tests: Atomic JSON writing consistency
# ---------------------------------------------------------------------------


class TestAtomicJsonConsistency:
    """Verify atomic_write_json behaves consistently across runners."""

    def test_gemma3n_pytensor_atomic_write(self, tmp_path):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import atomic_write_json

        dest = tmp_path / "gemma3n.json"
        data = {"model": "gemma3n", "value": 42}
        atomic_write_json(data, dest)
        assert dest.exists()
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        assert loaded == data

    def test_gemma_mlx_atomic_write(self, tmp_path):
        from cetagostini.utils.pytensor.run_gemma_mlx import atomic_write_json

        dest = tmp_path / "gemma_mlx.json"
        data = {"model": "gemma_mlx", "value": 42}
        atomic_write_json(data, dest)
        assert dest.exists()
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        assert loaded == data

    def test_smollm2_pytensor_atomic_write(self, tmp_path):
        from cetagostini.utils.pytensor.run_smollm2_pytensor import atomic_write_json

        dest = tmp_path / "smollm2.json"
        data = {"model": "smollm2", "value": 42}
        atomic_write_json(data, dest)
        assert dest.exists()
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        assert loaded == data

    def test_smollm2_mlx_atomic_write(self, tmp_path):
        from cetagostini.utils.pytensor.run_smollm2_mlx import atomic_write_json

        dest = tmp_path / "smollm2_mlx.json"
        data = {"model": "smollm2_mlx", "value": 42}
        atomic_write_json(data, dest)
        assert dest.exists()
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        assert loaded == data

    def test_all_runners_reject_nan(self, tmp_path):
        """All runners must reject NaN values."""
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import atomic_write_json as gemma3n_write
        from cetagostini.utils.pytensor.run_gemma_mlx import atomic_write_json as gemma_mlx_write
        from cetagostini.utils.pytensor.run_smollm2_pytensor import atomic_write_json as smollm2_write
        from cetagostini.utils.pytensor.run_smollm2_mlx import atomic_write_json as smollm2_mlx_write

        bad_data = {"value": float("nan")}

        for i, write_fn in enumerate([gemma3n_write, gemma_mlx_write, smollm2_write, smollm2_mlx_write]):
            dest = tmp_path / f"test_{i}.json"
            with pytest.raises(ValueError):
                write_fn(bad_data, dest)
            assert not dest.exists()

    def test_all_runners_reject_inf(self, tmp_path):
        """All runners must reject Inf values."""
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import atomic_write_json as gemma3n_write
        from cetagostini.utils.pytensor.run_gemma_mlx import atomic_write_json as gemma_mlx_write
        from cetagostini.utils.pytensor.run_smollm2_pytensor import atomic_write_json as smollm2_write
        from cetagostini.utils.pytensor.run_smollm2_mlx import atomic_write_json as smollm2_mlx_write

        bad_data = {"value": float("inf")}

        for i, write_fn in enumerate([gemma3n_write, gemma_mlx_write, smollm2_write, smollm2_mlx_write]):
            dest = tmp_path / f"test_{i}.json"
            with pytest.raises(ValueError):
                write_fn(bad_data, dest)
            assert not dest.exists()


# ---------------------------------------------------------------------------
# Tests: Version collection patterns
# ---------------------------------------------------------------------------


class TestVersionCollectionPatterns:
    """Verify version collection follows consistent patterns."""

    def test_gemma3n_pytensor_versions(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import collect_versions

        v = collect_versions()
        assert "python" in v
        assert "numpy" in v
        assert v["python"] != "unavailable"
        assert v["numpy"] != "unavailable"

    def test_gemma_mlx_versions(self):
        from cetagostini.utils.pytensor.run_gemma_mlx import collect_versions

        v = collect_versions()
        assert "python" in v
        assert "numpy" in v
        assert v["python"] != "unavailable"
        assert v["numpy"] != "unavailable"

    def test_smollm2_pytensor_versions(self):
        from cetagostini.utils.pytensor.run_smollm2_pytensor import collect_versions

        v = collect_versions()
        assert "python" in v
        assert "numpy" in v
        assert v["python"] != "unavailable"
        assert v["numpy"] != "unavailable"

    def test_smollm2_mlx_versions(self):
        from cetagostini.utils.pytensor.run_smollm2_mlx import collect_versions

        v = collect_versions()
        assert "python" in v
        assert "numpy" in v
        assert v["python"] != "unavailable"
        assert v["numpy"] != "unavailable"


# ---------------------------------------------------------------------------
# Tests: Report schema compatibility
# ---------------------------------------------------------------------------


class TestReportSchemaCompatibility:
    """Verify report schemas are compatible across runners."""

    def test_gemma3n_report_has_required_keys(self, tmp_path, monkeypatch):
        """Gemma3n PyTensor report must have required top-level keys."""
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
            EXPECTED_ARCHITECTURE,
            EXPECTED_BITS,
            EXPECTED_GROUP_SIZE,
            EXPECTED_MODEL_TYPE,
            EXPECTED_REPO,
            EXPECTED_REVISION,
            REQUIRED_FILES,
            build_file_manifest,
            check_optional_statuses,
            collect_versions,
            get_backend_info,
            sanitize_result,
        )
        import hashlib

        # Create minimal snapshot
        snap = tmp_path / EXPECTED_REVISION
        snap.mkdir()
        config = {
            "model_type": EXPECTED_MODEL_TYPE,
            "architectures": [EXPECTED_ARCHITECTURE],
            "quantization": {"bits": EXPECTED_BITS, "group_size": EXPECTED_GROUP_SIZE},
        }
        (snap / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (snap / "model.safetensors").write_bytes(b"\x00" * 64)
        (snap / "tokenizer.json").write_text("{}", encoding="utf-8")
        (snap / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (snap / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")

        # Patch manifest
        from cetagostini.utils.pytensor import run_gemma3n_pytensor
        patched = {}
        for name in REQUIRED_FILES:
            fpath = snap / name
            if fpath.exists() and fpath.stat().st_size > 0:
                patched[name] = {
                    "size": fpath.stat().st_size,
                    "sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
                }
        monkeypatch.setattr(run_gemma3n_pytensor, "EXPECTED_MANIFEST", patched)

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
            config_dict=config,
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
            memory={},
        )

        # Required top-level keys
        required_keys = ["model", "prompt", "backend", "versions", "device", "reference", "timing", "memory"]
        for key in required_keys:
            assert key in report, f"Missing required key: {key}"

    def test_smollm2_report_has_required_keys(self, tmp_path):
        """SmolLM2 PyTensor report must have required top-level keys."""
        from cetagostini.utils.pytensor.run_smollm2_pytensor import sanitize_result
        from types import SimpleNamespace

        config = SimpleNamespace(
            vocab_size=100,
            hidden_size=32,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            head_dim=16,
            intermediate_size=64,
            context_length=32,
            rms_eps=1e-5,
            rope_theta=100_000.0,
            bos=1,
            eos=2,
        )

        first_logits = np.random.default_rng(42).standard_normal(100).astype(np.float32)

        # Mock the gguf_weights imports
        import sys
        from unittest.mock import MagicMock
        mock_gguf = MagicMock()
        mock_gguf.EXPECTED_REPO = "test/repo"
        mock_gguf.EXPECTED_REVISION = "abc123"
        mock_gguf.EXPECTED_SHA256 = "deadbeef"
        mock_gguf.EXPECTED_ARCHITECTURE = "llama"
        mock_gguf.EXPECTED_GGUF_VERSION = 3
        sys.modules["cetagostini.utils.pytensor.gguf_weights"] = mock_gguf

        report = sanitize_result(
            model_path=Path("model.gguf"),
            config=config,
            versions={"python": "3.11", "numpy": "2.0"},
            prompt_text="test",
            formatted_text="<fmt>",
            token_ids=[1, 2, 3],
            generated_ids=[4, 5],
            generated_text="test output",
            first_logits=first_logits,
            first_token_id=4,
            timings={},
            memory={},
            cache_capacity=256,
            cache_status="ok",
        )

        # Required top-level keys
        required_keys = ["model", "config", "versions", "prompt", "generation", "timing", "memory", "cache"]
        for key in required_keys:
            assert key in report, f"Missing required key: {key}"


# ---------------------------------------------------------------------------
# Tests: Shared constants
# ---------------------------------------------------------------------------


class TestSharedConstants:
    """Verify shared constants are consistent."""

    def test_gemma3n_expected_revision(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import EXPECTED_REVISION
        from cetagostini.utils.pytensor.run_gemma_mlx import EXPECTED_REVISION as MLX_REVISION

        assert EXPECTED_REVISION == MLX_REVISION

    def test_gemma3n_expected_repo(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import EXPECTED_REPO
        from cetagostini.utils.pytensor.run_gemma_mlx import EXPECTED_REPO as MLX_REPO

        assert EXPECTED_REPO == MLX_REPO

    def test_gemma3n_valid_backends(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import VALID_BACKENDS

        assert VALID_BACKENDS == ("c", "numba")
        assert "mlx" not in VALID_BACKENDS

    def test_smollm2_default_prompt(self):
        from cetagostini.utils.pytensor.run_smollm2_pytensor import DEFAULT_PROMPT
        from cetagostini.utils.pytensor.run_smollm2_mlx import DEFAULT_PROMPT as MLX_PROMPT

        assert DEFAULT_PROMPT == MLX_PROMPT

    def test_smollm2_default_max_tokens(self):
        from cetagostini.utils.pytensor.run_smollm2_pytensor import DEFAULT_MAX_TOKENS
        from cetagostini.utils.pytensor.run_smollm2_mlx import DEFAULT_MAX_TOKENS as MLX_MAX

        assert DEFAULT_MAX_TOKENS == MLX_MAX
