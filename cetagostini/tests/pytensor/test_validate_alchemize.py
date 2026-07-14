"""Tests for cetagostini.utils.pytensor.validate_alchemize.

Focused tests for the Alchemize artifact validator covering provenance,
syntax, policy, required components, dequantization, and symbolic reshape checks.
"""

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from cetagostini.utils.pytensor import validate_alchemize as va


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify module-level constants are correctly defined."""

    def test_expected_artifact_sha256(self):
        assert va.EXPECTED_ARTIFACT_SHA256 == (
            "106cd2f597d7ab84bd955f733e8fdc5352311a0f76194e1a0cabf09575e57405"
        )

    def test_expected_artifact_lines(self):
        assert va.EXPECTED_ARTIFACT_LINES == 358

    def test_expected_gguf_sha256(self):
        assert va.EXPECTED_GGUF_SHA256 == (
            "2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d"
        )

    def test_expected_gguf_filename(self):
        assert va.EXPECTED_GGUF_FILENAME == "SmolLM2-135M-Instruct-Q4_K_M.gguf"

    def test_expected_gguf_revision(self):
        assert va.EXPECTED_GGUF_REVISION == (
            "09816acd5d99df7be770d85ea30822623dab342c"
        )

    def test_allowcisted_import_modules_is_frozenset(self):
        assert isinstance(va.ALLOWCISTED_IMPORT_MODULES, frozenset)

    def test_allowcisted_import_modules_contains_numpy(self):
        assert "numpy" in va.ALLOWCISTED_IMPORT_MODULES

    def test_allowcisted_import_modules_contains_pytensor(self):
        assert "pytensor" in va.ALLOWCISTED_IMPORT_MODULES

    def test_allowcisted_import_modules_contains_gguf(self):
        assert "gguf" in va.ALLOWCISTED_IMPORT_MODULES

    def test_forbidden_import_modules_is_frozenset(self):
        assert isinstance(va.FORBIDDEN_IMPORT_MODULES, frozenset)

    def test_forbidden_import_modules_contains_subprocess(self):
        assert "subprocess" in va.FORBIDDEN_IMPORT_MODULES

    def test_forbidden_call_names_is_frozenset(self):
        assert isinstance(va.FORBIDDEN_CALL_NAMES, frozenset)

    def test_forbidden_call_names_contains_eval(self):
        assert "eval" in va.FORBIDDEN_CALL_NAMES

    def test_required_components_is_frozenset(self):
        assert isinstance(va.REQUIRED_COMPONENTS, frozenset)

    def test_required_components_count(self):
        assert len(va.REQUIRED_COMPONENTS) == 16

    def test_required_components_contains_build_model(self):
        assert "build_model" in va.REQUIRED_COMPONENTS

    def test_allowed_result_keys_is_frozenset(self):
        assert isinstance(va.ALLOWED_RESULT_KEYS, frozenset)

    def test_artifact_dir_exists(self):
        assert va.ARTIFACT_DIR.exists()

    def test_raw_artifact_exists(self):
        assert (va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME).exists()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestComputeSha256:
    """Test SHA-256 computation."""

    def test_known_content(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert va.compute_sha256(f) == expected

    def test_artifact_hash_matches_provenance(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        assert va.compute_sha256(artifact) == va.EXPECTED_ARTIFACT_SHA256


class TestCountLines:
    """Test line counting."""

    def test_with_trailing_newline(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"line1\nline2\nline3\n")
        assert va.count_lines(f) == 3

    def test_without_trailing_newline(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"line1\nline2\nline3")
        assert va.count_lines(f) == 3

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"")
        assert va.count_lines(f) == 0

    def test_artifact_line_count(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        assert va.count_lines(artifact) == va.EXPECTED_ARTIFACT_LINES


class TestReadSource:
    """Test source reading."""

    def test_returns_string(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        source = va.read_source(artifact)
        assert isinstance(source, str)

    def test_contains_import_numpy(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        source = va.read_source(artifact)
        assert "import numpy" in source


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

class TestCheckGeneration:
    """Test generation provenance check."""

    def test_passes_on_real_artifact(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        result = va.check_generation(artifact)
        assert result["status"] == "PASS"
        assert result["hash_match"] is True
        assert result["line_match"] is True

    def test_fails_on_tampered_hash(self, tmp_path: Path):
        f = tmp_path / "tampered.py.txt"
        f.write_text("# tampered content\n", encoding="utf-8")
        result = va.check_generation(f)
        assert result["status"] == "FAIL"
        assert result["hash_match"] is False

    def test_result_keys(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        result = va.check_generation(artifact)
        expected_keys = {
            "status", "sha256", "expected_sha256", "lines",
            "expected_lines", "hash_match", "line_match",
        }
        assert set(result.keys()) == expected_keys


class TestCheckSyntax:
    """Test syntax validation."""

    def test_valid_python(self):
        result = va.check_syntax("x = 1\ny = 2\n")
        assert result["status"] == "PASS"

    def test_invalid_python(self):
        result = va.check_syntax("def foo(\n")
        assert result["status"] == "FAIL"
        assert "error" in result

    def test_real_artifact_syntax(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        source = va.read_source(artifact)
        result = va.check_syntax(source)
        assert result["status"] == "PASS"


class TestCheckPolicy:
    """Test import policy validation."""

    def test_clean_source_passes(self):
        source = "import numpy as np\nimport pytensor\n"
        result = va.check_policy(source)
        assert result["status"] == "PASS"

    def test_forbidden_import_fails(self):
        source = "import subprocess\n"
        result = va.check_policy(source)
        assert result["status"] == "FAIL"
        assert any("subprocess" in v for v in result["violations"])

    def test_eval_call_detected(self):
        source = "import numpy\nx = eval('1+1')\n"
        result = va.check_policy(source)
        assert result["status"] == "FAIL"
        assert any("eval" in c for c in result["forbidden_calls"])

    def test_real_artifact_policy(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        source = va.read_source(artifact)
        result = va.check_policy(source)
        assert result["status"] == "PASS"
        assert result["violations"] == []
        assert result["forbidden_calls"] == []

    def test_imports_listed(self):
        source = "import numpy\nimport pytensor\n"
        result = va.check_policy(source)
        assert len(result["imports"]) == 2

    def test_forbidden_alias_detected(self):
        source = "import numpy\ndanger = eval\n"
        result = va.check_policy(source)
        assert result["status"] == "FAIL"
        assert len(result["forbidden_aliases"]) > 0


class TestCheckRequiredComponents:
    """Test required component detection."""

    def test_all_present(self):
        source = "\n".join(
            f"def {name}(): pass" for name in va.REQUIRED_COMPONENTS
        )
        result = va.check_required_components(source)
        assert result["status"] == "PASS"
        assert result["missing"] == []

    def test_missing_detected(self):
        source = "def foo(): pass\n"
        result = va.check_required_components(source)
        assert result["status"] == "FAIL"
        assert len(result["missing"]) == len(va.REQUIRED_COMPONENTS)

    def test_real_artifact_components(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        source = va.read_source(artifact)
        result = va.check_required_components(source)
        assert result["status"] == "PASS"
        assert result["missing"] == []
        assert len(result["found"]) == len(va.REQUIRED_COMPONENTS)


class TestCheckDequantization:
    """Test dequantization static analysis."""

    def test_real_artifact_static_fail(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        source = va.read_source(artifact)
        result = va.check_dequantization(source)
        assert result["status"] == "STATIC_FAIL"
        assert result["has_gguf_dequantize_call"] is False
        assert result["materialize_reads_data_attr"] is True

    def test_with_dequantize_call(self):
        source = """
def materialize_tensor(tensor):
    import gguf
    return gguf.dequantize(tensor)
"""
        result = va.check_dequantization(source)
        assert result["status"] == "PASS"
        assert result["has_gguf_dequantize_call"] is True

    def test_missing_function(self):
        source = "def foo(): pass\n"
        result = va.check_dequantization(source)
        assert result["status"] == "FAIL"


class TestCheckSymbolicHeadReshape:
    """Test symbolic head reshape static analysis."""

    def test_real_artifact_static_fail(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        source = va.read_source(artifact)
        result = va.check_symbolic_head_reshape(source)
        assert result["status"] == "STATIC_FAIL"
        assert result["uses_split_dims"] is False
        assert result["uses_join_dims"] is False
        assert result["uses_raw_reshape"] is True

    def test_with_split_join(self):
        source = """
def attention_layer(x, wq, wk, wv, wo, cfg, cos, sin, causal_mask):
    import pytensor.tensor as pt
    q = pt.split_dims(x, shape=(3, 4), axis=-1)
    return pt.join_dims(q, start_axis=-2, n_axes=2)
"""
        result = va.check_symbolic_head_reshape(source)
        assert result["status"] == "PASS"
        assert result["uses_split_dims"] is True
        assert result["uses_join_dims"] is True

    def test_missing_function(self):
        source = "def foo(): pass\n"
        result = va.check_symbolic_head_reshape(source)
        assert result["status"] == "FAIL"


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestRunAllChecks:
    """Test the complete validation pipeline."""

    def test_all_checks_return_allowed_keys(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        result = va.run_all_checks(artifact)
        assert set(result.keys()).issubset(va.ALLOWED_RESULT_KEYS)

    def test_all_checks_generation_pass(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        result = va.run_all_checks(artifact)
        assert result["generation"]["status"] == "PASS"

    def test_all_checks_syntax_pass(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        result = va.run_all_checks(artifact)
        assert result["syntax"]["status"] == "PASS"

    def test_all_checks_policy_pass(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        result = va.run_all_checks(artifact)
        assert result["policy"]["status"] == "PASS"

    def test_all_checks_runtime_blocked(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        result = va.run_all_checks(artifact)
        assert result["runtime"]["status"] == "BLOCKED"

    def test_all_checks_semantic_unverified(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        result = va.run_all_checks(artifact)
        assert result["semantic"]["status"] == "UNVERIFIED"

    def test_all_checks_dequantization_static_fail(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        result = va.run_all_checks(artifact)
        assert result["dequantization"]["status"] == "STATIC_FAIL"

    def test_all_checks_reshape_static_fail(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        result = va.run_all_checks(artifact)
        assert result["symbolic_head_reshape"]["status"] == "STATIC_FAIL"

    def test_metadata_present(self):
        artifact = va.ARTIFACT_DIR / va.RAW_ARTIFACT_NAME
        result = va.run_all_checks(artifact)
        assert "validator" in result["metadata"]
        assert result["metadata"]["validator"] == "validate_alchemize.py"


class TestWriteJsonAtomic:
    """Test atomic JSON writing."""

    def test_writes_valid_json(self, tmp_path: Path):
        out = tmp_path / "result.json"
        data = {"status": "PASS", "count": 42}
        va.write_json_atomic(out, data)
        loaded = json.loads(out.read_text())
        assert loaded == data

    def test_file_ends_with_newline(self, tmp_path: Path):
        out = tmp_path / "result.json"
        va.write_json_atomic(out, {"key": "value"})
        assert out.read_text().endswith("\n")


class TestMainExitCodes:
    """Test CLI exit code behavior."""

    def test_returns_zero_on_real_artifact(self):
        code = va.main([])
        assert code == 0

    def test_returns_nonzero_on_missing_artifact(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(va, "ARTIFACT_DIR", tmp_path)
        code = va.main([])
        assert code != 0

    def test_output_flag_writes_json(self, tmp_path: Path):
        out = tmp_path / "result.json"
        code = va.main(["--output", str(out)])
        assert code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "generation" in data
