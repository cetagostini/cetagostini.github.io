"""Tests for cetagostini.utils.pytensor.validate_pytensor_mlx.

Focused tests for the MLX compatibility matrix validator covering
deterministic arrays, comparison logic, expected outcomes matching,
discrepancy detection, and metadata collection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cetagostini.utils.pytensor import validate_pytensor_mlx as vpm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify module-level constants are correctly defined."""

    def test_allowed_keys_is_frozenset(self):
        assert isinstance(vpm.ALLOWED_KEYS, frozenset)

    def test_allowed_keys_contains_metadata(self):
        assert "metadata" in vpm.ALLOWED_KEYS

    def test_allowed_keys_contains_matrix_match(self):
        assert "matrix_match" in vpm.ALLOWED_KEYS

    def test_allowed_keys_contains_discrepancies(self):
        assert "discrepancies" in vpm.ALLOWED_KEYS

    def test_expected_outcomes_is_dict(self):
        assert isinstance(vpm.EXPECTED_OUTCOMES, dict)

    def test_expected_outcomes_has_five_checks(self):
        assert len(vpm.EXPECTED_OUTCOMES) == 5

    def test_expected_outcomes_linear_rank2(self):
        assert vpm.EXPECTED_OUTCOMES["linear_rank2"] == {
            "fast_compile": True,
            "mlx": True,
        }

    def test_expected_outcomes_linear_rank3(self):
        assert vpm.EXPECTED_OUTCOMES["linear_rank3"] == {
            "fast_compile": True,
            "mlx": False,
        }

    def test_expected_outcomes_linear_rank4(self):
        assert vpm.EXPECTED_OUTCOMES["linear_rank4"] == {
            "fast_compile": True,
            "mlx": False,
        }

    def test_expected_outcomes_split_join(self):
        assert vpm.EXPECTED_OUTCOMES["split_join"] == {"error": True}

    def test_expected_outcomes_multihead_attention(self):
        assert vpm.EXPECTED_OUTCOMES["multihead_attention"] == {"error": True}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestDeterministicArray:
    """Test deterministic array generation."""

    def test_shape(self):
        arr = vpm.deterministic_array((3, 4), seed=42)
        assert arr.shape == (3, 4)

    def test_dtype_float32(self):
        arr = vpm.deterministic_array((2, 3), seed=42)
        assert arr.dtype == np.float32

    def test_no_zeros(self):
        arr = vpm.deterministic_array((10, 10), seed=42)
        assert not np.any(arr == 0)

    def test_deterministic(self):
        arr1 = vpm.deterministic_array((3, 4), seed=42)
        arr2 = vpm.deterministic_array((3, 4), seed=42)
        np.testing.assert_array_equal(arr1, arr2)

    def test_different_seeds_differ(self):
        arr1 = vpm.deterministic_array((3, 4), seed=42)
        arr2 = vpm.deterministic_array((3, 4), seed=99)
        assert not np.array_equal(arr1, arr2)

    def test_asymmetric_rows(self):
        arr = vpm.deterministic_array((5, 3), seed=42)
        # No two rows should be identical
        for i in range(arr.shape[0]):
            for j in range(i + 1, arr.shape[0]):
                assert not np.array_equal(arr[i], arr[j])

    def test_custom_bounds(self):
        arr = vpm.deterministic_array((100,), seed=42, low=0.5, high=1.0)
        assert np.all(arr >= 0.5)
        assert np.all(arr <= 1.1)  # ramp adds small amount


class TestCompareArrays:
    """Test array comparison."""

    def test_matching_arrays(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = vpm.compare_arrays(a, a, "test")
        assert result["pass"] is True
        assert result["allclose"] is True

    def test_shape_mismatch(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = vpm.compare_arrays(a, b, "test")
        assert result["pass"] is False
        assert result["max_abs_diff"] is None

    def test_value_mismatch(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 99.0], dtype=np.float32)
        result = vpm.compare_arrays(a, b, "test")
        assert result["pass"] is False
        assert result["max_abs_diff"] is not None
        assert result["max_abs_diff"] > 1.0

    def test_close_arrays_pass(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = a + 1e-6
        result = vpm.compare_arrays(a, b, "test")
        assert result["pass"] is True

    def test_result_keys(self):
        a = np.array([1.0], dtype=np.float32)
        result = vpm.compare_arrays(a, a, "test")
        expected_keys = {
            "label", "pass", "expected_shape", "actual_shape",
            "max_abs_diff", "allclose",
        }
        assert set(result.keys()) == expected_keys

    def test_label_preserved(self):
        a = np.array([1.0], dtype=np.float32)
        result = vpm.compare_arrays(a, a, "my_label")
        assert result["label"] == "my_label"


# ---------------------------------------------------------------------------
# Expected outcomes matching
# ---------------------------------------------------------------------------

class TestExtractBackendPass:
    """Test backend pass extraction."""

    def test_nested_dict_style(self):
        entry = {"fast_compile": {"pass": True}}
        assert vpm._extract_backend_pass(entry, "fast_compile") is True

    def test_nested_dict_style_fail(self):
        entry = {"mlx": {"pass": False}}
        assert vpm._extract_backend_pass(entry, "mlx") is False

    def test_mha_style(self):
        entry = {"fast_compile_pass": True}
        assert vpm._extract_backend_pass(entry, "fast_compile") is True

    def test_missing_backend(self):
        entry = {"other": {"pass": True}}
        assert vpm._extract_backend_pass(entry, "fast_compile") is False


class TestMatchesExpectedOutcomes:
    """Test compatibility matrix matching."""

    def test_perfect_match(self):
        result = {
            "linear_rank2": {
                "fast_compile": {"pass": True},
                "mlx": {"pass": True},
            },
            "linear_rank3": {
                "fast_compile": {"pass": True},
                "mlx": {"pass": False},
            },
            "linear_rank4": {
                "fast_compile": {"pass": True},
                "mlx": {"pass": False},
            },
            "split_join": {"error": "No MLX conversion", "pass": False},
            "multihead_attention": {"error": "No MLX conversion", "pass": False},
        }
        match, discrepancies = vpm.matches_expected_outcomes(result)
        assert match is True
        assert discrepancies == []

    def test_unexpected_pass(self):
        result = {
            "linear_rank2": {
                "fast_compile": {"pass": True},
                "mlx": {"pass": True},
            },
            "linear_rank3": {
                "fast_compile": {"pass": True},
                "mlx": {"pass": True},  # Expected False!
            },
            "linear_rank4": {
                "fast_compile": {"pass": True},
                "mlx": {"pass": False},
            },
            "split_join": {"error": "No MLX conversion", "pass": False},
            "multihead_attention": {"error": "No MLX conversion", "pass": False},
        }
        match, discrepancies = vpm.matches_expected_outcomes(result)
        assert match is False
        assert len(discrepancies) > 0
        assert any("linear_rank3" in d for d in discrepancies)

    def test_unexpected_error(self):
        result = {
            "linear_rank2": {"error": "something broke"},
            "linear_rank3": {
                "fast_compile": {"pass": True},
                "mlx": {"pass": False},
            },
            "linear_rank4": {
                "fast_compile": {"pass": True},
                "mlx": {"pass": False},
            },
            "split_join": {"error": "No MLX conversion", "pass": False},
            "multihead_attention": {"error": "No MLX conversion", "pass": False},
        }
        match, discrepancies = vpm.matches_expected_outcomes(result)
        assert match is False
        assert any("linear_rank2" in d for d in discrepancies)

    def test_error_expected_but_got_result(self):
        result = {
            "linear_rank2": {
                "fast_compile": {"pass": True},
                "mlx": {"pass": True},
            },
            "linear_rank3": {
                "fast_compile": {"pass": True},
                "mlx": {"pass": False},
            },
            "linear_rank4": {
                "fast_compile": {"pass": True},
                "mlx": {"pass": False},
            },
            "split_join": {
                "fast_compile": {"pass": True},
                "mlx": {"pass": True},
            },
            "multihead_attention": {"error": "No MLX conversion", "pass": False},
        }
        match, discrepancies = vpm.matches_expected_outcomes(result)
        assert match is False
        assert any("split_join" in d for d in discrepancies)

    def test_missing_check(self):
        result = {
            "linear_rank2": {
                "fast_compile": {"pass": True},
                "mlx": {"pass": True},
            },
            # linear_rank3 missing entirely
            "linear_rank4": {
                "fast_compile": {"pass": True},
                "mlx": {"pass": False},
            },
            "split_join": {"error": "No MLX conversion", "pass": False},
            "multihead_attention": {"error": "No MLX conversion", "pass": False},
        }
        match, discrepancies = vpm.matches_expected_outcomes(result)
        assert match is False


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestCollectMetadata:
    """Test metadata collection."""

    def test_returns_dict(self):
        try:
            result = vpm.collect_metadata()
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("pytensor/pytensor_ml not available")

    def test_contains_python_version(self):
        try:
            result = vpm.collect_metadata()
            assert "python" in result
            assert isinstance(result["python"], str)
        except ImportError:
            pytest.skip("pytensor/pytensor_ml not available")

    def test_contains_platform(self):
        try:
            result = vpm.collect_metadata()
            assert "platform" in result
            assert "machine" in result
        except ImportError:
            pytest.skip("pytensor/pytensor_ml not available")


# ---------------------------------------------------------------------------
# Write JSON
# ---------------------------------------------------------------------------

class TestWriteJsonAtomic:
    """Test atomic JSON writing."""

    def test_writes_valid_json(self, tmp_path: Path):
        out = tmp_path / "result.json"
        data = {"matrix_match": True, "discrepancies": []}
        vpm.write_json_atomic(out, data)
        loaded = json.loads(out.read_text())
        assert loaded == data

    def test_file_ends_with_newline(self, tmp_path: Path):
        out = tmp_path / "result.json"
        vpm.write_json_atomic(out, {"key": "value"})
        assert out.read_text().endswith("\n")


# ---------------------------------------------------------------------------
# NumPy linear reference
# ---------------------------------------------------------------------------

class TestNumpyLinear:
    """Test the pure NumPy linear reference."""

    def test_output_shape_rank2(self):
        x = np.ones((3, 4), dtype=np.float32)
        W = np.ones((4, 5), dtype=np.float32)
        b = np.zeros(5, dtype=np.float32)
        out = vpm._numpy_linear(x, W, b)
        assert out.shape == (3, 5)

    def test_output_shape_rank3(self):
        x = np.ones((2, 3, 4), dtype=np.float32)
        W = np.ones((4, 5), dtype=np.float32)
        b = np.zeros(5, dtype=np.float32)
        out = vpm._numpy_linear(x, W, b)
        assert out.shape == (2, 3, 5)

    def test_correctness(self):
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        W = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        b = np.array([0.5, 0.5], dtype=np.float32)
        out = vpm._numpy_linear(x, W, b)
        expected = np.array([[1.5, 2.5]], dtype=np.float32)
        np.testing.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# Integration (requires pytensor_ml + mlx)
# ---------------------------------------------------------------------------

class TestRunAllChecksIntegration:
    """Integration tests that require pytensor_ml and mlx with the exact
    MLX backend configuration used during the article generation.

    These tests verify the full pipeline produces a result matching the
    expected compatibility matrix. They are skipped if the required
    packages are not available or if the MLX backend import path differs
    from the one used during the original evidence collection.
    """

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        try:
            import pytensor
            import pytensor_ml
            import mlx.core
            # Verify the exact MLX import path used by make_mlx_mode
            from pytensor.link.mlx import MLX
            from pytensor.link.mlx.linker import MLXLinker
        except (ImportError, AttributeError):
            pytest.skip(
                "pytensor_ml, mlx, or pytensor.link.mlx.MLX not available "
                "in this environment"
            )

    def test_result_keys(self):
        result = vpm.run_all_checks()
        assert set(result.keys()).issubset(vpm.ALLOWED_KEYS)

    def test_matrix_match(self):
        result = vpm.run_all_checks()
        assert result["matrix_match"] is True

    def test_no_discrepancies(self):
        result = vpm.run_all_checks()
        assert result["discrepancies"] == []

    def test_expected_outcomes_in_result(self):
        result = vpm.run_all_checks()
        assert result["expected_outcomes"] == vpm.EXPECTED_OUTCOMES

    def test_main_returns_zero(self):
        code = vpm.main([])
        assert code == 0
