"""Tests for the evidence artifact utilities.

Covers:
- Atomic JSON writer roundtrip and error cleanup
- Atomic NPY writer roundtrip and error cleanup
- NPY manifest building and verification roundtrip
- Every corruption class: trailing bytes, wrong dtype, wrong order,
  wrong shape, wrong size, wrong file hash, wrong canonical hash,
  malformed/truncated arrays
- No absolute path leakage in manifests
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from cetagostini.utils.pytensor.evidence import (
    GEMMA3N_ORACLE_SCHEMA_VERSION,
    ArtifactVerificationError,
    atomic_write_json,
    atomic_write_npy,
    build_npy_manifest,
    verify_npy_artifact,
)


# ---------------------------------------------------------------------------
# Schema constant
# ---------------------------------------------------------------------------


class TestSchemaConstant:
    def test_schema_version_is_string(self):
        assert isinstance(GEMMA3N_ORACLE_SCHEMA_VERSION, str)
        assert "gemma3n-oracle" in GEMMA3N_ORACLE_SCHEMA_VERSION

    def test_schema_version_has_version(self):
        assert "v1" in GEMMA3N_ORACLE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Atomic JSON writer
# ---------------------------------------------------------------------------


class TestAtomicWriteJson:
    def test_roundtrip(self, tmp_path):
        dest = tmp_path / "test.json"
        data = {"key": "value", "number": 42, "nested": {"a": [1, 2, 3]}}
        atomic_write_json(data, dest)
        assert dest.exists()
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        assert loaded == data

    def test_rejects_nan(self, tmp_path):
        dest = tmp_path / "nan.json"
        with pytest.raises(ValueError):
            atomic_write_json({"value": float("nan")}, dest)
        assert not dest.exists()

    def test_rejects_inf(self, tmp_path):
        dest = tmp_path / "inf.json"
        with pytest.raises(ValueError):
            atomic_write_json({"value": float("inf")}, dest)
        assert not dest.exists()

    def test_rejects_negative_inf(self, tmp_path):
        dest = tmp_path / "ninf.json"
        with pytest.raises(ValueError):
            atomic_write_json({"value": float("-inf")}, dest)
        assert not dest.exists()

    def test_cleanup_on_serialization_error(self, tmp_path):
        """Partial file must be removed when serialization fails."""
        dest = tmp_path / "bad.json"

        class Unserializable:
            pass

        with pytest.raises(TypeError):
            atomic_write_json({"obj": Unserializable()}, dest)
        assert not dest.exists()
        # Also check no leftover temp files
        leftovers = list(tmp_path.glob(".evidence_*.tmp"))
        assert leftovers == []

    def test_creates_parent_directories(self, tmp_path):
        dest = tmp_path / "sub" / "dir" / "test.json"
        atomic_write_json({"key": "value"}, dest)
        assert dest.exists()

    def test_overwrites_existing_file(self, tmp_path):
        dest = tmp_path / "test.json"
        atomic_write_json({"version": 1}, dest)
        atomic_write_json({"version": 2}, dest)
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        assert loaded["version"] == 2

    def test_fsync_dir_false(self, tmp_path):
        dest = tmp_path / "test.json"
        atomic_write_json({"key": "value"}, dest, fsync_dir=False)
        assert dest.exists()


# ---------------------------------------------------------------------------
# Atomic NPY writer
# ---------------------------------------------------------------------------


class TestAtomicWriteNpy:
    def test_roundtrip(self, tmp_path):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dest = tmp_path / "test.npy"
        atomic_write_npy(arr, dest)
        assert dest.exists()
        loaded = np.load(str(dest), allow_pickle=False)
        np.testing.assert_array_equal(loaded, arr)

    def test_2d_array(self, tmp_path):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        dest = tmp_path / "test2d.npy"
        atomic_write_npy(arr, dest)
        loaded = np.load(str(dest), allow_pickle=False)
        np.testing.assert_array_equal(loaded, arr)
        assert loaded.shape == (3, 4)

    def test_rejects_wrong_dtype(self, tmp_path):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        dest = tmp_path / "bad.npy"
        with pytest.raises(ValueError, match="dtype"):
            atomic_write_npy(arr, dest)
        assert not dest.exists()

    def test_rejects_int_dtype(self, tmp_path):
        arr = np.array([1, 2, 3], dtype=np.int32)
        dest = tmp_path / "bad.npy"
        with pytest.raises(ValueError, match="dtype"):
            atomic_write_npy(arr, dest)
        assert not dest.exists()

    def test_rejects_fortran_order(self, tmp_path):
        arr = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        dest = tmp_path / "bad.npy"
        with pytest.raises(ValueError, match="C-contiguous"):
            atomic_write_npy(arr, dest)
        assert not dest.exists()

    def test_cleanup_on_error(self, tmp_path):
        """Partial file must be removed when writing fails."""
        arr = np.array([1.0], dtype=np.float64)  # wrong dtype
        dest = tmp_path / "bad.npy"
        with pytest.raises(ValueError):
            atomic_write_npy(arr, dest)
        leftovers = list(tmp_path.glob(".evidence_*.tmp"))
        assert leftovers == []

    def test_creates_parent_directories(self, tmp_path):
        arr = np.array([1.0], dtype=np.float32)
        dest = tmp_path / "sub" / "dir" / "test.npy"
        atomic_write_npy(arr, dest)
        assert dest.exists()


# ---------------------------------------------------------------------------
# NPY manifest
# ---------------------------------------------------------------------------


class TestBuildNpyManifest:
    def test_roundtrip_manifest(self, tmp_path):
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        path = tmp_path / "oracle_logits.npy"
        atomic_write_npy(arr, path)

        manifest = build_npy_manifest(path)

        assert manifest["basename"] == "oracle_logits.npy"
        assert manifest["format"] == "npy"
        assert manifest["dtype"] == "<f4"
        assert manifest["byte_order"] == "little"
        assert manifest["order"] == "C"
        assert manifest["shape"] == [2, 3, 4]
        assert manifest["payload_bytes"] == arr.nbytes
        assert manifest["file_size"] == path.stat().st_size
        assert len(manifest["file_sha256"]) == 64
        assert len(manifest["canonical_sha256"]) == 64

    def test_no_absolute_paths(self, tmp_path):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        path = tmp_path / "test.npy"
        atomic_write_npy(arr, path)

        manifest = build_npy_manifest(path)
        manifest_str = json.dumps(manifest)
        assert str(tmp_path) not in manifest_str
        assert "/" not in manifest["basename"]

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_npy_manifest(tmp_path / "nonexistent.npy")

    def test_rejects_wrong_dtype(self, tmp_path):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        path = tmp_path / "bad.npy"
        np.save(str(path), arr, allow_pickle=False)
        with pytest.raises(ValueError, match="dtype"):
            build_npy_manifest(path)

    def test_rejects_fortran_order(self, tmp_path):
        arr = np.asfortranarray(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )
        path = tmp_path / "bad.npy"
        np.save(str(path), arr, allow_pickle=False)
        with pytest.raises(ValueError, match="C-contiguous"):
            build_npy_manifest(path)

    def test_rejects_non_npy_file(self, tmp_path):
        path = tmp_path / "not_npy.npy"
        path.write_bytes(b"this is not a npy file")
        with pytest.raises(ValueError, match="not a valid"):
            build_npy_manifest(path)


# ---------------------------------------------------------------------------
# NPY verification
# ---------------------------------------------------------------------------


class TestVerifyNpyArtifact:
    def test_successful_roundtrip(self, tmp_path):
        arr = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        path = tmp_path / "logits.npy"
        atomic_write_npy(arr, path)

        manifest = build_npy_manifest(path)
        result = verify_npy_artifact(path, manifest)

        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.dtype("<f4")
        assert result.flags["C_CONTIGUOUS"]

    def test_file_not_found(self, tmp_path):
        manifest = {
            "basename": "missing.npy",
            "format": "npy",
            "dtype": "<f4",
            "byte_order": "little",
            "order": "C",
            "shape": [1],
            "payload_bytes": 4,
            "file_size": 100,
            "file_sha256": "a" * 64,
            "canonical_sha256": "b" * 64,
        }
        with pytest.raises(ArtifactVerificationError, match="not found"):
            verify_npy_artifact(tmp_path / "missing.npy", manifest)

    def test_wrong_file_size(self, tmp_path):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        path = tmp_path / "test.npy"
        atomic_write_npy(arr, path)

        manifest = build_npy_manifest(path)
        manifest["file_size"] = manifest["file_size"] + 100

        with pytest.raises(ArtifactVerificationError, match="size mismatch"):
            verify_npy_artifact(path, manifest)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("basename", "other.npy"),
            ("format", "hdf5"),
            ("byte_order", "big"),
            ("order", "F"),
        ],
    )
    def test_manifest_metadata_mismatch(self, tmp_path, field, value):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        path = tmp_path / "test.npy"
        atomic_write_npy(arr, path)
        manifest = build_npy_manifest(path)
        manifest[field] = value

        with pytest.raises(ArtifactVerificationError, match=field):
            verify_npy_artifact(path, manifest)

    def test_wrong_file_hash(self, tmp_path):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        path = tmp_path / "test.npy"
        atomic_write_npy(arr, path)

        manifest = build_npy_manifest(path)
        manifest["file_sha256"] = "0" * 64

        with pytest.raises(ArtifactVerificationError, match="SHA-256"):
            verify_npy_artifact(path, manifest)

    def test_wrong_shape(self, tmp_path):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = tmp_path / "test.npy"
        atomic_write_npy(arr, path)

        manifest = build_npy_manifest(path)
        manifest["shape"] = [4, 3]  # wrong shape, same total elements

        with pytest.raises(ArtifactVerificationError, match="shape"):
            verify_npy_artifact(path, manifest)

    def test_wrong_dtype(self, tmp_path):
        """Write a float64 .npy manually and verify rejection."""
        arr_f64 = np.array([1.0, 2.0], dtype=np.float64)
        path = tmp_path / "test.npy"
        np.save(str(path), arr_f64, allow_pickle=False)

        # Build a manifest that claims <f4 but the file is actually f64
        # We need to construct a valid manifest for the f64 file first,
        # then tamper with the dtype field
        file_size = path.stat().st_size
        file_sha = build_npy_manifest.__module__  # just need a hash

        import hashlib
        file_sha = hashlib.sha256(path.read_bytes()).hexdigest()

        manifest = {
            "basename": "test.npy",
            "format": "npy",
            "dtype": "<f4",
            "byte_order": "little",
            "order": "C",
            "shape": [2],
            "payload_bytes": 16,  # 2 * 8 bytes for f64
            "file_size": file_size,
            "file_sha256": file_sha,
            "canonical_sha256": hashlib.sha256(arr_f64.tobytes()).hexdigest(),
        }

        with pytest.raises(ArtifactVerificationError, match="dtype"):
            verify_npy_artifact(path, manifest)

    def test_trailing_bytes(self, tmp_path):
        """Append extra bytes to a valid .npy and verify rejection."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        path = tmp_path / "test.npy"
        atomic_write_npy(arr, path)

        manifest = build_npy_manifest(path)

        # Append trailing bytes
        with path.open("ab") as f:
            f.write(b"\x00" * 16)

        with pytest.raises(ArtifactVerificationError, match="size mismatch"):
            verify_npy_artifact(path, manifest)

    def test_wrong_canonical_hash(self, tmp_path):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        path = tmp_path / "test.npy"
        atomic_write_npy(arr, path)

        manifest = build_npy_manifest(path)
        manifest["canonical_sha256"] = "0" * 64

        # The file hash still matches, but canonical won't
        with pytest.raises(ArtifactVerificationError, match="canonical"):
            verify_npy_artifact(path, manifest)

    def test_truncated_file(self, tmp_path):
        """Truncate a valid .npy and verify rejection."""
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        path = tmp_path / "test.npy"
        atomic_write_npy(arr, path)

        manifest = build_npy_manifest(path)

        # Truncate the file (remove last 40 bytes = 10 float32 values)
        data = path.read_bytes()
        path.write_bytes(data[:-40])

        # File size won't match
        with pytest.raises(ArtifactVerificationError, match="size mismatch"):
            verify_npy_artifact(path, manifest)

    def test_malformed_npy(self, tmp_path):
        """Write garbage that looks like .npy magic but is invalid."""
        path = tmp_path / "bad.npy"
        # Valid magic + version but garbage header
        path.write_bytes(b"\x93NUMPY\x01\x00\x05\x00garbage")

        manifest = {
            "basename": "bad.npy",
            "format": "npy",
            "dtype": "<f4",
            "byte_order": "little",
            "order": "C",
            "shape": [1],
            "payload_bytes": 4,
            "file_size": path.stat().st_size,
            "file_sha256": "a" * 64,
            "canonical_sha256": "b" * 64,
        }

        # File hash won't match first
        with pytest.raises(ArtifactVerificationError):
            verify_npy_artifact(path, manifest)

    def test_returns_owning_array(self, tmp_path):
        """The returned array must be a standalone copy (not memory-mapped)."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        path = tmp_path / "test.npy"
        atomic_write_npy(arr, path)

        manifest = build_npy_manifest(path)
        result = verify_npy_artifact(path, manifest)

        # Modify the result — should not affect the file
        result[0] = 999.0
        reloaded = np.load(str(path), allow_pickle=False)
        assert reloaded[0] == 1.0
