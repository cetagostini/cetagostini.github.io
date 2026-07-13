"""Artifact evidence utilities for Gemma 3n oracle validation.

Provides atomic writers, NPY manifest/verification helpers, and a versioned
schema constant for the Gemma 3n oracle artifact contract.  All APIs are
deliberately simple and stateless so that runner modules and cross-report
validators can reuse them without coupling.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Schema constant
# ---------------------------------------------------------------------------

GEMMA3N_ORACLE_SCHEMA_VERSION = "gemma3n-oracle-v1"
"""Versioned contract identifier for Gemma 3n oracle artifacts."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file using chunked reads."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of the directory containing *path*."""
    dir_fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _fsync_path(path: Path) -> None:
    """fsync a single file by path."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Atomic writers
# ---------------------------------------------------------------------------


def atomic_write_json(
    data: dict[str, Any],
    dest: Path,
    *,
    fsync_dir: bool = True,
) -> None:
    """Write JSON atomically via a partial file in *dest*'s parent directory.

    Uses ``allow_nan=False`` to reject any NaN/Inf values that would produce
    invalid JSON.  The partial file is flushed, fsynced, then atomically
    renamed via ``os.replace``.  On any failure the partial file is removed.

    Parameters
    ----------
    data : dict
        Data to serialize.
    dest : Path
        Destination file path.
    fsync_dir : bool
        When ``True`` (default), fsync the parent directory after rename.

    Raises
    ------
    ValueError
        If *data* contains non-finite float values.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(dest.parent), suffix=".tmp", prefix=".evidence_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(dest))
        if fsync_dir:
            _fsync_dir(dest.parent)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_npy(
    array: np.ndarray,
    dest: Path,
    *,
    fsync_dir: bool = True,
) -> None:
    """Write a little-endian float32 C-order ``.npy`` atomically.

    The array is validated to be ``<f4`` dtype and C-contiguous before
    writing.  The partial file is flushed, fsynced, then atomically renamed
    via ``os.replace``.  On any failure the partial file is removed.

    Parameters
    ----------
    array : np.ndarray
        Array to write.  Must be ``<f4`` dtype and C-contiguous.
    dest : Path
        Destination file path.
    fsync_dir : bool
        When ``True`` (default), fsync the parent directory after rename.

    Raises
    ------
    ValueError
        If *array* has wrong dtype or is not C-contiguous.
    """
    dest = Path(dest)

    if array.dtype != np.dtype("<f4"):
        raise ValueError(
            f"array dtype must be <f4 (little-endian float32), got {array.dtype}"
        )
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError("array must be C-contiguous")

    dest.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(dest.parent), suffix=".tmp", prefix=".evidence_"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(dest))
        if fsync_dir:
            _fsync_dir(dest.parent)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# NPY manifest / verification
# ---------------------------------------------------------------------------


def _npy_header_bytes(path: Path) -> int:
    """Return the number of bytes consumed by the ``.npy`` header."""
    with path.open("rb") as f:
        magic = f.read(6)
        if len(magic) < 6 or magic[:6] != b"\x93NUMPY":
            raise ValueError(f"{path.name}: not a valid .npy file (bad magic)")
        version_bytes = f.read(2)
        if len(version_bytes) < 2:
            raise ValueError(f"{path.name}: truncated .npy header")
        major, minor = version_bytes
        if major == 1:
            header_len_raw = f.read(2)
            if len(header_len_raw) < 2:
                raise ValueError(f"{path.name}: truncated .npy header length")
            header_len = struct.unpack("<H", header_len_raw)[0]
            return 6 + 2 + 2 + header_len
        elif major == 2 or major == 3:
            header_len_raw = f.read(4)
            if len(header_len_raw) < 4:
                raise ValueError(f"{path.name}: truncated .npy header length")
            header_len = struct.unpack("<I", header_len_raw)[0]
            return 6 + 2 + 4 + header_len
        else:
            raise ValueError(
                f"{path.name}: unsupported .npy version {major}.{minor}"
            )


def build_npy_manifest(path: Path) -> dict[str, Any]:
    """Build a raw artifact manifest for a ``.npy`` file.

    The manifest records everything needed to verify the artifact later:
    basename, format, dtype, byte order, memory order, shape, logical payload
    bytes, actual file size, file SHA-256, and canonical payload SHA-256
    (the raw data bytes after the header).

    Absolute paths are never included — only the basename.

    Parameters
    ----------
    path : Path
        Path to a ``.npy`` file.

    Returns
    -------
    dict
        Manifest dictionary.

    Raises
    ------
    ValueError
        If the file is not a valid little-endian float32 C-order ``.npy``.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")

    # Validate magic bytes before np.load to give a clear error for non-npy files
    _npy_header_bytes(path)

    file_size = path.stat().st_size
    file_sha256 = _sha256_file(path)

    arr = np.load(str(path), allow_pickle=False)

    if arr.dtype != np.dtype("<f4"):
        raise ValueError(
            f"{path.name}: dtype must be <f4, got {arr.dtype}"
        )
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{path.name}: array must be C-contiguous")

    header_bytes = _npy_header_bytes(path)
    payload_bytes = file_size - header_bytes
    expected_payload = int(arr.nbytes)

    if payload_bytes != expected_payload:
        raise ValueError(
            f"{path.name}: payload size mismatch "
            f"(header says {expected_payload}, file has {payload_bytes})"
        )

    canonical_sha256 = hashlib.sha256(arr.tobytes()).hexdigest()

    return {
        "basename": path.name,
        "format": "npy",
        "dtype": "<f4",
        "byte_order": "little",
        "order": "C",
        "shape": list(arr.shape),
        "payload_bytes": payload_bytes,
        "file_size": file_size,
        "file_sha256": file_sha256,
        "canonical_sha256": canonical_sha256,
    }


class ArtifactVerificationError(Exception):
    """Raised when an artifact fails verification against its manifest."""


def verify_npy_artifact(
    path: Path,
    manifest: dict[str, Any],
) -> np.ndarray:
    """Verify a ``.npy`` artifact against its manifest and return the array.

    Performs the following checks:

    1. File exists and has the expected size.
    2. File SHA-256 matches the manifest.
    3. ``np.load(allow_pickle=False)`` succeeds.
    4. No trailing bytes beyond the declared payload.
    5. dtype is ``<f4``, C-contiguous, and shape matches.
    6. Canonical payload SHA-256 matches.

    Parameters
    ----------
    path : Path
        Path to the ``.npy`` file.
    manifest : dict
        Manifest produced by :func:`build_npy_manifest`.

    Returns
    -------
    np.ndarray
        An owning C-contiguous ``<f4`` array.

    Raises
    ------
    ArtifactVerificationError
        If any verification check fails.
    """
    path = Path(path)

    if not path.exists():
        raise ArtifactVerificationError(f"artifact not found: {path}")

    actual_size = path.stat().st_size
    expected_size = manifest["file_size"]
    if actual_size != expected_size:
        raise ArtifactVerificationError(
            f"{path.name}: file size mismatch "
            f"(expected {expected_size}, got {actual_size})"
        )

    actual_sha = _sha256_file(path)
    expected_sha = manifest["file_sha256"]
    if actual_sha != expected_sha:
        raise ArtifactVerificationError(
            f"{path.name}: file SHA-256 mismatch "
            f"(expected {expected_sha}, got {actual_sha})"
        )

    try:
        arr = np.load(str(path), allow_pickle=False)
    except Exception as exc:
        raise ArtifactVerificationError(
            f"{path.name}: np.load failed: {exc}"
        ) from exc

    expected_shape = tuple(manifest["shape"])
    if arr.shape != expected_shape:
        raise ArtifactVerificationError(
            f"{path.name}: shape mismatch "
            f"(expected {expected_shape}, got {arr.shape})"
        )

    if arr.dtype != np.dtype("<f4"):
        raise ArtifactVerificationError(
            f"{path.name}: dtype mismatch "
            f"(expected <f4, got {arr.dtype})"
        )

    if not arr.flags["C_CONTIGUOUS"]:
        raise ArtifactVerificationError(
            f"{path.name}: array is not C-contiguous"
        )

    header_bytes = _npy_header_bytes(path)
    payload_bytes = actual_size - header_bytes
    expected_payload = manifest["payload_bytes"]
    if payload_bytes != expected_payload:
        raise ArtifactVerificationError(
            f"{path.name}: trailing bytes detected "
            f"(expected {expected_payload} payload bytes, got {payload_bytes})"
        )

    canonical_sha = hashlib.sha256(arr.tobytes()).hexdigest()
    expected_canonical = manifest["canonical_sha256"]
    if canonical_sha != expected_canonical:
        raise ArtifactVerificationError(
            f"{path.name}: canonical payload SHA-256 mismatch "
            f"(expected {expected_canonical}, got {canonical_sha})"
        )

    return np.ascontiguousarray(arr, dtype=np.dtype("<f4"))
