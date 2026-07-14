"""Load SmolLM2-135M-Instruct weights from a GGUF artifact.

Validates the artifact against a known manifest, dequantizes tensors,
undoes the GGUF llama RoPE permutation for Q/K projections, and returns
contiguous float32 arrays in a structured dataclass.

Usage::

    python -m cetagostini.utils.pytensor.gguf_weights --model /path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf
    python -m cetagostini.utils.pytensor.gguf_weights --model /path/to/model.gguf --output results/weights.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Pinned artifact constants
# ---------------------------------------------------------------------------

EXPECTED_FILENAME = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
EXPECTED_SIZE = 105454432
EXPECTED_SHA256 = "2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d"
EXPECTED_GGUF_VERSION = 3
EXPECTED_TENSOR_COUNT = 272
EXPECTED_ARCHITECTURE = "llama"
EXPECTED_REVISION = "09816acd5d99df7be770d85ea30822623dab342c"
EXPECTED_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
EXPECTED_QUANT_TYPE_COUNTS = {
    "Q8_0": 15,
    "F32": 61,
    "Q6_K": 14,
    "Q5_0": 166,
    "Q4_K": 16,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmolLM2Config:
    """Frozen configuration for SmolLM2-135M-Instruct."""

    vocab_size: int = 49152
    hidden_size: int = 576
    n_layers: int = 30
    n_heads: int = 9
    n_kv_heads: int = 3
    head_dim: int = 64
    intermediate_size: int = 1536
    context_length: int = 8192
    rms_eps: float = 1e-5
    rope_theta: float = 100000.0


# ---------------------------------------------------------------------------
# Weights container
# ---------------------------------------------------------------------------


@dataclass
class SmolLM2Weights:
    """Loaded weights with all arrays as contiguous float32.

    Attributes
    ----------
    config : SmolLM2Config
        Model configuration.
    token_embedding : np.ndarray
        Token embedding table, shape ``[vocab_size, hidden_size]``.
    layers : list[dict[str, np.ndarray]]
        Per-layer weights (one dict per layer).
    final_norm : np.ndarray
        Final RMSNorm weights, shape ``[hidden_size]``.
    reader : Any
        GGUF reader (kept open for manifest building).
    """

    config: SmolLM2Config
    token_embedding: np.ndarray
    layers: list[dict[str, np.ndarray]]
    final_norm: np.ndarray
    reader: Any = field(repr=False)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hex digest string.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_artifact(path: Path, verify_hash: bool = True) -> None:
    """Validate the GGUF artifact against the expected manifest.

    Parameters
    ----------
    path : Path
        Path to the GGUF file.
    verify_hash : bool
        Whether to verify the SHA-256 hash.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If size, hash, version, tensor count, or architecture do not match.
    """
    if not path.exists():
        raise FileNotFoundError(f"GGUF file not found: {path}")
    actual_size = path.stat().st_size
    if actual_size != EXPECTED_SIZE:
        raise ValueError(f"Expected file size {EXPECTED_SIZE}, got {actual_size}")
    if verify_hash:
        actual_hash = compute_sha256(path)
        if actual_hash != EXPECTED_SHA256:
            raise ValueError(
                f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_hash}"
            )


def validate_gguf_metadata(reader: Any) -> None:
    """Validate GGUF metadata fields.

    ``GGUF.version``, ``GGUF.tensor_count``, and ``general.architecture``
    are required — their absence is an error.

    Parameters
    ----------
    reader : GGUFReader
        An opened GGUF reader.

    Raises
    ------
    ValueError
        If a required field is missing, or if version, tensor count, or
        architecture do not match expected values.
    """
    version_field = reader.get_field("GGUF.version")
    if version_field is None:
        raise ValueError("GGUF.version field is missing")
    version = int(version_field.parts[version_field.data[0]])
    if version != EXPECTED_GGUF_VERSION:
        raise ValueError(
            f"Expected GGUF version {EXPECTED_GGUF_VERSION}, got {version}"
        )

    tensor_count_field = reader.get_field("GGUF.tensor_count")
    if tensor_count_field is None:
        raise ValueError("GGUF.tensor_count field is missing")
    tensor_count = int(tensor_count_field.parts[tensor_count_field.data[0]])
    if tensor_count != EXPECTED_TENSOR_COUNT:
        raise ValueError(
            f"Expected tensor count {EXPECTED_TENSOR_COUNT}, got {tensor_count}"
        )

    arch_field = reader.get_field("general.architecture")
    if arch_field is None:
        raise ValueError("general.architecture field is missing")
    arch = str(arch_field.parts[arch_field.data[0]])
    if arch != EXPECTED_ARCHITECTURE:
        raise ValueError(
            f"Expected architecture '{EXPECTED_ARCHITECTURE}', got '{arch}'"
        )


def _validate_quant_distribution(reader: Any) -> None:
    """Validate that the quant type distribution matches expectations.

    Parameters
    ----------
    reader : GGUFReader
        An opened GGUF reader.

    Raises
    ------
    ValueError
        If the actual quant type counts do not exactly match
        ``EXPECTED_QUANT_TYPE_COUNTS``.
    """
    actual_counts: dict[str, int] = {}
    for tensor in reader.tensors:
        dtype_name = tensor.tensor_type.name
        actual_counts[dtype_name] = actual_counts.get(dtype_name, 0) + 1
    if actual_counts != EXPECTED_QUANT_TYPE_COUNTS:
        raise ValueError(
            f"Quant type distribution mismatch: expected {EXPECTED_QUANT_TYPE_COUNTS}, "
            f"got {actual_counts}"
        )


def _build_expected_tensor_names(config: SmolLM2Config) -> set[str]:
    """Build the set of expected tensor names for the model.

    Parameters
    ----------
    config : SmolLM2Config
        Model configuration.

    Returns
    -------
    set[str]
        Expected tensor names.
    """
    names = {
        "token_embd.weight",
        "output_norm.weight",
    }
    for i in range(config.n_layers):
        names.update(
            {
                f"blk.{i}.attn_norm.weight",
                f"blk.{i}.attn_q.weight",
                f"blk.{i}.attn_k.weight",
                f"blk.{i}.attn_v.weight",
                f"blk.{i}.attn_output.weight",
                f"blk.{i}.ffn_norm.weight",
                f"blk.{i}.ffn_gate.weight",
                f"blk.{i}.ffn_up.weight",
                f"blk.{i}.ffn_down.weight",
            }
        )
    return names


def validate_tensor_inventory(reader: Any, config: SmolLM2Config) -> None:
    """Validate that all expected tensor names exist with correct shapes/types.

    Also validates that the quant type distribution across all tensors
    matches ``EXPECTED_QUANT_TYPE_COUNTS``.

    Parameters
    ----------
    reader : GGUFReader
        An opened GGUF reader.
    config : SmolLM2Config
        Model configuration.

    Raises
    ------
    ValueError
        If any expected tensor is missing, has wrong shape, unexpected
        duplicates are found, or quant distribution does not match.
    """
    tensor_names = [t.name for t in reader.tensors]
    name_counts: dict[str, int] = {}
    for name in tensor_names:
        name_counts[name] = name_counts.get(name, 0) + 1
    duplicates = {k: v for k, v in name_counts.items() if v > 1}
    if duplicates:
        raise ValueError(f"Duplicate tensor names found: {duplicates}")

    expected_names = _build_expected_tensor_names(config)
    actual_names = set(tensor_names)
    missing = expected_names - actual_names
    if missing:
        raise ValueError(f"Missing expected tensors: {sorted(missing)}")
    unexpected = actual_names - expected_names
    if unexpected:
        raise ValueError(f"Unexpected tensors found: {sorted(unexpected)}")

    _validate_quant_distribution(reader)


def _validate_tensor_shape(tensor: Any, config: SmolLM2Config, layer_idx: int | None) -> None:
    """Validate a single tensor's GGUF descriptor shape.

    ``tensor.shape`` is in GGUF descriptor order (the dimension ordering
    stored in the GGUF metadata), which is the *reverse* of the NumPy
    logical shape returned by ``gguf.dequantize``.

    Parameters
    ----------
    tensor : ReaderTensor
        The GGUF tensor to validate.
    config : SmolLM2Config
        Model configuration.
    layer_idx : int or None
        Layer index (None for non-layer tensors).

    Raises
    ------
    ValueError
        If the tensor descriptor shape does not match expectations.
    """
    name = tensor.name
    shape = tuple(int(d) for d in tensor.shape)

    if name == "token_embd.weight":
        expected = (config.hidden_size, config.vocab_size)
    elif name == "output_norm.weight":
        expected = (config.hidden_size,)
    elif layer_idx is not None:
        prefix = f"blk.{layer_idx}."
        suffix = name[len(prefix):]
        if suffix in ("attn_norm.weight", "ffn_norm.weight"):
            expected = (config.hidden_size,)
        elif suffix == "attn_q.weight":
            expected = (config.hidden_size, config.n_heads * config.head_dim)
        elif suffix == "attn_k.weight":
            expected = (config.hidden_size, config.n_kv_heads * config.head_dim)
        elif suffix == "attn_v.weight":
            expected = (config.hidden_size, config.n_kv_heads * config.head_dim)
        elif suffix == "attn_output.weight":
            expected = (config.n_heads * config.head_dim, config.hidden_size)
        elif suffix == "ffn_gate.weight":
            expected = (config.hidden_size, config.intermediate_size)
        elif suffix == "ffn_up.weight":
            expected = (config.hidden_size, config.intermediate_size)
        elif suffix == "ffn_down.weight":
            expected = (config.intermediate_size, config.hidden_size)
        else:
            raise ValueError(f"Unknown layer tensor suffix: {suffix}")
    else:
        raise ValueError(f"Unknown non-layer tensor: {name}")

    if shape != expected:
        raise ValueError(
            f"Tensor '{name}': descriptor shape {shape} != expected {expected}"
        )


def _extract_layer_index(name: str) -> int | None:
    """Extract the layer index from a tensor name.

    Parameters
    ----------
    name : str
        Tensor name (e.g., ``blk.5.attn_q.weight``).

    Returns
    -------
    int or None
        Layer index, or None if not a layer tensor.
    """
    if not name.startswith("blk."):
        return None
    parts = name.split(".")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------


def _undo_rope_permutation(
    arr: np.ndarray, n_heads: int, head_dim: int
) -> np.ndarray:
    """Undo the GGUF llama RoPE permutation on output rows.

    The GGUF format permutes Q/K rows so that even/odd pairs are interleaved.
    This function reverses that permutation.

    Parameters
    ----------
    arr : np.ndarray
        Dequantized array with shape ``[out, in]`` where
        ``out = n_heads * head_dim``.
    n_heads : int
        Number of attention heads (or KV heads for K projection).
    head_dim : int
        Dimension per head.

    Returns
    -------
    np.ndarray
        Array with RoPE permutation undone, still shape ``[out, in]``.
    """
    out, in_features = arr.shape
    if out != n_heads * head_dim:
        raise ValueError(
            f"Expected out={n_heads * head_dim}, got {out}"
        )

    # Reshape to [n_heads, head_dim, in_features]
    reshaped = arr.reshape(n_heads, head_dim, in_features)

    # Undo the permutation: even indices go to first half, odd to second half
    # Original permutation: [0, 2, 4, ..., 1, 3, 5, ...]
    # Inverse: first half gets evens, second half gets odds
    half_dim = head_dim // 2
    result = np.empty_like(reshaped)
    result[:, :half_dim, :] = reshaped[:, 0::2, :]  # even indices
    result[:, half_dim:, :] = reshaped[:, 1::2, :]  # odd indices

    return result.reshape(out, in_features)


def _dequantize_tensor(tensor: Any) -> np.ndarray:
    """Dequantize a GGUF tensor to float32.

    Parameters
    ----------
    tensor : ReaderTensor
        The GGUF tensor.

    Returns
    -------
    np.ndarray
        Dequantized float32 array.
    """
    import gguf

    raw = gguf.dequantize(tensor.data, tensor.tensor_type)
    return raw.astype(np.float32)


def _transform_tensor(
    tensor: Any, config: SmolLM2Config, layer_idx: int | None
) -> np.ndarray:
    """Dequantize and transform a single tensor to target orientation.

    ``gguf.dequantize`` returns matrices in ``[out, in]`` order. Linear
    matrices are transposed to the graph's ``[in, out]`` convention after
    applying any role-specific transformation. Vectors and embedding tables
    retain their dequantized orientation.

    Parameters
    ----------
    tensor : ReaderTensor
        The GGUF tensor.
    config : SmolLM2Config
        Model configuration.
    layer_idx : int or None
        Layer index (None for non-layer tensors).

    Returns
    -------
    np.ndarray
        Transformed contiguous float32 array in target orientation.
    """
    name = tensor.name
    dequantized = _dequantize_tensor(tensor)
    descriptor_shape = tuple(int(d) for d in tensor.shape)
    expected_dequant_shape = tuple(reversed(descriptor_shape))

    if dequantized.shape != expected_dequant_shape:
        raise ValueError(
            f"Tensor '{name}': dequantized shape {dequantized.shape} != "
            f"reversed descriptor {expected_dequant_shape}"
        )

    if name == "token_embd.weight":
        result = dequantized
    elif name == "output_norm.weight":
        result = dequantized
    elif layer_idx is not None:
        prefix = f"blk.{layer_idx}."
        suffix = name[len(prefix):]
        if suffix in ("attn_norm.weight", "ffn_norm.weight"):
            result = dequantized
        elif suffix == "attn_q.weight":
            result = _undo_rope_permutation(
                dequantized, config.n_heads, config.head_dim
            ).T
        elif suffix == "attn_k.weight":
            result = _undo_rope_permutation(
                dequantized, config.n_kv_heads, config.head_dim
            ).T
        elif suffix in (
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ):
            result = dequantized.T
        else:
            raise ValueError(f"Unknown layer tensor suffix: {suffix}")
    else:
        raise ValueError(f"Unknown non-layer tensor: {name}")

    return np.ascontiguousarray(result, dtype=np.float32)


def _validate_finite(arr: np.ndarray, name: str) -> None:
    """Validate that an array contains only finite values.

    Parameters
    ----------
    arr : np.ndarray
        Array to validate.
    name : str
        Tensor name for error reporting.

    Raises
    ------
    ValueError
        If the array contains non-finite values.
    """
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(f"Tensor '{name}' contains {n_bad} non-finite values")


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------


def build_inventory(reader: Any) -> list[dict[str, Any]]:
    """Build a tensor inventory from a GGUF reader.

    Parameters
    ----------
    reader : GGUFReader
        An opened GGUF reader.

    Returns
    -------
    list[dict]
        List of dicts with keys: ``name``, ``shape``, ``dtype``, ``n_bytes``,
        ``n_elements``.
    """
    inventory = []
    for tensor in reader.tensors:
        inventory.append(
            {
                "name": tensor.name,
                "shape": [int(d) for d in tensor.shape],
                "dtype": tensor.tensor_type.name,
                "n_bytes": int(tensor.n_bytes),
                "n_elements": int(tensor.n_elements),
            }
        )
    return inventory


def build_manifest(reader: Any) -> dict[str, Any]:
    """Build a summary manifest from a GGUF reader.

    Parameters
    ----------
    reader : GGUFReader
        An opened GGUF reader.

    Returns
    -------
    dict
        Keys: ``architecture``, ``tensor_count``, ``quant_type_counts``,
        ``total_bytes``, ``total_elements``.
    """
    quant_counts: dict[str, int] = {}
    total_bytes = 0
    total_elements = 0
    for tensor in reader.tensors:
        dtype_name = tensor.tensor_type.name
        quant_counts[dtype_name] = quant_counts.get(dtype_name, 0) + 1
        total_bytes += int(tensor.n_bytes)
        total_elements += int(tensor.n_elements)

    arch_field = reader.get_field("general.architecture")
    arch = str(arch_field.parts[arch_field.data[0]]) if arch_field else "unknown"

    return {
        "architecture": arch,
        "tensor_count": len(reader.tensors),
        "quant_type_counts": dict(sorted(quant_counts.items())),
        "total_bytes": total_bytes,
        "total_elements": total_elements,
    }


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_smollm2_weights(path: str | Path, verify_hash: bool = True) -> SmolLM2Weights:
    """Load SmolLM2-135M-Instruct weights from a GGUF file.

    Validates the artifact against the known manifest, dequantizes all
    tensors, applies role-specific transformations, and returns a
    structured weights container.

    Parameters
    ----------
    path : str or Path
        Path to the GGUF file.
    verify_hash : bool
        Whether to verify the SHA-256 hash (default True).

    Returns
    -------
    SmolLM2Weights
        Loaded weights with all arrays as contiguous float32.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If validation fails (size, hash, metadata, tensor names/shapes).
    """
    import gguf

    path = Path(path)
    validate_artifact(path, verify_hash=verify_hash)

    reader = gguf.GGUFReader(str(path), mode="r")
    validate_gguf_metadata(reader)

    config = SmolLM2Config()
    validate_tensor_inventory(reader, config)

    for tensor in reader.tensors:
        layer_idx = _extract_layer_index(tensor.name)
        _validate_tensor_shape(tensor, config, layer_idx)

    tensor_map = {t.name: t for t in reader.tensors}

    token_embedding = _transform_tensor(tensor_map["token_embd.weight"], config, None)
    _validate_finite(token_embedding, "token_embd.weight")

    layers = []
    for i in range(config.n_layers):
        prefix = f"blk.{i}."
        layer_weights: dict[str, np.ndarray] = {}
        for gguf_suffix, key in (
            ("attn_norm.weight", "attn_norm"),
            ("attn_q.weight", "wq"),
            ("attn_k.weight", "wk"),
            ("attn_v.weight", "wv"),
            ("attn_output.weight", "wo"),
            ("ffn_norm.weight", "ffn_norm"),
            ("ffn_gate.weight", "w_gate"),
            ("ffn_up.weight", "w_up"),
            ("ffn_down.weight", "w_down"),
        ):
            gguf_name = prefix + gguf_suffix
            arr = _transform_tensor(tensor_map[gguf_name], config, i)
            _validate_finite(arr, gguf_name)
            layer_weights[key] = arr
        layers.append(layer_weights)

    final_norm = _transform_tensor(tensor_map["output_norm.weight"], config, None)
    _validate_finite(final_norm, "output_norm.weight")

    return SmolLM2Weights(
        config=config,
        token_embedding=token_embedding,
        layers=layers,
        final_norm=final_norm,
        reader=reader,
    )


# ---------------------------------------------------------------------------
# Sanitized reporting
# ---------------------------------------------------------------------------


def sanitize_weights_report(weights: SmolLM2Weights, path: Path) -> dict[str, Any]:
    """Build a sanitized JSON-safe report without absolute paths.

    Parameters
    ----------
    weights : SmolLM2Weights
        Loaded weights.
    path : Path
        Original GGUF file path (used only for filename).

    Returns
    -------
    dict
        Sanitized report with model metadata and weight statistics.
    """
    config = weights.config
    manifest = build_manifest(weights.reader)

    layer_stats = []
    for i, layer in enumerate(weights.layers):
        stats: dict[str, Any] = {"layer": i}
        for key, arr in layer.items():
            stats[key] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        layer_stats.append(stats)

    return {
        "model_repo": EXPECTED_REPO,
        "model_revision": EXPECTED_REVISION,
        "filename": path.name,
        "architecture": EXPECTED_ARCHITECTURE,
        "gguf_version": EXPECTED_GGUF_VERSION,
        "config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "head_dim": config.head_dim,
            "intermediate_size": config.intermediate_size,
            "context_length": config.context_length,
            "rms_eps": config.rms_eps,
            "rope_theta": config.rope_theta,
        },
        "manifest": manifest,
        "token_embedding": {
            "shape": list(weights.token_embedding.shape),
            "dtype": str(weights.token_embedding.dtype),
        },
        "final_norm": {
            "shape": list(weights.final_norm.shape),
            "dtype": str(weights.final_norm.dtype),
        },
        "layer_stats": layer_stats,
    }


def atomic_write_json(data: dict[str, Any], dest: Path) -> None:
    """Write JSON atomically via a temporary file + rename.

    Parameters
    ----------
    data : dict
        Data to serialize.
    dest : Path
        Destination file path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(dest.parent), suffix=".tmp", prefix=".gguf_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True, allow_nan=False)
            f.write("\n")
        os.replace(tmp_path, str(dest))
    except BaseException:
        os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] or None
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Load SmolLM2-135M-Instruct weights from GGUF."
    )
    parser.add_argument(
        "--model", required=True, type=Path, help="Path to the GGUF model file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON weights report.",
    )
    parser.add_argument(
        "--no-verify-hash",
        action="store_true",
        help="Skip SHA-256 hash verification.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv : list[str] or None
        Argument list.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    args = parse_args(argv)
    model_name = args.model.name
    model_path = args.model.resolve()

    try:
        weights = load_smollm2_weights(model_path, verify_hash=not args.no_verify_hash)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(weights.layers)} layers from {model_path.name}")
    print(f"  token_embedding: {weights.token_embedding.shape}")
    print(f"  final_norm: {weights.final_norm.shape}")
    print(f"  layer 0 wq: {weights.layers[0]['wq'].shape}")

    if args.output:
        report = sanitize_weights_report(weights, Path(model_name))
        atomic_write_json(report, args.output)
        print(f"Report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
