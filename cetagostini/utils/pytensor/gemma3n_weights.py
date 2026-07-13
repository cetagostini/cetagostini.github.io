"""Load Gemma3n-E4B-it-lm-4bit weights from an MLX safetensors artifact.

Provides streaming, memory-mapped access to the quantized model without
materializing the full ~3.6 GB file.  Supports:

* Exact text-config dataclass with assertion guards.
* A generated 35-row layer-signature manifest (one row per transformer layer).
* Safetensors header parsing (mmap-safe, no full load).
* Per-module classification of affine-4 triplets vs unquantized tensors.
* BF16 → float32 conversion.
* Low-nibble-first affine-4 dequantization matching ``mx.dequantize``.
* Arbitrary row gather (preserving duplicates and order).
* Range substitution for per-layer embedding IDs ≥ 262 144 → ID 0.
* Contiguous vocabulary-chunk iteration.
* Per-layer weight loading with stable logical-key mapping.
* Selected input-embedding and per-layer-embedding row loading.
* Output-embedding chunk iteration.

All returned arrays are finite, C-contiguous ``np.float32``.

Usage::

    from cetagostini.utils.pytensor.gemma3n_weights import Gemma3nWeightLoader
    loader = Gemma3nWeightLoader.from_snapshot("/path/to/snapshot")
    cfg = loader.config
    layer0 = loader.load_layer(0)
    embed_rows = loader.load_input_embedding_rows([1, 2, 3])
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_LAYERS = 35
GROUP_SIZE = 64
BITS = 4
NIBBLES_PER_U32 = 8  # 32 bits / 4 bits

SAFETENSORS_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"
INDEX_FILENAME = "model.safetensors.index.json"

PREFIX = "model.language_model"

# Per-layer module suffixes (relative to ``layers.{i}.``).
# Each entry is (suffix, is_quantized_triplet, bare_key).
# ``bare_key=True`` means the safetensors key is ``{prefix}.{suffix}``
# (no ``.weight`` suffix).  This applies to ``altup.correct_output_scale``.
LAYER_MODULE_SPECS: list[tuple[str, bool, bool]] = [
    ("self_attn.q_proj", True, False),
    ("self_attn.k_proj", True, False),
    ("self_attn.v_proj", True, False),
    ("self_attn.o_proj", True, False),
    ("self_attn.q_norm", False, False),
    ("self_attn.k_norm", False, False),
    ("mlp.gate_proj", True, False),
    ("mlp.up_proj", True, False),
    ("mlp.down_proj", True, False),
    ("input_layernorm", False, False),
    ("post_attention_layernorm", False, False),
    ("pre_feedforward_layernorm", False, False),
    ("post_feedforward_layernorm", False, False),
    ("altup.correct_output_scale", False, True),
    ("altup.correction_coefs", False, False),
    ("altup.modality_router", True, False),
    ("altup.prediction_coefs", False, False),
    ("altup.router_norm", False, False),
    ("laurel.linear_left", True, False),
    ("laurel.linear_right", True, False),
    ("laurel.post_laurel_norm", False, False),
    ("per_layer_input_gate", True, False),
    ("per_layer_projection", True, False),
    ("post_per_layer_input_norm", False, False),
]

# Global (non-layer) module specs.
GLOBAL_MODULE_SPECS: list[tuple[str, bool]] = [
    ("embed_tokens", True),
    ("embed_tokens_per_layer", True),
    ("norm", False),
    ("per_layer_model_projection", True),
    ("per_layer_projection_norm", False),
    ("altup_projections.0", True),
    ("altup_projections.1", True),
    ("altup_projections.2", True),
    ("altup_unembed_projections.0", True),
    ("altup_unembed_projections.1", True),
    ("altup_unembed_projections.2", True),
]


# ---------------------------------------------------------------------------
# Text configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gemma3nTextConfig:
    """Frozen text configuration for Gemma3n-E4B-it-lm-4bit.

    All fields are extracted from ``config.json → text_config``.
    """

    vocab_size: int = 262_400
    vocab_size_per_layer_input: int = 262_144
    hidden_size: int = 2048
    hidden_size_per_layer_input: int = 256
    intermediate_size: int = 16_384
    num_hidden_layers: int = 35
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    max_position_embeddings: int = 32_768
    sliding_window: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_activation: str = "gelu_pytorch_tanh"
    final_logit_softcapping: float = 30.0
    altup_active_idx: int = 0
    altup_coef_clip: float = 120.0
    altup_correct_scale: bool = True
    altup_lr_multiplier: float = 1.0
    altup_num_inputs: int = 4
    laurel_rank: int = 64
    num_kv_shared_layers: int = 15
    query_pre_attn_scalar: int = 256
    layer_types: tuple[str, ...] = (
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "full_attention",
    )
    activation_sparsity_pattern: tuple[float, ...] = (
        0.95, 0.95, 0.95, 0.95, 0.95,
        0.95, 0.95, 0.95, 0.95, 0.95,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    )

    def __post_init__(self) -> None:
        if self.num_hidden_layers != NUM_LAYERS:
            raise ValueError(f"Expected {NUM_LAYERS} layers, got {self.num_hidden_layers}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per layer")
        if len(self.activation_sparsity_pattern) != self.num_hidden_layers:
            raise ValueError("activation_sparsity_pattern must contain one entry per layer")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.head_dim != self.hidden_size // self.num_attention_heads:
            raise ValueError("head_dim does not match hidden_size / num_attention_heads")
        if self.vocab_size_per_layer_input != 262_144:
            raise ValueError("Expected vocab_size_per_layer_input=262144")


def parse_text_config(config_path: Path) -> Gemma3nTextConfig:
    """Parse ``text_config`` from a Gemma3n ``config.json``.

    Parameters
    ----------
    config_path : Path
        Path to ``config.json``.

    Returns
    -------
    Gemma3nTextConfig
        Validated frozen config.
    """
    with open(config_path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not isinstance(raw.get("text_config"), dict):
        raise ValueError("config.json must contain a text_config object")
    tc = raw["text_config"]
    required_fields = {
        "vocab_size", "vocab_size_per_layer_input", "hidden_size",
        "hidden_size_per_layer_input", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "max_position_embeddings", "sliding_window",
        "rms_norm_eps", "rope_theta", "rope_local_base_freq",
        "attention_bias", "attention_dropout", "hidden_activation",
        "final_logit_softcapping", "altup_active_idx", "altup_coef_clip",
        "altup_correct_scale", "altup_lr_multiplier", "altup_num_inputs",
        "laurel_rank", "num_kv_shared_layers", "query_pre_attn_scalar",
        "layer_types", "activation_sparsity_pattern",
    }
    missing = sorted(required_fields - tc.keys())
    if missing:
        raise ValueError(f"Missing text_config fields: {missing}")
    integer_fields = {
        "vocab_size", "vocab_size_per_layer_input", "hidden_size",
        "hidden_size_per_layer_input", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "max_position_embeddings", "sliding_window",
        "altup_active_idx", "altup_num_inputs", "laurel_rank",
        "num_kv_shared_layers", "query_pre_attn_scalar",
    }
    invalid_integer_fields = sorted(
        name for name in integer_fields if not isinstance(tc[name], int)
    )
    if invalid_integer_fields:
        raise ValueError(
            f"Expected integer text_config fields: {invalid_integer_fields}"
        )
    lt = tuple(tc["layer_types"])
    asp = tuple(tc["activation_sparsity_pattern"])
    return Gemma3nTextConfig(
        vocab_size=tc["vocab_size"],
        vocab_size_per_layer_input=tc["vocab_size_per_layer_input"],
        hidden_size=tc["hidden_size"],
        hidden_size_per_layer_input=tc["hidden_size_per_layer_input"],
        intermediate_size=tc["intermediate_size"],
        num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"],
        max_position_embeddings=tc["max_position_embeddings"],
        sliding_window=tc["sliding_window"],
        rms_norm_eps=tc["rms_norm_eps"],
        rope_theta=tc["rope_theta"],
        rope_local_base_freq=tc["rope_local_base_freq"],
        attention_bias=tc["attention_bias"],
        attention_dropout=tc["attention_dropout"],
        hidden_activation=tc["hidden_activation"],
        final_logit_softcapping=tc["final_logit_softcapping"],
        altup_active_idx=tc["altup_active_idx"],
        altup_coef_clip=tc["altup_coef_clip"],
        altup_correct_scale=tc["altup_correct_scale"],
        altup_lr_multiplier=tc["altup_lr_multiplier"],
        altup_num_inputs=tc["altup_num_inputs"],
        laurel_rank=tc["laurel_rank"],
        num_kv_shared_layers=tc["num_kv_shared_layers"],
        query_pre_attn_scalar=tc["query_pre_attn_scalar"],
        layer_types=lt,
        activation_sparsity_pattern=asp,
    )


# ---------------------------------------------------------------------------
# Layer-signature manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleSignature:
    """Describes one module inside a transformer layer.

    Attributes
    ----------
    suffix : str
        Dot-separated suffix relative to ``layers.{i}.``.
    is_quantized : bool
        True if the module uses affine-4 quantization (weight/scales/biases).
    logical_keys : tuple[str, ...]
        Stable logical key names returned by :func:`load_layer`.
    """

    suffix: str
    is_quantized: bool
    logical_keys: tuple[str, ...]


@dataclass(frozen=True)
class LayerSignature:
    """Complete signature for one transformer layer.

    Attributes
    ----------
    layer_idx : int
        Zero-based layer index.
    layer_type : str
        ``"sliding_attention"`` or ``"full_attention"``.
    modules : tuple[ModuleSignature, ...]
        Ordered module signatures.
    total_keys : int
        Total number of safetensor keys for this layer.
    attention_kind : str
        ``"sliding"`` or ``"full"``.
    sparsity_kind : str
        ``"sparse"`` (activation sparsity > 0) or ``"dense"``.
    rope_base : float
        RoPE base frequency: ``rope_theta`` for full attention,
        ``rope_local_base_freq`` for sliding attention.
    computes_kv : bool
        Whether this layer computes its own K/V projections (always True).
    template_key : str
        Canonical template identifier derived from attention and sparsity
        kinds (e.g. ``"sliding_sparse"``, ``"full_dense"``).
    """

    layer_idx: int
    layer_type: str
    modules: tuple[ModuleSignature, ...]
    total_keys: int
    attention_kind: str
    sparsity_kind: str
    rope_base: float
    computes_kv: bool
    template_key: str


def _logical_keys_for_module(suffix: str, is_quantized: bool) -> tuple[str, ...]:
    """Return stable logical key names for a module.

    Quantized modules expose ``{suffix}`` (dequantized float32 matrix).
    Unquantized modules expose ``{suffix}`` (float32 vector/matrix).
    """
    return (suffix,)


def build_layer_manifest(config: Gemma3nTextConfig) -> list[LayerSignature]:
    """Build the 35-row layer-signature manifest.

    Parameters
    ----------
    config : Gemma3nTextConfig
        Validated text config.

    Returns
    -------
    list[LayerSignature]
        One entry per layer, length == ``config.num_hidden_layers``.
    """
    manifest: list[LayerSignature] = []
    for i in range(config.num_hidden_layers):
        modules: list[ModuleSignature] = []
        total_keys = 0
        for suffix, is_q, _bare in LAYER_MODULE_SPECS:
            lkeys = _logical_keys_for_module(suffix, is_q)
            modules.append(ModuleSignature(suffix=suffix, is_quantized=is_q, logical_keys=lkeys))
            total_keys += 3 if is_q else 1

        attn_kind = "full" if config.layer_types[i] == "full_attention" else "sliding"
        sparse_kind = "sparse" if config.activation_sparsity_pattern[i] > 0.0 else "dense"
        rope_base = config.rope_theta if attn_kind == "full" else config.rope_local_base_freq
        template_key = f"{attn_kind}_{sparse_kind}"

        manifest.append(LayerSignature(
            layer_idx=i,
            layer_type=config.layer_types[i],
            modules=tuple(modules),
            total_keys=total_keys,
            attention_kind=attn_kind,
            sparsity_kind=sparse_kind,
            rope_base=rope_base,
            computes_kv=True,
            template_key=template_key,
        ))
    assert len(manifest) == NUM_LAYERS
    template_keys = {sig.template_key for sig in manifest}
    if len(template_keys) != 4:
        raise ValueError(
            f"Expected exactly 4 template keys, got {len(template_keys)}: "
            f"{sorted(template_keys)}"
        )
    return manifest


# ---------------------------------------------------------------------------
# Safetensors header parsing (mmap-safe)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorInfo:
    """Metadata for a single tensor in a safetensors file.

    Attributes
    ----------
    shape : tuple[int, ...]
        Tensor shape.
    dtype : str
        Safetensors dtype string (e.g. ``"BF16"``, ``"U32"``).
    offset_start : int
        Byte offset of tensor data start (relative to file start).
    offset_end : int
        Byte offset of tensor data end.
    """

    shape: tuple[int, ...]
    dtype: str
    offset_start: int
    offset_end: int


def parse_safetensors_header(path: Path) -> dict[str, TensorInfo]:
    """Parse the safetensors header without loading tensor data.

    Parameters
    ----------
    path : Path
        Path to ``model.safetensors``.

    Returns
    -------
    dict[str, TensorInfo]
        Mapping from tensor key to metadata.
    """
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("Safetensors file is missing its 8-byte header length")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > file_size - 8:
            raise ValueError("Safetensors header extends beyond the file")
        header_bytes = f.read(header_size)
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Safetensors header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be a JSON object")
    # Header size is JSON only; data starts at 8 + header_size.
    data_offset = 8 + header_size
    result: dict[str, TensorInfo] = {}
    dtype_sizes = {"BF16": 2, "U32": 4}
    ranges: list[tuple[int, int, str]] = []
    data_size = file_size - data_offset
    for key, info in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise ValueError(f"Invalid tensor metadata for {key}")
        dtype = info.get("dtype")
        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if dtype not in dtype_sizes:
            raise ValueError(f"Unsupported safetensors dtype for {key}: {dtype}")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or dim < 0 for dim in shape
        ):
            raise ValueError(f"Invalid tensor shape for {key}: {shape}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(
            not isinstance(offset, int) for offset in offsets
        ):
            raise ValueError(f"Invalid data offsets for {key}: {offsets}")
        start, end = offsets
        if not 0 <= start <= end <= data_size:
            raise ValueError(f"Tensor offsets for {key} are outside the data region")
        if end - start != math.prod(shape) * dtype_sizes[dtype]:
            raise ValueError(f"Tensor byte count does not match shape for {key}")
        ranges.append((start, end, key))
        result[key] = TensorInfo(
            shape=tuple(shape),
            dtype=dtype,
            offset_start=data_offset + start,
            offset_end=data_offset + end,
        )
    ordered_ranges = sorted(ranges)
    for (_, previous_end, previous_key), (start, _, key) in zip(
        ordered_ranges, ordered_ranges[1:], strict=False
    ):
        if start < previous_end:
            raise ValueError(f"Tensor data ranges overlap: {previous_key} and {key}")
    return result


# ---------------------------------------------------------------------------
# BF16 → float32
# ---------------------------------------------------------------------------


def bf16_to_float32(raw_bytes: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Convert raw BF16 bytes to a C-contiguous float32 array.

    Parameters
    ----------
    raw_bytes : bytes
        Raw BF16 data (little-endian, 2 bytes per element).
    shape : tuple[int, ...]
        Target shape.

    Returns
    -------
    np.ndarray
        C-contiguous float32 array.
    """
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    # BF16 → float32: shift left 16 bits into the upper half of float32.
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return np.ascontiguousarray(f32.reshape(shape), dtype=np.float32)


# ---------------------------------------------------------------------------
# Affine-4 dequantization (low-nibble-first, matches mx.dequantize)
# ---------------------------------------------------------------------------


def dequantize_affine4(
    weight_u32: np.ndarray,
    scales_bf16: np.ndarray,
    biases_bf16: np.ndarray,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
) -> np.ndarray:
    """Dequantize an affine-4 quantized matrix (low-nibble-first).

    Matches ``mx.dequantize(w, scales, biases, group_size=64, bits=4)``.

    Parameters
    ----------
    weight_u32 : np.ndarray
        Quantized weights, shape ``[out, in // 8]``, dtype uint32.
    scales_bf16 : np.ndarray
        Scales, shape ``[out, in // group_size]``, raw BF16 as uint16.
    biases_bf16 : np.ndarray
        Biases, shape ``[out, in // group_size]``, raw BF16 as uint16.
    group_size : int
        Quantization group size (default 64).
    bits : int
        Bits per element (default 4).

    Returns
    -------
    np.ndarray
        Dequantized float32 matrix, shape ``[out, in]``, C-contiguous.
    """
    if weight_u32.ndim != 2 or weight_u32.dtype != np.uint32:
        raise ValueError("weight_u32 must be a rank-2 uint32 array")
    if scales_bf16.ndim != 2 or scales_bf16.dtype != np.uint16:
        raise ValueError("scales_bf16 must be a rank-2 uint16 array")
    if biases_bf16.ndim != 2 or biases_bf16.dtype != np.uint16:
        raise ValueError("biases_bf16 must be a rank-2 uint16 array")
    if bits != 4:
        raise ValueError(f"Only affine 4-bit weights are supported, got bits={bits}")
    if group_size <= 0 or group_size % NIBBLES_PER_U32 != 0:
        raise ValueError("group_size must be a positive multiple of 8")
    out_features = weight_u32.shape[0]
    in_packed = weight_u32.shape[1]
    in_features = in_packed * NIBBLES_PER_U32
    if group_size > in_features or in_features % group_size != 0:
        raise ValueError("group_size must divide the dequantized input dimension")
    n_groups = in_features // group_size
    expected_affine_shape = (out_features, n_groups)
    if scales_bf16.shape != expected_affine_shape:
        raise ValueError(
            f"scales shape {scales_bf16.shape} does not match "
            f"{expected_affine_shape}"
        )
    if biases_bf16.shape != expected_affine_shape:
        raise ValueError(
            f"biases shape {biases_bf16.shape} does not match "
            f"{expected_affine_shape}"
        )

    # Extract all nibbles: [out, in_features]
    # Low nibble first: nibble_i = (word >> (4*i)) & 0xF
    nibbles = np.empty((out_features, in_features), dtype=np.uint8)
    for w_idx in range(in_packed):
        word = weight_u32[:, w_idx]
        base = w_idx * NIBBLES_PER_U32
        for n in range(NIBBLES_PER_U32):
            nibbles[:, base + n] = ((word >> (4 * n)) & 0xF).astype(np.uint8)

    # Convert scales and biases from BF16 to float32
    scales_f32 = bf16_to_float32(scales_bf16.tobytes(), scales_bf16.shape)
    biases_f32 = bf16_to_float32(biases_bf16.tobytes(), biases_bf16.shape)

    # Dequantize per group
    nibbles_f32 = nibbles.astype(np.float32)
    result = np.empty((out_features, in_features), dtype=np.float32)
    for g in range(n_groups):
        col_start = g * group_size
        col_end = col_start + group_size
        result[:, col_start:col_end] = (
            nibbles_f32[:, col_start:col_end] * scales_f32[:, g : g + 1]
            + biases_f32[:, g : g + 1]
        )

    return np.ascontiguousarray(result)


def _validate_finite(arr: np.ndarray, key: str) -> np.ndarray:
    """Return *arr* after rejecting non-finite materialized weights."""
    if not np.all(np.isfinite(arr)):
        count = int(np.sum(~np.isfinite(arr)))
        raise ValueError(f"Tensor {key!r} contains {count} non-finite values")
    return arr


# ---------------------------------------------------------------------------
# Row gather (preserves duplicates and order)
# ---------------------------------------------------------------------------


def gather_rows(arr: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Gather rows from a 2-D array, preserving duplicates and order.

    Parameters
    ----------
    arr : np.ndarray
        2-D source array, shape ``[N, D]``.
    indices : Sequence[int]
        Row indices to gather (may contain duplicates).

    Returns
    -------
    np.ndarray
        Gathered array, shape ``[len(indices), D]``, C-contiguous float32.
    """
    idx = np.asarray(indices, dtype=np.intp)
    result = arr[idx]
    return np.ascontiguousarray(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# Range substitution for per-layer embedding IDs
# ---------------------------------------------------------------------------

PER_LAYER_VOCAB_BOUNDARY = 262_144


def substitute_per_layer_ids(ids: np.ndarray) -> np.ndarray:
    """Replace IDs ≥ 262 144 with ID 0 for per-layer embedding lookup.

    The per-layer embedding table has ``vocab_size_per_layer_input = 262 144``
    rows.  Token IDs at or above this boundary (image/audio special tokens)
    are mapped to row 0.

    Parameters
    ----------
    ids : np.ndarray
        1-D integer array of token IDs.

    Returns
    -------
    np.ndarray
        Clamped copy with IDs ≥ 262 144 replaced by 0.
    """
    result = ids.copy()
    result[result >= PER_LAYER_VOCAB_BOUNDARY] = 0
    return result


# ---------------------------------------------------------------------------
# Vocabulary chunk iteration
# ---------------------------------------------------------------------------


def vocab_chunks(
    vocab_size: int, chunk_size: int = 4096
) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` ranges covering ``[0, vocab_size)``.

    Parameters
    ----------
    vocab_size : int
        Total vocabulary size.
    chunk_size : int
        Maximum rows per chunk.

    Yields
    ------
    tuple[int, int]
        ``(start, end)`` half-open ranges.
    """
    if not isinstance(vocab_size, int) or vocab_size < 0:
        raise ValueError("vocab_size must be a non-negative integer")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    for start in range(0, vocab_size, chunk_size):
        yield start, min(start + chunk_size, vocab_size)


# ---------------------------------------------------------------------------
# Weight loader
# ---------------------------------------------------------------------------


@dataclass
class Gemma3nWeightLoader:
    """Streaming weight loader for Gemma3n-E4B-it-lm-4bit.

    Opens the safetensors file via mmap and provides per-layer, per-module,
    and embedding accessors without materializing the full model.

    Attributes
    ----------
    snapshot_dir : Path
        Path to the model snapshot directory.
    config : Gemma3nTextConfig
        Parsed text configuration.
    tensor_info : dict[str, TensorInfo]
        Safetensors header metadata.
    manifest : list[LayerSignature]
        35-row layer-signature manifest.
    _mmap : np.memmap | None
        Memory-mapped view of the safetensors data region.
    """

    snapshot_dir: Path
    config: Gemma3nTextConfig
    tensor_info: dict[str, TensorInfo]
    manifest: list[LayerSignature]
    _mmap: np.memmap | None = field(default=None, repr=False)

    @classmethod
    def from_snapshot(cls, snapshot_dir: str | Path) -> Gemma3nWeightLoader:
        """Create a loader from a snapshot directory.

        Parameters
        ----------
        snapshot_dir : str or Path
            Path containing ``config.json`` and ``model.safetensors``.

        Returns
        -------
        Gemma3nWeightLoader
            Initialized loader with mmap open.
        """
        snapshot_dir = Path(snapshot_dir)
        config = parse_text_config(snapshot_dir / CONFIG_FILENAME)
        st_path = snapshot_dir / SAFETENSORS_FILENAME
        tensor_info = parse_safetensors_header(st_path)
        manifest = build_layer_manifest(config)

        # Open mmap over the entire file (lazy — pages loaded on access).
        file_size = st_path.stat().st_size
        mmap = np.memmap(str(st_path), dtype=np.uint8, mode="r", shape=(file_size,))

        return cls(
            snapshot_dir=snapshot_dir,
            config=config,
            tensor_info=tensor_info,
            manifest=manifest,
            _mmap=mmap,
        )

    def close(self) -> None:
        """Release the memory map."""
        if self._mmap is not None:
            del self._mmap
            self._mmap = None

    def __enter__(self) -> Gemma3nWeightLoader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # -- Raw tensor access --------------------------------------------------

    def _read_raw(self, key: str) -> bytes:
        """Read raw bytes for a tensor key from the mmap."""
        info = self.tensor_info[key]
        return bytes(self._mmap[info.offset_start : info.offset_end])

    def _load_bf16_tensor(self, key: str) -> np.ndarray:
        """Load a BF16 tensor as float32."""
        info = self.tensor_info[key]
        if info.dtype != "BF16":
            raise TypeError(f"Expected BF16 for {key}, got {info.dtype}")
        raw = self._read_raw(key)
        return _validate_finite(bf16_to_float32(raw, info.shape), key)

    def _load_u32_tensor(self, key: str) -> np.ndarray:
        """Load a U32 tensor as uint32."""
        info = self.tensor_info[key]
        if info.dtype != "U32":
            raise TypeError(f"Expected U32 for {key}, got {info.dtype}")
        raw = self._read_raw(key)
        arr = np.frombuffer(raw, dtype=np.uint32).reshape(info.shape)
        return np.ascontiguousarray(arr)

    # -- Module classification ----------------------------------------------

    def _is_quantized_key(self, full_key: str) -> bool:
        """Check if a safetensors key belongs to a quantized triplet."""
        return full_key.endswith(".weight") and (
            full_key.replace(".weight", ".scales") in self.tensor_info
        )

    # -- Dequantization -----------------------------------------------------

    def _dequantize_module(self, prefix: str) -> np.ndarray:
        """Load and dequantize a single affine-4 module.

        Parameters
        ----------
        prefix : str
            Full safetensors key prefix (e.g.
            ``"model.language_model.layers.0.mlp.gate_proj"``).

        Returns
        -------
        np.ndarray
            Dequantized float32 matrix, shape ``[out, in]``.
        """
        w_key = f"{prefix}.weight"
        s_key = f"{prefix}.scales"
        b_key = f"{prefix}.biases"
        weight_u32 = self._load_u32_tensor(w_key)
        # Load scales/biases as raw uint16 (BF16 storage)
        s_info = self.tensor_info[s_key]
        b_info = self.tensor_info[b_key]
        scales_raw = np.frombuffer(self._read_raw(s_key), dtype=np.uint16).reshape(s_info.shape)
        biases_raw = np.frombuffer(self._read_raw(b_key), dtype=np.uint16).reshape(b_info.shape)
        return _validate_finite(
            dequantize_affine4(weight_u32, scales_raw, biases_raw),
            prefix,
        )

    # -- Row-sliced mmap reading (embedding tables) -------------------------

    def _read_rows_from_mmap(
        self, key: str, row_indices: np.ndarray
    ) -> np.ndarray:
        """Read specific rows from a 2-D tensor via mmap.

        Only the requested rows are paged in — the full tensor is never
        materialized.

        Parameters
        ----------
        key : str
            Safetensors tensor key (must be 2-D).
        row_indices : np.ndarray
            1-D array of **unique, non-negative** row indices.

        Returns
        -------
        np.ndarray
            ``[len(row_indices), n_cols]`` — uint32 for U32 tensors,
            uint16 for BF16 tensors.
        """
        if self._mmap is None:
            raise ValueError("Weight loader is closed")
        info = self.tensor_info[key]
        if len(info.shape) != 2:
            raise ValueError(f"Expected 2-D tensor for {key}")
        if np.any(row_indices < 0) or np.any(row_indices >= info.shape[0]):
            raise IndexError(f"Row index out of bounds for {key}")
        n_cols = info.shape[1]

        if info.dtype == "U32":
            elem_size = 4
            np_dtype: type[np.generic] = np.uint32
        elif info.dtype == "BF16":
            elem_size = 2
            np_dtype = np.uint16
        else:
            raise ValueError(f"Unsupported dtype for row read: {info.dtype}")

        row_bytes = n_cols * elem_size
        n_rows = len(row_indices)
        if n_rows == 0:
            return np.empty((0, n_cols), dtype=np_dtype)

        result = np.empty((n_rows, n_cols), dtype=np_dtype)
        for i in range(n_rows):
            byte_start = info.offset_start + int(row_indices[i]) * row_bytes
            byte_end = byte_start + row_bytes
            raw = bytes(self._mmap[byte_start:byte_end])
            result[i] = np.frombuffer(raw, dtype=np_dtype)
        return result

    def _read_contiguous_rows(
        self, key: str, start: int, end: int
    ) -> np.ndarray:
        """Read contiguous rows ``[start, end)`` from a 2-D tensor via mmap.

        Parameters
        ----------
        key : str
            Safetensors tensor key (must be 2-D).
        start : int
            First row index (inclusive).
        end : int
            Last row index (exclusive).

        Returns
        -------
        np.ndarray
            ``[end - start, n_cols]`` — uint32 for U32, uint16 for BF16.
            Always a writable copy (not a read-only buffer view).
        """
        if self._mmap is None:
            raise ValueError("Weight loader is closed")
        info = self.tensor_info[key]
        if len(info.shape) != 2:
            raise ValueError(f"Expected 2-D tensor for {key}")
        if not 0 <= start <= end <= info.shape[0]:
            raise IndexError(f"Row range [{start}, {end}) is out of bounds for {key}")
        n_cols = info.shape[1]

        if info.dtype == "U32":
            elem_size = 4
            np_dtype: type[np.generic] = np.uint32
        elif info.dtype == "BF16":
            elem_size = 2
            np_dtype = np.uint16
        else:
            raise ValueError(f"Unsupported dtype for contiguous read: {info.dtype}")

        row_bytes = n_cols * elem_size
        byte_start = info.offset_start + start * row_bytes
        byte_end = info.offset_start + end * row_bytes
        raw = bytes(self._mmap[byte_start:byte_end])
        return np.frombuffer(raw, dtype=np_dtype).reshape(end - start, n_cols).copy()

    def _dequantize_gather(
        self, prefix: str, ids: np.ndarray, vocab_size: int
    ) -> np.ndarray:
        """Row-sliced dequantization with gather (preserves duplicates/order).

        Reads only the unique requested rows from the mmap, dequantizes
        them, then gathers to restore the original order and duplicates.

        Parameters
        ----------
        prefix : str
            Safetensors key prefix (e.g. ``"model.language_model.embed_tokens"``).
        ids : np.ndarray
            1-D int64 array of row indices (already validated and substituted).
        vocab_size : int
            Number of rows in the embedding table (for bounds context).

        Returns
        -------
        np.ndarray
            Float32 array, shape ``[len(ids), dequantized_in_features]``,
            C-contiguous.
        """
        if len(ids) == 0:
            # Determine output width from the weight tensor metadata.
            w_info = self.tensor_info[f"{prefix}.weight"]
            out_width = w_info.shape[1] * NIBBLES_PER_U32
            return np.empty((0, out_width), dtype=np.float32)

        unique_ids, inverse = np.unique(ids, return_inverse=True)

        w_key = f"{prefix}.weight"
        s_key = f"{prefix}.scales"
        b_key = f"{prefix}.biases"

        weight_rows = self._read_rows_from_mmap(w_key, unique_ids)
        scales_rows = self._read_rows_from_mmap(s_key, unique_ids)
        biases_rows = self._read_rows_from_mmap(b_key, unique_ids)

        dequant = dequantize_affine4(weight_rows, scales_rows, biases_rows)

        return _validate_finite(
            np.ascontiguousarray(dequant[inverse], dtype=np.float32),
            prefix,
        )

    def _dequantize_contiguous(
        self, prefix: str, start: int, end: int
    ) -> np.ndarray:
        """Dequantize a contiguous block of rows ``[start, end)``.

        Reads only the requested row range from the mmap.

        Parameters
        ----------
        prefix : str
            Safetensors key prefix.
        start : int
            First row (inclusive).
        end : int
            Last row (exclusive).

        Returns
        -------
        np.ndarray
            Float32 array, shape ``[end - start, dequantized_in_features]``,
            C-contiguous.
        """
        w_key = f"{prefix}.weight"
        s_key = f"{prefix}.scales"
        b_key = f"{prefix}.biases"

        weight_chunk = self._read_contiguous_rows(w_key, start, end)
        scales_chunk = self._read_contiguous_rows(s_key, start, end)
        biases_chunk = self._read_contiguous_rows(b_key, start, end)

        return _validate_finite(
            dequantize_affine4(weight_chunk, scales_chunk, biases_chunk),
            prefix,
        )

    # -- Per-layer loading --------------------------------------------------

    def load_layer(self, layer_idx: int) -> dict[str, np.ndarray]:
        """Load all weights for a single transformer layer.

        Returns a dictionary keyed by stable logical names.  Quantized
        modules are dequantized to float32 matrices; unquantized modules
        are converted from BF16 to float32.

        **Array orientation**: all weight matrices are ``[out_features,
        in_features]`` (row-major).  For graph consumption, transpose as
        needed (e.g. ``W.T @ x`` for ``y = Wx``).

        Parameters
        ----------
        layer_idx : int
            Zero-based layer index (0–34).

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from logical key to float32 array.
        """
        if not 0 <= layer_idx < NUM_LAYERS:
            raise IndexError(f"Layer index must be in [0, {NUM_LAYERS}), got {layer_idx}")
        sig = self.manifest[layer_idx]
        result: dict[str, np.ndarray] = {}
        layer_prefix = f"{PREFIX}.layers.{layer_idx}"

        for mod in sig.modules:
            full_prefix = f"{layer_prefix}.{mod.suffix}"
            for lkey in mod.logical_keys:
                if mod.is_quantized:
                    result[lkey] = self._dequantize_module(full_prefix)
                else:
                    # Unquantized: single tensor (may or may not have .weight suffix)
                    # Check if bare key exists first (e.g. altup.correct_output_scale)
                    bare_key = full_prefix
                    weight_key = f"{full_prefix}.weight"
                    if bare_key in self.tensor_info and weight_key not in self.tensor_info:
                        result[lkey] = self._load_bf16_tensor(bare_key)
                    else:
                        result[lkey] = self._load_bf16_tensor(weight_key)

        return result

    # -- Embedding access ---------------------------------------------------

    def load_input_embedding_rows(
        self, ids: Sequence[int]
    ) -> np.ndarray:
        """Load selected rows from the main input embedding table.

        Reads only the requested rows from the mmap — the full
        ``[262400, 2048]`` table is never materialized.

        Parameters
        ----------
        ids : Sequence[int]
            Token IDs to look up.  Must satisfy ``0 <= id < vocab_size``.

        Returns
        -------
        np.ndarray
            Float32 array, shape ``[len(ids), hidden_size]``, C-contiguous.

        Raises
        ------
        ValueError
            If any ID is negative or >= ``vocab_size``.
        """
        ids_arr = np.asarray(ids, dtype=np.int64)
        if len(ids_arr) > 0:
            if np.any(ids_arr < 0):
                raise ValueError(
                    f"Negative token IDs not allowed, got min={int(ids_arr.min())}"
                )
            if np.any(ids_arr >= self.config.vocab_size):
                raise ValueError(
                    f"Token ID out of bounds: max allowed "
                    f"{self.config.vocab_size - 1}, got {int(ids_arr.max())}"
                )
        prefix = f"{PREFIX}.embed_tokens"
        return self._dequantize_gather(prefix, ids_arr, self.config.vocab_size)

    def load_per_layer_embedding_rows(
        self, ids: Sequence[int]
    ) -> np.ndarray:
        """Load selected rows from the per-layer embedding table.

        IDs ≥ 262 144 are substituted to 0 before lookup (row zero is
        retained as the fallback for image/audio special tokens).  Reads
        only the requested rows — the full ``[262144, 8960]`` table is
        never materialized.

        Parameters
        ----------
        ids : Sequence[int]
            Token IDs.  Negative IDs raise ``ValueError``; IDs ≥ 262 144
            are silently mapped to 0.

        Returns
        -------
        np.ndarray
            Float32 array, shape ``[len(ids), hidden_size_per_layer * 35]``
            which can be reshaped to ``[len(ids), 35, 256]``.  C-contiguous.

        Raises
        ------
        ValueError
            If any ID is negative.
        """
        ids_arr = np.asarray(ids, dtype=np.int64)
        if len(ids_arr) > 0 and np.any(ids_arr < 0):
            raise ValueError(
                f"Negative token IDs not allowed, got min={int(ids_arr.min())}"
            )
        # Substitute IDs >= boundary → 0 (row zero retained).
        ids_arr = substitute_per_layer_ids(ids_arr)

        prefix = f"{PREFIX}.embed_tokens_per_layer"
        return self._dequantize_gather(
            prefix, ids_arr, self.config.vocab_size_per_layer_input
        )

    def iter_output_embedding_chunks(
        self, chunk_size: int = 4096
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Iterate over the input embedding table in contiguous chunks.

        Each chunk is read independently from the mmap and dequantized
        on the fly — the full ``[262400, 2048]`` table is never held in
        memory.  Previous chunks become eligible for GC after each yield.

        Useful for computing the output projection (tied embeddings)
        without materializing ~2.15 GB of float32.

        Parameters
        ----------
        chunk_size : int
            Maximum rows per chunk.

        Yields
        ------
        tuple[int, int, np.ndarray]
            ``(start, end, chunk)`` where ``chunk`` has shape
            ``[end - start, hidden_size]``, C-contiguous float32.
        """
        prefix = f"{PREFIX}.embed_tokens"
        for start, end in vocab_chunks(self.config.vocab_size, chunk_size):
            chunk = self._dequantize_contiguous(prefix, start, end)
            yield start, end, chunk

    # -- Global weight access -----------------------------------------------

    def load_global(self, module_suffix: str) -> np.ndarray:
        """Load a global (non-layer) module.

        Parameters
        ----------
        module_suffix : str
            Module suffix relative to ``model.language_model.``
            (e.g. ``"norm"``, ``"per_layer_model_projection"``).

        Returns
        -------
        np.ndarray
            Float32 array.
        """
        full_prefix = f"{PREFIX}.{module_suffix}"
        if self._is_quantized_key(f"{full_prefix}.weight"):
            return self._dequantize_module(full_prefix)
        return self._load_bf16_tensor(f"{full_prefix}.weight")

    # -- Key completeness check ---------------------------------------------

    def validate_keys(self) -> None:
        """Validate that all expected keys are present in the safetensors.

        Raises
        ------
        ValueError
            If any expected key is missing.
        """
        expected: set[str] = set()
        # Layer keys
        for i in range(NUM_LAYERS):
            layer_prefix = f"{PREFIX}.layers.{i}"
            for suffix, is_q, bare_key in LAYER_MODULE_SPECS:
                base = f"{layer_prefix}.{suffix}"
                if is_q:
                    expected.add(f"{base}.weight")
                    expected.add(f"{base}.scales")
                    expected.add(f"{base}.biases")
                elif bare_key:
                    expected.add(base)
                else:
                    expected.add(f"{base}.weight")
        # Global keys
        for suffix, is_q in GLOBAL_MODULE_SPECS:
            base = f"{PREFIX}.{suffix}"
            if is_q:
                expected.add(f"{base}.weight")
                expected.add(f"{base}.scales")
                expected.add(f"{base}.biases")
            else:
                expected.add(f"{base}.weight")

        actual = set(self.tensor_info.keys())
        missing = expected - actual
        if missing:
            raise ValueError(f"Missing {len(missing)} keys: {sorted(missing)[:5]}...")
        extra = actual - expected
        if extra:
            raise ValueError(f"Unexpected {len(extra)} keys: {sorted(extra)[:5]}...")
