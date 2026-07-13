#!/usr/bin/env python3
"""Run Gemma 3n E4B-it through PyTensor against a standalone MLX-LM oracle.

Orchestration, reference, and reporting unit.  Loads a pinned local HF
snapshot, tokenizes a fixed prompt via the snapshot's chat template, verifies
precomputed all-position MLX-LM reference logits, then runs a sequential-layer
PyTensor forward through C, Numba, or MLX and compares every prompt position.

No KV cache or autoregressive generation is performed — this is a single
prefill-style forward pass producing logits for all input positions.

Sibling modules ``gemma3n_weights`` and ``gemma3n_pytensor`` are imported
lazily inside execution paths via narrow adapter functions so that this
orchestration module can be developed and tested independently.

Usage::

    python -m cetagostini.utils.pytensor.run_gemma3n_pytensor probe --snapshot /path/to/snapshot
    python -m cetagostini.utils.pytensor.run_gemma3n_pytensor run \
        --snapshot /path/to/snapshot --run-id RUN_ID \
        --reference-report oracle.json --reference-logits oracle.npy \
        --backend mlx --output result.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from cetagostini.utils.pytensor.evidence import (
    GEMMA3N_ORACLE_SCHEMA_VERSION,
    ArtifactVerificationError,
    atomic_write_npy,
    build_npy_manifest,
    verify_npy_artifact,
)
from cetagostini.utils.pytensor.provenance import (
    GEMMA3N_ENVIRONMENT_YML,
    GEMMA3N_IMPLEMENTATION_SOURCE_FILES,
    GEMMA3N_PROVENANCE_MODULES,
    GEMMA3N_PROVENANCE_PACKAGES,
    build_implementation_manifest,
    build_provenance_report,
    find_repo_root,
)


# ---------------------------------------------------------------------------
# Constants — pinned snapshot identity (matches run_gemma_mlx.py)
# ---------------------------------------------------------------------------

EXPECTED_REPO = "mlx-community/gemma-3n-E4B-it-lm-4bit"
EXPECTED_REVISION = "00b5ecdc79ba872a9b4cd32f4327e263bab5936c"
EXPECTED_MODEL_TYPE = "gemma3n"
EXPECTED_ARCHITECTURE = "Gemma3nForConditionalGeneration"
EXPECTED_BITS = 4
EXPECTED_GROUP_SIZE = 64

REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]

EXPECTED_MANIFEST: dict[str, dict[str, Any]] = {
    "config.json": {
        "size": 124435,
        "sha256": "9525851826c5f5da3d190edcc3a11ba0b9a588b4307298aceb0c62cab22719ca",
    },
    "model.safetensors": {
        "size": 3863598176,
        "sha256": "94401d496aa8a68c0d853adcbb0acea9635e71e390afeb24678acd0dbf530007",
    },
    "tokenizer.json": {
        "size": 33442553,
        "sha256": "b6c35ee648c07754b44cd9e371c75d4caa05c4504910b7ad29b1847ee9d8ba5d",
    },
    "tokenizer_config.json": {
        "size": 1202305,
        "sha256": "0579706d90acb1f4dfa057324b05e3fec18e77d285ced1dc7d617c61c77ef863",
    },
    "chat_template.jinja": {
        "size": 1626,
        "sha256": "ac03dcb3b09726f1e50ac55ae58fc8d5930eadc3b3a12cb04286dc1d82ac8001",
    },
}

DEFAULT_PROMPT = "Explain in two sentences what a symbolic tensor graph is."
EXPECTED_PROMPT_TOKEN_IDS = (
    2, 105, 2364, 107, 155122, 528, 1156, 23974, 1144, 496,
    42988, 18441, 3753, 563, 236761, 106, 107, 105, 4368, 107,
)
EXPECTED_PROMPT_TOKEN_HASH = (
    "bec5926dff4bdc1ae70cb754a2078ad616f830aa1a31fcd6fdc5b72512299545"
)
EXPECTED_VOCAB_SIZE = 262400

# C, Numba, and MLX backends are accepted for ``run``.
VALID_BACKENDS = ("c", "numba", "mlx")

# Publication thresholds (hard gates for the report).
PUB_COSINE_MIN = 0.99
PUB_PEARSON_MIN = 0.99
PUB_ALL_TOP1_MATCH = True
PUB_TOP10_OVERLAP_MEAN_MIN = 8.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments with probe/run subcommands.

    Parameters
    ----------
    argv : list[str] or None
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``command`` set to ``'probe'`` or ``'run'``.
    """
    parser = argparse.ArgumentParser(
        description="Run Gemma 3n E4B-it via PyTensor with MLX-LM reference.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- probe --
    probe_p = sub.add_parser("probe", help="Check environment and snapshot.")
    probe_p.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Optional path to local HF snapshot directory.",
    )

    # -- run --
    run_p = sub.add_parser(
        "run",
        help="Execute backend forward pass against pre-computed oracle.",
    )
    run_p.add_argument(
        "--snapshot",
        required=True,
        type=Path,
        help="Path to the local HF snapshot directory.",
    )
    run_p.add_argument(
        "--run-id",
        required=True,
        type=str,
        help="Unique run identifier for this evidence run.",
    )
    run_p.add_argument(
        "--reference-report",
        required=True,
        type=Path,
        help="Path to the standalone oracle JSON report.",
    )
    run_p.add_argument(
        "--reference-logits",
        required=True,
        type=Path,
        help="Path to the standalone oracle logits .npy artifact.",
    )
    run_p.add_argument(
        "--logits-output",
        required=True,
        type=Path,
        help="Path for the backend logits .npy validation artifact.",
    )
    run_p.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"User prompt text (default: {DEFAULT_PROMPT!r}).",
    )
    run_p.add_argument(
        "--backend",
        type=str,
        choices=VALID_BACKENDS,
        default="c",
        help="PyTensor backend: 'c' (default), 'numba', or 'mlx'.",
    )
    run_p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON result report.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Validation helpers (same patterns as run_gemma_mlx.py)
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file using chunked reads.

    Parameters
    ----------
    path : Path
        File to hash.
    chunk_size : int
        Read chunk size in bytes (default 1 MiB).

    Returns
    -------
    str
        Hex digest string.
    """
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


def detect_revision(snapshot_dir: Path) -> str:
    """Detect the revision hash from the snapshot directory name.

    Parameters
    ----------
    snapshot_dir : Path
        Root directory of the local snapshot.

    Returns
    -------
    str
        The directory basename (which equals ``EXPECTED_REVISION``).

    Raises
    ------
    ValueError
        If the directory basename does not exactly equal
        ``EXPECTED_REVISION``.
    """
    name = snapshot_dir.resolve().name
    if name != EXPECTED_REVISION:
        raise ValueError(
            f"Snapshot directory basename '{name}' does not match "
            f"expected revision '{EXPECTED_REVISION}'"
        )
    return name


def validate_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    """Validate a local HF snapshot directory.

    Parameters
    ----------
    snapshot_dir : Path
        Root directory of the local snapshot.

    Returns
    -------
    dict
        Parsed ``config.json`` contents.

    Raises
    ------
    FileNotFoundError
        If a required file is missing.
    ValueError
        If revision, file integrity, or config fields do not match.
    """
    detect_revision(snapshot_dir)

    for name in REQUIRED_FILES:
        fpath = snapshot_dir / name
        if not fpath.exists():
            raise FileNotFoundError(f"Required file missing: {fpath}")
        if fpath.stat().st_size == 0:
            raise ValueError(f"Required file is empty (0 bytes): {fpath}")

    for name in REQUIRED_FILES:
        fpath = snapshot_dir / name
        expected = EXPECTED_MANIFEST[name]
        actual_size = fpath.stat().st_size
        if actual_size != expected["size"]:
            raise ValueError(
                f"File '{name}' size mismatch: "
                f"expected {expected['size']}, got {actual_size}"
            )
        actual_hash = _sha256_file(fpath)
        if actual_hash != expected["sha256"]:
            raise ValueError(
                f"File '{name}' SHA-256 mismatch: "
                f"expected {expected['sha256']}, got {actual_hash}"
            )

    config_path = snapshot_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    model_type = config.get("model_type", "")
    if model_type != EXPECTED_MODEL_TYPE:
        raise ValueError(
            f"Expected model_type '{EXPECTED_MODEL_TYPE}', got '{model_type}'"
        )

    architectures = config.get("architectures", [])
    if EXPECTED_ARCHITECTURE not in architectures:
        raise ValueError(
            f"Expected architecture '{EXPECTED_ARCHITECTURE}' "
            f"not found in {architectures}"
        )

    quant = config.get("quantization", config.get("quantization_config", {}))
    bits = quant.get("bits")
    group_size = quant.get("group_size")

    if bits != EXPECTED_BITS:
        raise ValueError(f"Expected quantization bits={EXPECTED_BITS}, got {bits}")
    if group_size != EXPECTED_GROUP_SIZE:
        raise ValueError(
            f"Expected quantization group_size={EXPECTED_GROUP_SIZE}, "
            f"got {group_size}"
        )

    return config


def build_file_manifest(snapshot_dir: Path) -> list[dict[str, Any]]:
    """Compute SHA-256 and size for each required snapshot file.

    Parameters
    ----------
    snapshot_dir : Path
        Root directory of the local snapshot.

    Returns
    -------
    list[dict]
        One entry per required file with keys ``name``, ``size_bytes``,
        and ``sha256``.
    """
    manifest: list[dict[str, Any]] = []
    for name in REQUIRED_FILES:
        fpath = snapshot_dir / name
        size_bytes = fpath.stat().st_size
        sha256 = _sha256_file(fpath)
        manifest.append({
            "name": name,
            "size_bytes": size_bytes,
            "sha256": sha256,
        })
    return manifest


# ---------------------------------------------------------------------------
# Package / hardware status
# ---------------------------------------------------------------------------


def collect_versions() -> dict[str, str]:
    """Collect package versions for the result report.

    Returns
    -------
    dict[str, str]
        Mapping of package name to version string.
    """
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for mod_name, distribution_name in (
        ("pytensor", "pytensor"),
        ("numba", "numba"),
        ("mlx", "mlx"),
        ("mlx_lm", "mlx-lm"),
        ("transformers", "transformers"),
    ):
        try:
            versions[mod_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            versions[mod_name] = "unavailable"

    try:
        import pytensor_ml
        versions["pytensor_ml"] = pytensor_ml.__version__
    except ImportError:
        versions["pytensor_ml"] = "unavailable"

    return versions


def check_optional_statuses() -> dict[str, Any]:
    """Check JAX and PyTensor-MLX availability.

    Returns
    -------
    dict
        Keys: ``jax_installed``, ``jax_version``,
        ``pytensor_ml_installed``, ``pytensor_ml_version``.
    """
    result: dict[str, Any] = {}

    try:
        import jax
        result["jax_installed"] = True
        result["jax_version"] = jax.__version__
    except ImportError:
        result["jax_installed"] = False
        result["jax_version"] = None

    try:
        import pytensor_ml
        result["pytensor_ml_installed"] = True
        result["pytensor_ml_version"] = pytensor_ml.__version__
    except ImportError:
        result["pytensor_ml_installed"] = False
        result["pytensor_ml_version"] = None

    return result


def get_device() -> str:
    """Return a short string describing the compute device.

    Returns
    -------
    str
        ``"apple_silicon"`` on Apple Silicon Macs, otherwise the
        ``platform.machine()`` value.
    """
    machine = platform.machine()
    if platform.system() == "Darwin" and machine == "arm64":
        return "apple_silicon"
    return machine


def get_peak_rss_mib() -> float:
    """Get peak RSS in mebibytes.

    Returns
    -------
    float
        Peak resident set size in MiB.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return round(rss / (1024 * 1024), 2)
    return round(rss / 1024, 2)


def get_mlx_peak_memory_mib() -> float | None:
    """Get MLX peak memory in mebibytes, or None if unavailable.

    Returns
    -------
    float or None
        Peak MLX memory in MiB, or None.
    """
    try:
        import mlx.core as mx
        if hasattr(mx, "get_peak_memory"):
            return round(mx.get_peak_memory() / (1024 * 1024), 2)
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
            return round(mx.metal.get_peak_memory() / (1024 * 1024), 2)
    except Exception:
        pass
    return None


def hash_token_ids(token_ids: list[int]) -> str:
    """Compute a SHA-256 hex digest over a list of token IDs.

    Accepts Python ``int`` and NumPy integer types (``np.int32``,
    ``np.int64``, etc.) by converting to ``int`` before range validation.

    Parameters
    ----------
    token_ids : list[int]
        Token identifiers (Python int or NumPy integer types).

    Returns
    -------
    str
        Hex digest string.
    """
    h = hashlib.sha256()
    for tid in token_ids:
        tid_int = int(tid)
        if tid_int != tid and not isinstance(tid, (int, np.integer)):
            raise ValueError(f"Token ID is not an integer type: {tid!r}")
        if not 0 <= tid_int <= 0xFFFFFFFF:
            raise ValueError(f"Token ID out of uint32 range: {tid!r}")
        h.update(tid_int.to_bytes(4, byteorder="little", signed=False))
    return h.hexdigest()


def validate_publication_prompt_tokens(
    prompt_text: str,
    token_ids: Sequence[int],
) -> None:
    """Require the pinned publication prompt to retain its known tokenization."""
    if prompt_text != DEFAULT_PROMPT:
        return
    actual_ids = tuple(int(token_id) for token_id in token_ids)
    if actual_ids != EXPECTED_PROMPT_TOKEN_IDS:
        raise ValueError(
            "Default prompt token IDs changed; refusing to record publication evidence"
        )
    actual_hash = hash_token_ids(list(actual_ids))
    if actual_hash != EXPECTED_PROMPT_TOKEN_HASH:
        raise ValueError("Default prompt token hash changed")


def decode_single_token(tokenizer: Any, token_id: int) -> str:
    """Decode one token and normalize the tokenizer result to text."""
    text = tokenizer.decode([token_id])
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    if not isinstance(text, str):
        return str(text)
    return text


def decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode a complete token sequence so subword boundaries are preserved.

    Parameters
    ----------
    tokenizer : Any
        mlx_lm tokenizer wrapper.
    token_ids : Sequence[int]
        Token identifiers to decode.

    Returns
    -------
    str
        Decoded text.
    """
    if not token_ids:
        return ""
    text = tokenizer.decode([int(token_id) for token_id in token_ids])
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    if not isinstance(text, str):
        return str(text)
    return text


def get_stop_token_ids(tokenizer: Any) -> frozenset[int]:
    """Read all tokenizer-defined end-of-generation token IDs.

    Parameters
    ----------
    tokenizer : Any
        mlx_lm tokenizer wrapper.

    Returns
    -------
    frozenset[int]
        Set of stop token IDs.
    """
    stop_ids = getattr(tokenizer, "eos_token_ids", None)
    if stop_ids is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            return frozenset([int(eos_id)])
        return frozenset()
    return frozenset(int(tid) for tid in stop_ids)


# ---------------------------------------------------------------------------
# MLX sync helper (imports mlx only when backend is mlx)
# ---------------------------------------------------------------------------


def _mlx_eval_tree(value: Any, *, label: str = "") -> float:
    """Evaluate all MLX leaves in *value* without host copy.

    Calls ``mx.eval`` to force materialization of the lazy computation
    graph, releasing references to predecessor weight tensors.  The value
    remains device-resident (MLX arrays are not copied to host).

    Preserves tuple/list/dict structure — never silently stacks.

    Parameters
    ----------
    value : array-like or tree
        MLX array, tuple, list, or dict of MLX arrays.
    label : str
        Diagnostic label for debugging.

    Returns
    -------
    float
        Elapsed seconds for the ``mx.eval`` call.
    """
    import mlx.core as mx

    t_start = time.perf_counter()
    mx.eval(value)
    return time.perf_counter() - t_start


def _mlx_host_copy(value: Any) -> tuple[float, Any]:
    """Host-copy an evaluated MLX tree while preserving its structure.

    The value must already have been evaluated via :func:`_mlx_eval_tree`.
    The returned array always owns its memory (a standalone copy, not a
    view of MLX device memory).

    Parameters
    ----------
    value : array-like
        Evaluated MLX array.

    Returns
    -------
    tuple of (float, Any)
        ``(elapsed_seconds, copied_tree)``. Array leaves are C-contiguous
        ``<f4`` NumPy arrays that own their data.
    """
    def copy_leaf_or_tree(item: Any) -> Any:
        if isinstance(item, tuple):
            return tuple(copy_leaf_or_tree(child) for child in item)
        if isinstance(item, list):
            return [copy_leaf_or_tree(child) for child in item]
        if isinstance(item, dict):
            return {
                key: copy_leaf_or_tree(child)
                for key, child in item.items()
            }
        return np.array(item, dtype=np.dtype("<f4"), copy=True, order="C")

    t_start = time.perf_counter()
    result = copy_leaf_or_tree(value)
    return time.perf_counter() - t_start, result


def _maybe_eval_tree(value: Any, backend: str, *, label: str = "") -> float:
    """Evaluate MLX tree only when *backend* is ``'mlx'``.

    For C and Numba backends the value is already materialized and no
    evaluation is needed.

    Parameters
    ----------
    value : array-like or tree
        Compiled function output.
    backend : str
        Current backend name.
    label : str
        Diagnostic label.

    Returns
    -------
    float
        Eval seconds (``0.0`` for non-MLX backends).
    """
    if backend != "mlx":
        return 0.0
    return _mlx_eval_tree(value, label=label)


def _maybe_host_copy(value: Any, backend: str) -> tuple[float, Any]:
    """Host-copy only when *backend* is ``'mlx'``.

    For C and Numba backends the value is already a NumPy array and no
    copy is needed.

    Parameters
    ----------
    value : array-like
        Compiled function output (already evaluated for MLX).
    backend : str
        Current backend name.

    Returns
    -------
    tuple of (float, numpy.ndarray)
        ``(copy_seconds, numpy_array)``.  ``copy_seconds`` is ``0.0``
        for non-MLX backends.
    """
    if backend != "mlx":
        return 0.0, value
    return _mlx_host_copy(value)


def _mlx_sync(value, *, label: str = "") -> tuple[float, Any]:
    """Legacy combined eval + host-copy (kept for backward compatibility).

    .. deprecated::
        Use :func:`_mlx_eval_tree` + :func:`_mlx_host_copy` separately.
    """
    eval_s = _mlx_eval_tree(value, label=label)
    copy_s, result = _mlx_host_copy(value)
    return eval_s + copy_s, result


def _maybe_sync(
    value: Any,
    backend: str,
    *,
    label: str = "",
) -> tuple[float, Any]:
    """Legacy combined sync (kept for backward compatibility).

    .. deprecated::
        Use :func:`_maybe_eval_tree` + :func:`_maybe_host_copy` separately.
    """
    if backend != "mlx":
        return 0.0, value
    return _mlx_sync(value, label=label)


# ---------------------------------------------------------------------------
# Backend linker/mode helpers
# ---------------------------------------------------------------------------


def get_backend_info(backend: str) -> dict[str, str]:
    """Return backend linker/mode description.

    Parameters
    ----------
    backend : str
        One of ``'c'``, ``'numba'``, or ``'mlx'``.

    Returns
    -------
    dict
        Keys: ``name``, ``linker``, ``mode``.
    """
    if backend == "c":
        return {"name": "c", "linker": "cvm", "mode": "o2"}
    elif backend == "numba":
        return {"name": "numba", "linker": "numba", "mode": "fast_compile"}
    elif backend == "mlx":
        return {"name": "mlx", "linker": "mlx", "mode": "fast_run+mlx"}
    raise ValueError(f"Unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# Tokenizer / chat template
# ---------------------------------------------------------------------------


def load_tokenizer_from_snapshot(snapshot_dir: Path) -> Any:
    """Load the tokenizer from a local snapshot without loading model weights.

    Uses ``transformers.AutoTokenizer`` which reads only tokenizer files
    (``tokenizer.json``, ``tokenizer_config.json``, ``special_tokens_map.json``)
    and never loads ``model.safetensors``.

    Parameters
    ----------
    snapshot_dir : Path
        Root directory of the local snapshot.

    Returns
    -------
    tokenizer
        A ``PreTrainedTokenizerFast`` instance with chat template support.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(snapshot_dir), local_files_only=True
    )


def format_and_tokenize(
    tokenizer: Any,
    prompt_text: str,
) -> tuple[str, list[int]]:
    """Apply chat template and tokenize.

    Parameters
    ----------
    tokenizer : tokenizer
        mlx_lm tokenizer wrapper.
    prompt_text : str
        User message text.

    Returns
    -------
    tuple
        ``(formatted_text, token_ids)``
    """
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # The Gemma chat template already emits its explicit <bos> token.
    token_ids = tokenizer.encode(formatted, add_special_tokens=False)
    return formatted, [int(t) for t in token_ids]


# ---------------------------------------------------------------------------
# MLX-LM reference forward pass
# ---------------------------------------------------------------------------


def run_mlx_reference(
    snapshot_dir: Path,
    token_ids: list[int],
) -> dict[str, Any]:
    """Run a direct MLX-LM forward pass with ``cache=None``.

    Calls ``model(mx.array(token_ids)[None], cache=None)`` and
    synchronizes with ``mx.eval`` to obtain all-position float32 logits.

    Parameters
    ----------
    snapshot_dir : Path
        Root directory of the local snapshot.
    token_ids : list[int]
        Prompt token IDs.

    Returns
    -------
    dict
        Keys: ``logits`` (np.ndarray, shape ``(1, T, V)``),
        ``load_s``, ``forward_s``, ``sync_s``,
        ``peak_memory_mib``, ``vocab_size``, ``seq_len``.
    """
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    t_load_start = time.time()
    model, _tokenizer = mlx_load(str(snapshot_dir))
    t_load = time.time() - t_load_start

    mx.reset_peak_memory()

    input_ids = mx.array(token_ids)[None]

    t_fwd_start = time.time()
    output = model(input_ids, cache=None).astype(mx.float32)
    t_fwd = time.time() - t_fwd_start

    t_sync_start = time.time()
    mx.eval(output)
    t_sync = time.time() - t_sync_start

    logits_np = np.asarray(output, dtype=np.float32)
    peak_mem = get_mlx_peak_memory_mib()

    return {
        "logits": logits_np,
        "load_s": t_load,
        "forward_s": t_fwd,
        "sync_s": t_sync,
        "peak_memory_mib": peak_mem,
        "vocab_size": logits_np.shape[-1],
        "seq_len": logits_np.shape[1],
    }


# ---------------------------------------------------------------------------
# Sibling adapter: gemma3n_weights
# ---------------------------------------------------------------------------


def _create_weight_loader(snapshot_dir: Path) -> Any:
    """Adapter for ``gemma3n_weights.Gemma3nWeightLoader.from_snapshot``.

    Parameters
    ----------
    snapshot_dir : Path
        Snapshot directory.

    Returns
    -------
    Gemma3nWeightLoader
        Initialized loader with mmap open.
    """
    from cetagostini.utils.pytensor.gemma3n_weights import Gemma3nWeightLoader

    return Gemma3nWeightLoader.from_snapshot(snapshot_dir)


# ---------------------------------------------------------------------------
# Sibling adapter: gemma3n_pytensor
# ---------------------------------------------------------------------------


def _build_pytensor_config(text_config: Any) -> Any:
    """Bridge ``Gemma3nTextConfig`` → ``gemma3n_pytensor.Gemma3nConfig``.

    Parameters
    ----------
    text_config : Gemma3nTextConfig
        From gemma3n_weights.

    Returns
    -------
    Gemma3nConfig
        For gemma3n_pytensor compile functions.
    """
    from cetagostini.utils.pytensor.gemma3n_pytensor import Gemma3nConfig

    return Gemma3nConfig.from_text_config(text_config)


def _compile_graphs(
    pt_config: Any,
    seq_len: int,
    backend: str,
    text_config: Any,
) -> dict[str, Any]:
    """Compile all PyTensor graph functions.

    Parameters
    ----------
    pt_config : Gemma3nConfig
        For gemma3n_pytensor.
    seq_len : int
        Sequence length.
    backend : str
        ``'c'`` or ``'numba'``.
    text_config : Gemma3nTextConfig
        For per-layer sparsity info.

    Returns
    -------
    dict
        Compiled functions keyed by purpose.
    """
    from cetagostini.utils.pytensor.gemma3n_pytensor import (
        compile_initial_projections,
        compile_per_layer_projection,
        compile_decoder_layer,
        compile_final_unembed,
        compile_per_chunk_logits,
    )

    B = 1
    T = seq_len

    initial_fn = compile_initial_projections(pt_config, B, T, backend=backend)
    per_layer_proj_fn = compile_per_layer_projection(pt_config, B, T, backend=backend)
    final_fn = compile_final_unembed(pt_config, B, T, backend=backend)

    # Layer equations differ only by sparse versus dense GELU. Attention
    # kind is supplied through the mask and RoPE tables at runtime.
    # Compile one function per distinct numeric sparsity value in the
    # activation_sparsity_pattern (not bool/hardcoded 0.95).
    from dataclasses import replace

    distinct_sparsity_values = sorted(set(text_config.activation_sparsity_pattern))
    layer_fns: dict[float, Any] = {}
    for sparsity_value in distinct_sparsity_values:
        has_sparsity = sparsity_value > 0.0
        layer_config = replace(
            pt_config,
            activation_sparsity=sparsity_value,
        )
        layer_fns[sparsity_value] = compile_decoder_layer(
            layer_config,
            B,
            T,
            has_sparsity=has_sparsity,
            backend=backend,
        )

    chunk_size = 4096
    full_chunk_size = min(chunk_size, text_config.vocab_size)
    chunk_sizes = {full_chunk_size}
    remainder = text_config.vocab_size % chunk_size
    if text_config.vocab_size > chunk_size and remainder:
        chunk_sizes.add(remainder)
    logit_fns = {
        size: compile_per_chunk_logits(
            hidden_size=text_config.hidden_size,
            batch_size=B,
            seq_len=T,
            chunk_size=size,
            softcap=text_config.final_logit_softcapping,
            backend=backend,
        )
        for size in chunk_sizes
    }
    return {
        "initial_fn": initial_fn,
        "per_layer_proj_fn": per_layer_proj_fn,
        "layer_fns": layer_fns,
        "final_fn": final_fn,
        "logit_fns": logit_fns,
    }


# ---------------------------------------------------------------------------
# Weight transposition helpers
# ---------------------------------------------------------------------------

# Keys that are Linear projections stored as [out, in] and need transpose
# to [in, out] for graph consumption.
_LINEAR_KEYS = frozenset({
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "laurel.linear_left",
    "laurel.linear_right",
    "altup.modality_router",
    "altup.prediction_coefs",
    "altup.correction_coefs",
    "per_layer_input_gate",
    "per_layer_projection",
})


def transpose_layer_weights(layer_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Transpose all Linear [out, in] weights to [in, out] for graph use.

    Parameters
    ----------
    layer_dict : dict
        Output of ``Gemma3nWeightLoader.load_layer(i)``.

    Returns
    -------
    dict
        Same keys, with Linear weights transposed.
    """
    result: dict[str, np.ndarray] = {}
    for key, arr in layer_dict.items():
        if key in _LINEAR_KEYS:
            result[key] = np.ascontiguousarray(arr.T, dtype=np.float32)
        else:
            result[key] = arr
    return result


def _unpack_layer_args(w: dict[str, np.ndarray]) -> tuple:
    """Unpack transposed layer weights into positional args for decoder_layer.

    Parameters
    ----------
    w : dict
        Transposed layer weight dict.

    Returns
    -------
    tuple
        24 positional weight arrays matching compile_decoder_layer's
        input order (after hidden, mask, pli, cos, sin).
    """
    return (
        w["self_attn.q_proj"],          # q_w
        w["self_attn.k_proj"],          # k_w
        w["self_attn.v_proj"],          # v_w
        w["self_attn.o_proj"],          # o_w
        w["self_attn.q_norm"],          # q_ng
        w["self_attn.k_norm"],          # k_ng
        w["mlp.gate_proj"],             # gate_w
        w["mlp.up_proj"],               # up_w
        w["mlp.down_proj"],             # down_w
        w["laurel.linear_left"],        # ll_w
        w["laurel.linear_right"],       # lr_w
        w["laurel.post_laurel_norm"],   # ln_g
        w["altup.prediction_coefs"],    # pred_w
        w["altup.correction_coefs"],    # corr_w
        w["altup.modality_router"],     # mr_w
        w["altup.router_norm"],         # rn_g
        w["altup.correct_output_scale"],# cos_scale
        w["input_layernorm"],           # iln_g
        w["post_attention_layernorm"],  # paln_g
        w["pre_feedforward_layernorm"], # pfln_g
        w["post_feedforward_layernorm"],# pfln2_g
        w["post_per_layer_input_norm"], # plin_g
        w["per_layer_input_gate"],      # pli_gw
        w["per_layer_projection"],      # pli_pw
    )


def transpose_global_weight(arr: np.ndarray) -> np.ndarray:
    """Transpose a global Linear weight [out, in] → [in, out].

    Parameters
    ----------
    arr : np.ndarray
        Weight array from loader.

    Returns
    -------
    np.ndarray
        Transposed contiguous float32 array.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected a rank-2 global weight, got rank {arr.ndim}")
    return np.ascontiguousarray(arr.T, dtype=np.float32)


# ---------------------------------------------------------------------------
# PyTensor full pipeline
# ---------------------------------------------------------------------------


def run_pytensor_forward(
    loader: Any,
    compiled: dict[str, Any],
    token_ids: list[int],
    text_config: Any,
    pt_config: Any,
    backend: str,
    *,
    layer_weight_cache: Sequence[dict[str, np.ndarray]] | None = None,
    capture_layer_weights: bool = False,
) -> dict[str, Any]:
    """Run the full PyTensor forward pass.

    Implements the exact pipeline:
    1. Row-load main embeddings, multiply sqrt(H)
    2. Row-load per-layer embedding rows, reshape [B,T,L,H_pl], multiply sqrt(H_pl)
    3. Load global per_layer_model_projection (transpose) and norm
    4. Initial AltUp projections
    5. Per-layer projection
    6. Sequential layers 0..34 with mask/RoPE routing
    7. Final unembed (transpose + norm)
    8. Chunked logit projection preserving vocab order with softcap

    Parameters
    ----------
    loader : Gemma3nWeightLoader
        Weight loader with mmap open.
    compiled : dict
        Compiled graph functions from ``_compile_graphs``.
    token_ids : list[int]
        Prompt token IDs.
    text_config : Gemma3nTextConfig
        From gemma3n_weights.
    pt_config : Gemma3nConfig
        For gemma3n_pytensor.
    backend : str
        ``'c'``, ``'numba'``, or ``'mlx'``.
    layer_weight_cache : sequence of dict or None
        Optional already-dequantized, transposed decoder weights.
    capture_layer_weights : bool
        Retain newly loaded decoder weights for later prefix evaluations.

    Returns
    -------
    dict
        Keys: ``logits``, ``per_layer_s``, ``embed_s``,
        ``initial_s``, ``per_layer_proj_s``, ``final_s``,
        ``logits_s``, ``total_s``, ``layers_completed``,
        ``layer_types_used``, ``rope_bases_used``,
        ``sparse_layers_used``, ``chunks_processed``,
        ``mlx_sync_s``, ``mlx_sync_stages``.
    """
    from cetagostini.utils.pytensor.gemma3n_pytensor import (
        build_rope_table,
        causal_mask,
        sliding_window_mask,
    )

    B = 1
    T = len(token_ids)
    H = text_config.hidden_size
    H_pl = text_config.hidden_size_per_layer_input
    L = text_config.num_hidden_layers
    n = text_config.altup_num_inputs

    if layer_weight_cache is not None and len(layer_weight_cache) != L:
        raise ValueError(
            f"layer_weight_cache must contain {L} layers, "
            f"got {len(layer_weight_cache)}"
        )
    captured_layer_weights = [] if capture_layer_weights else None

    t_total_start = time.time()
    mlx_eval_s = 0.0
    mlx_host_copy_s = 0.0
    mlx_stages: list[dict[str, Any]] = []

    def _record_stage(label: str, eval_s: float, host_copy_s: float) -> None:
        """Record an ordered stage entry and accumulate totals."""
        mlx_stages.append({
            "label": label,
            "eval_s": eval_s,
            "host_copy_s": host_copy_s,
        })

    # ── 1. Main input embeddings: row-load + sqrt(H) ────────────────
    t_emb_start = time.time()
    embed_rows = loader.load_input_embedding_rows(token_ids)
    # embed_rows: [T, H], scale by sqrt(H)
    h0 = (embed_rows * np.float32(math.sqrt(H))).reshape(B, T, H).astype(np.float32)
    t_emb = time.time() - t_emb_start

    # ── 2. Per-layer embedding rows: row-load + reshape + sqrt(H_pl) ─
    t_ple_start = time.time()
    ple_flat = loader.load_per_layer_embedding_rows(token_ids)
    # ple_flat: [T, H_pl * L] → reshape to [T, L, H_pl] → [B, T, L, H_pl]
    per_layer_embeds = (
        ple_flat.reshape(T, L, H_pl) * np.float32(math.sqrt(H_pl))
    ).reshape(B, T, L, H_pl).astype(np.float32)
    t_ple = time.time() - t_ple_start

    # ── 3. Global per_layer_model_projection (transpose) + norm ──────
    t_global_start = time.time()
    plm_proj_w = transpose_global_weight(
        loader.load_global("per_layer_model_projection")
    )
    plm_proj_norm_gamma = loader.load_global("per_layer_projection_norm")
    t_global = time.time() - t_global_start

    # ── 4. Initial AltUp projections ─────────────────────────────────
    t_initial_start = time.time()
    altup_proj_weights = []
    for i in range(n - 1):
        w = transpose_global_weight(
            loader.load_global(f"altup_projections.{i}")
        )
        altup_proj_weights.append(w)

    hidden_streams = compiled["initial_fn"](h0, *altup_proj_weights)
    # hidden_streams: [n, B, T, H]
    # Evaluate MLX leaves; keep device-resident for intermediate stages.
    eval_dt = _maybe_eval_tree(
        hidden_streams, backend, label="initial_projections",
    )
    mlx_eval_s += eval_dt
    _record_stage("initial_projections", eval_dt, 0.0)
    # Release predecessor weights after eval
    del altup_proj_weights
    gc.collect()
    t_initial = time.time() - t_initial_start

    # ── 5. Per-layer projection ──────────────────────────────────────
    t_plp_start = time.time()
    per_layer_inputs = compiled["per_layer_proj_fn"](
        h0, plm_proj_w, plm_proj_norm_gamma, per_layer_embeds,
    )
    # per_layer_inputs: [B, T, L, H_pl]
    eval_dt = _maybe_eval_tree(
        per_layer_inputs, backend, label="per_layer_projection",
    )
    mlx_eval_s += eval_dt
    _record_stage("per_layer_projection", eval_dt, 0.0)
    t_plp = time.time() - t_plp_start

    # Release per-layer embeds (no longer needed)
    del per_layer_embeds, ple_flat
    gc.collect()

    # ── 6. Build RoPE tables (both bases) ────────────────────────────
    hd = text_config.head_dim
    cos_full, sin_full = build_rope_table(text_config.rope_theta, hd, T)
    cos_local, sin_local = build_rope_table(text_config.rope_local_base_freq, hd, T)

    # ── 7. Sequential layer streaming ────────────────────────────────
    per_layer_s: list[float] = []
    layer_types_used: list[str] = []
    rope_bases_used: list[str] = []
    sparse_layers_used: list[int] = []

    for layer_idx in range(L):
        t_layer_start = time.time()

        # Load and transpose layer weights
        if layer_weight_cache is None:
            raw_weights = loader.load_layer(layer_idx)
            w = transpose_layer_weights(raw_weights)
            del raw_weights
            if captured_layer_weights is not None:
                captured_layer_weights.append(w)
        else:
            w = layer_weight_cache[layer_idx]

        # Determine layer type, mask, and RoPE
        layer_type = text_config.layer_types[layer_idx]
        layer_types_used.append(layer_type)

        if layer_type == "full_attention":
            mask = causal_mask(T)
            cos, sin = cos_full, sin_full
            rope_bases_used.append("1M")
        else:
            mask = sliding_window_mask(T, text_config.sliding_window)
            cos, sin = cos_local, sin_local
            rope_bases_used.append("10K")

        # Per-layer input for this layer: [B, T, H_pl]
        pli = per_layer_inputs[:, :, layer_idx, :]

        # Sparsity — select by distinct numeric value, not bool
        sparsity = text_config.activation_sparsity_pattern[layer_idx]
        if sparsity > 0.0:
            sparse_layers_used.append(layer_idx)

        # Get the reusable graph for this layer's activation sparsity value.
        layer_fn = compiled["layer_fns"][sparsity]

        # Call decoder layer
        layer_args = _unpack_layer_args(w)
        hidden_streams = layer_fn(
            hidden_streams, mask, pli, cos, sin, *layer_args,
        )

        # Evaluate MLX leaves before releasing layer weights.
        # Keep device-resident — no intermediate host copy.
        eval_dt = _maybe_eval_tree(
            hidden_streams, backend, label=f"layer_{layer_idx}",
        )
        mlx_eval_s += eval_dt
        _record_stage(f"layer_{layer_idx}", eval_dt, 0.0)

        per_layer_s.append(time.time() - t_layer_start)

        # Release layer weights immediately (after eval)
        del w, layer_args
        gc.collect()

    # Release per_layer_inputs
    del per_layer_inputs
    gc.collect()

    # ── 8. Final unembed (transpose + norm) ──────────────────────────
    t_final_start = time.time()
    altup_unembed_weights = []
    for i in range(n - 1):
        w = transpose_global_weight(
            loader.load_global(f"altup_unembed_projections.{i}")
        )
        altup_unembed_weights.append(w)

    final_norm_gamma = loader.load_global("norm")

    hidden_final = compiled["final_fn"](
        hidden_streams, *altup_unembed_weights, final_norm_gamma,
    )
    # hidden_final: [B, T, H]
    eval_dt = _maybe_eval_tree(
        hidden_final, backend, label="final_unembed",
    )
    mlx_eval_s += eval_dt
    _record_stage("final_unembed", eval_dt, 0.0)
    t_final = time.time() - t_final_start

    # Release streams
    del hidden_streams, altup_unembed_weights, final_norm_gamma
    gc.collect()

    # ── 9. Chunked logit projection (tied embeddings + softcap) ──────
    t_logits_start = time.time()
    vocab_size = text_config.vocab_size
    softcap = text_config.final_logit_softcapping
    chunk_size = 4096

    # Collect all chunks preserving vocab order.
    # Each chunk: eval MLX leaves, then host-copy to C-contiguous <f4 NumPy.
    all_logits_parts: list[np.ndarray] = []
    chunks_processed = 0

    for start, end, chunk_emb in loader.iter_output_embedding_chunks(chunk_size):
        chunk_logits = compiled["logit_fns"][end - start](
            hidden_final,
            chunk_emb,
        ).reshape(B * T, end - start)

        # Evaluate MLX leaves before releasing embedding chunk
        stage_label = f"vocab_chunk_{start}_{end}"
        eval_dt = _maybe_eval_tree(
            chunk_logits, backend, label=stage_label,
        )
        mlx_eval_s += eval_dt

        # Host-copy each logits chunk to owning C-contiguous <f4 NumPy
        copy_dt, chunk_logits_np = _maybe_host_copy(chunk_logits, backend)
        mlx_host_copy_s += copy_dt

        _record_stage(stage_label, eval_dt, copy_dt)

        all_logits_parts.append(chunk_logits_np)
        chunks_processed += 1

    # Concatenate preserving vocab order: [B*T, V]
    logits_flat = np.concatenate(all_logits_parts, axis=-1)
    logits_np = logits_flat.reshape(B, T, vocab_size).astype(np.float32)
    t_logits = time.time() - t_logits_start

    t_total = time.time() - t_total_start

    result = {
        "logits": logits_np,
        "per_layer_s": per_layer_s,
        "embed_s": t_emb,
        "ple_s": t_ple,
        "global_load_s": t_global,
        "initial_s": t_initial,
        "per_layer_proj_s": t_plp,
        "final_s": t_final,
        "logits_s": t_logits,
        "total_s": t_total,
        "layers_completed": L,
        "layer_types_used": layer_types_used,
        "rope_bases_used": rope_bases_used,
        "sparse_layers_used": sparse_layers_used,
        "chunks_processed": chunks_processed,
        "mlx_eval_s": mlx_eval_s,
        "mlx_host_copy_s": mlx_host_copy_s,
        "mlx_stages": mlx_stages,
        "stage_count": len(mlx_stages),
    }
    if captured_layer_weights is not None:
        result["_layer_weight_cache"] = captured_layer_weights
    return result


# ---------------------------------------------------------------------------
# All-position metrics
# ---------------------------------------------------------------------------


def compute_all_position_metrics(
    ref_logits: np.ndarray,
    pt_logits: np.ndarray,
) -> dict[str, Any]:
    """Compute comprehensive all-position comparison metrics.

    Parameters
    ----------
    ref_logits : np.ndarray
        Reference logits, shape ``(1, T, V)``.
    pt_logits : np.ndarray
        PyTensor logits, shape ``(1, T, V)``.

    Returns
    -------
    dict
        Keys: ``all_finite_ref``, ``all_finite_pt``, ``per_position``
        (list of per-position metrics), ``aggregate`` (summary metrics),
        ``final_top1_match``, ``final_top1_ref``, ``final_top1_pt``.
    """
    if ref_logits.ndim != 3 or pt_logits.ndim != 3:
        raise ValueError("Expected rank-3 logits with shape [1, sequence, vocabulary]")
    if ref_logits.shape != pt_logits.shape:
        raise ValueError(
            f"Logit shapes must match, got {ref_logits.shape} and {pt_logits.shape}"
        )
    if ref_logits.shape[0] != 1:
        raise ValueError(f"Only batch size 1 is supported, got {ref_logits.shape[0]}")
    if ref_logits.shape[1] == 0 or ref_logits.shape[2] == 0:
        raise ValueError("Sequence and vocabulary dimensions must be non-empty")
    if ref_logits.shape[2] < 10:
        raise ValueError("Vocabulary must contain at least 10 logits for top-10 metrics")

    ref = ref_logits[0]  # (T, V)
    pt = pt_logits[0]    # (T, V)
    T = ref.shape[0]

    all_finite_ref = bool(np.all(np.isfinite(ref)))
    all_finite_pt = bool(np.all(np.isfinite(pt)))

    per_position: list[dict[str, Any]] = []
    all_max_abs_diff: list[float] = []
    all_mean_abs_diff: list[float] = []
    all_rmse: list[float] = []
    all_cosine: list[float] = []
    all_pearson: list[float] = []
    all_top10_overlap: list[int] = []

    for pos in range(T):
        r = ref[pos]
        p = pt[pos]

        pos_metrics: dict[str, Any] = {"position": pos}
        pos_metrics["finite_ref"] = bool(np.all(np.isfinite(r)))
        pos_metrics["finite_pt"] = bool(np.all(np.isfinite(p)))

        if not (pos_metrics["finite_ref"] and pos_metrics["finite_pt"]):
            per_position.append(pos_metrics)
            continue

        r64 = r.astype(np.float64)
        p64 = p.astype(np.float64)
        abs_diff = np.abs(r64 - p64)
        max_abs = float(np.max(abs_diff))
        mean_abs = float(np.mean(abs_diff))
        rmse = float(np.sqrt(np.mean(abs_diff ** 2)))

        pos_metrics["max_abs_diff"] = max_abs
        pos_metrics["mean_abs_diff"] = mean_abs
        pos_metrics["rmse"] = rmse

        all_max_abs_diff.append(max_abs)
        all_mean_abs_diff.append(mean_abs)
        all_rmse.append(rmse)

        r_norm = float(np.linalg.norm(r64))
        p_norm = float(np.linalg.norm(p64))
        if r_norm > 0 and p_norm > 0:
            cosine = float(np.dot(r64, p64) / (r_norm * p_norm))
            cosine = float(np.clip(cosine, -1.0, 1.0))
        elif np.array_equal(r64, p64):
            cosine = 1.0
        else:
            cosine = 0.0
        pos_metrics["cosine"] = cosine
        all_cosine.append(cosine)

        r_std = float(np.std(r64))
        p_std = float(np.std(p64))
        if r_std > 0 and p_std > 0:
            pearson = float(np.corrcoef(r64, p64)[0, 1])
        elif np.array_equal(r64, p64):
            pearson = 1.0
        else:
            pearson = 0.0
        pos_metrics["pearson"] = pearson
        all_pearson.append(pearson)

        ref_top10 = set(np.argsort(r)[-10:].tolist())
        pt_top10 = set(np.argsort(p)[-10:].tolist())
        overlap = len(ref_top10 & pt_top10)
        pos_metrics["top10_overlap"] = overlap
        all_top10_overlap.append(overlap)

        pos_metrics["top1_match"] = int(np.argmax(r)) == int(np.argmax(p))

        per_position.append(pos_metrics)

    final_pos = T - 1
    final_is_finite = bool(
        np.all(np.isfinite(ref[final_pos]))
        and np.all(np.isfinite(pt[final_pos]))
    )
    if final_is_finite:
        final_top1_ref = int(np.argmax(ref[final_pos]))
        final_top1_pt = int(np.argmax(pt[final_pos]))
        final_top1_match = final_top1_ref == final_top1_pt
    else:
        final_top1_ref = None
        final_top1_pt = None
        final_top1_match = False

    aggregate: dict[str, Any] = {}
    if all_max_abs_diff:
        aggregate["max_abs_diff_max"] = max(all_max_abs_diff)
        aggregate["max_abs_diff_mean"] = float(np.mean(all_max_abs_diff))
        aggregate["mean_abs_diff_max"] = max(all_mean_abs_diff)
        aggregate["mean_abs_diff_mean"] = float(np.mean(all_mean_abs_diff))
        aggregate["rmse_max"] = max(all_rmse)
        aggregate["rmse_mean"] = float(np.mean(all_rmse))
        aggregate["cosine_min"] = min(all_cosine)
        aggregate["cosine_mean"] = float(np.mean(all_cosine))
        aggregate["pearson_min"] = min(all_pearson)
        aggregate["pearson_mean"] = float(np.mean(all_pearson))
        aggregate["top10_overlap_min"] = min(all_top10_overlap)
        aggregate["top10_overlap_mean"] = float(np.mean(all_top10_overlap))

    all_top1_match = all(
        position.get("top1_match", False) for position in per_position
    )

    return {
        "all_finite_ref": all_finite_ref,
        "all_finite_pt": all_finite_pt,
        "n_positions": T,
        "per_position": per_position,
        "aggregate": aggregate,
        "final_top1_ref": final_top1_ref,
        "final_top1_pt": final_top1_pt,
        "final_top1_match": final_top1_match,
        "all_top1_match": all_top1_match,
    }


# ---------------------------------------------------------------------------
# Publication thresholds
# ---------------------------------------------------------------------------


def check_publication_thresholds(metrics: dict[str, Any]) -> dict[str, Any]:
    """Check hard publication thresholds against metrics.

    Parameters
    ----------
    metrics : dict
        Output of ``compute_all_position_metrics``.

    Returns
    -------
    dict
        Keys: ``passed`` (bool), ``checks`` (list of per-check statuses).
    """
    agg = metrics.get("aggregate", {})
    checks: list[dict[str, Any]] = []

    for name in ("all_finite_ref", "all_finite_pt"):
        actual = metrics.get(name, False)
        checks.append({
            "name": name,
            "threshold": True,
            "actual": actual,
            "passed": actual is True,
        })

    # Worst-position cosine
    cosine_min = agg.get("cosine_min", -1.0)
    checks.append({
        "name": "cosine_min",
        "threshold": PUB_COSINE_MIN,
        "actual": cosine_min,
        "passed": cosine_min >= PUB_COSINE_MIN,
    })

    # Worst-position Pearson correlation
    pearson_min = agg.get("pearson_min", -1.0)
    checks.append({
        "name": "pearson_min",
        "threshold": PUB_PEARSON_MIN,
        "actual": pearson_min,
        "passed": pearson_min >= PUB_PEARSON_MIN,
    })

    # Every prompt position must preserve top-1.
    all_top1_match = metrics.get("all_top1_match", False)
    checks.append({
        "name": "all_top1_match",
        "threshold": PUB_ALL_TOP1_MATCH,
        "actual": all_top1_match,
        "passed": all_top1_match == PUB_ALL_TOP1_MATCH,
    })

    # Top10 overlap mean
    top10_mean = agg.get("top10_overlap_mean", 0.0)
    checks.append({
        "name": "top10_overlap_mean",
        "threshold": PUB_TOP10_OVERLAP_MEAN_MIN,
        "actual": top10_mean,
        "passed": top10_mean >= PUB_TOP10_OVERLAP_MEAN_MIN,
    })

    all_passed = all(c["passed"] for c in checks)
    return {"passed": all_passed, "checks": checks}


# ---------------------------------------------------------------------------
# Oracle consumption (reference report + logits artifact)
# ---------------------------------------------------------------------------


class OracleVerificationError(Exception):
    """Raised when the standalone oracle fails identity verification."""


def _load_strict_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object while rejecting non-standard non-finite tokens."""
    def reject_constant(token: str) -> Any:
        raise ValueError(f"non-finite JSON token: {token}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise OracleVerificationError(
            f"Failed to load reference report: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise OracleVerificationError("Reference report root must be an object")
    return value


def _verify_implementation_identity(
    oracle_implementation: dict[str, Any],
    current_implementation: dict[str, Any],
) -> None:
    """Require immutable implementation provenance to match the current run."""
    required_equal = (
        "git_commit",
        "environment_yml_sha256",
        "source_hashes",
        "implementation_manifest_sha256",
        "python_executable",
        "environment",
        "package_versions",
        "module_paths",
    )
    if oracle_implementation.get("git_clean") is not True:
        raise OracleVerificationError("Oracle was not produced from a clean worktree")
    if current_implementation.get("git_clean") is not True:
        raise OracleVerificationError("Current backend worktree is not clean")
    for key in required_equal:
        if oracle_implementation.get(key) != current_implementation.get(key):
            raise OracleVerificationError(
                f"Implementation provenance mismatch for {key!r}"
            )


def load_and_verify_reference_report(
    reference_report_path: Path,
    *,
    run_id: str,
    snapshot_dir: Path,
    config_dict: dict[str, Any],
    manifest: list[dict[str, Any]],
    prompt_text: str,
    formatted_text: str,
    token_ids: list[int],
    implementation_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Load and verify the standalone oracle JSON report.

    Performs strict identity checks before any backend execution:

    1. JSON loads successfully.
    2. Required top-level keys present (``model``, ``prompt``, ``reference``).
    3. ``model.repo`` matches ``EXPECTED_REPO``.
    4. ``model.revision`` matches snapshot directory basename.
    5. ``model.model_type`` matches ``EXPECTED_MODEL_TYPE``.
    6. ``model.architecture`` matches ``EXPECTED_ARCHITECTURE``.
    7. ``prompt.token_hash`` matches current prompt token IDs.
    8. ``file_manifest`` entries match current snapshot manifest.

    Parameters
    ----------
    reference_report_path : Path
        Path to the oracle JSON report.
    snapshot_dir : Path
        Current snapshot directory.
    config_dict : dict
        Parsed config.json.
    manifest : list[dict]
        Current snapshot file manifest.
    token_ids : list[int]
        Current prompt token IDs.

    Returns
    -------
    dict
        The loaded reference report.

    Raises
    ------
    OracleVerificationError
        If any identity check fails.
    """
    ref_report = _load_strict_json_object(reference_report_path)

    # Required top-level keys
    for key in (
        "schema_version",
        "run_id",
        "model",
        "prompt",
        "reference",
        "raw_artifact",
        "provenance",
    ):
        if key not in ref_report:
            raise OracleVerificationError(
                f"Reference report missing required key: {key!r}"
            )
    if ref_report["schema_version"] != GEMMA3N_ORACLE_SCHEMA_VERSION:
        raise OracleVerificationError("Oracle schema version mismatch")
    if ref_report["run_id"] != run_id:
        raise OracleVerificationError(
            f"Oracle run ID mismatch: expected {run_id!r}, "
            f"got {ref_report['run_id']!r}"
        )

    # Model identity
    ref_model = ref_report["model"]
    if ref_model.get("repo") != EXPECTED_REPO:
        raise OracleVerificationError(
            f"Model repo mismatch: expected {EXPECTED_REPO!r}, "
            f"got {ref_model.get('repo')!r}"
        )
    revision = detect_revision(snapshot_dir)
    if ref_model.get("revision") != revision:
        raise OracleVerificationError(
            f"Model revision mismatch: expected {revision!r}, "
            f"got {ref_model.get('revision')!r}"
        )
    if ref_model.get("model_type") != EXPECTED_MODEL_TYPE:
        raise OracleVerificationError(
            f"Model type mismatch: expected {EXPECTED_MODEL_TYPE!r}, "
            f"got {ref_model.get('model_type')!r}"
        )
    if ref_model.get("architecture") != EXPECTED_ARCHITECTURE:
        raise OracleVerificationError(
            f"Architecture mismatch: expected {EXPECTED_ARCHITECTURE!r}, "
            f"got {ref_model.get('architecture')!r}"
        )
    ref_quantization = ref_model.get("quantization", {})
    if ref_quantization.get("bits") != EXPECTED_BITS:
        raise OracleVerificationError("Quantization bits mismatch")
    if ref_quantization.get("group_size") != EXPECTED_GROUP_SIZE:
        raise OracleVerificationError("Quantization group size mismatch")
    if config_dict.get("model_type") != EXPECTED_MODEL_TYPE:
        raise OracleVerificationError("Current snapshot model type is invalid")
    if EXPECTED_ARCHITECTURE not in config_dict.get("architectures", []):
        raise OracleVerificationError("Current snapshot architecture is invalid")

    # Prompt / token identity
    ref_prompt = ref_report["prompt"]
    expected_token_ids = [int(token_id) for token_id in token_ids]
    prompt_expectations = {
        "text": prompt_text,
        "formatted": formatted_text,
        "token_ids": expected_token_ids,
        "n_tokens": len(expected_token_ids),
    }
    for key, expected in prompt_expectations.items():
        if ref_prompt.get(key) != expected:
            raise OracleVerificationError(
                f"Prompt identity mismatch for {key!r}"
            )
    expected_token_hash = hash_token_ids(token_ids)
    if ref_prompt.get("token_hash") != expected_token_hash:
        raise OracleVerificationError(
            f"Token hash mismatch: expected {expected_token_hash!r}, "
            f"got {ref_prompt.get('token_hash')!r}"
        )

    # File manifest identity
    ref_manifest = ref_model.get("manifest", [])
    if len(ref_manifest) != len(manifest):
        raise OracleVerificationError(
            f"File manifest length mismatch: expected {len(manifest)}, "
            f"got {len(ref_manifest)}"
        )
    for ref_entry, cur_entry in zip(ref_manifest, manifest):
        if ref_entry.get("name") != cur_entry["name"]:
            raise OracleVerificationError(
                f"File manifest name mismatch: "
                f"expected {cur_entry['name']!r}, "
                f"got {ref_entry.get('name')!r}"
            )
        if ref_entry.get("sha256") != cur_entry["sha256"]:
            raise OracleVerificationError(
                f"File manifest SHA-256 mismatch for "
                f"{cur_entry['name']!r}"
            )

    raw_artifact = ref_report["raw_artifact"]
    ref_info = ref_report["reference"]
    if ref_info.get("logits_sha256") != raw_artifact.get("canonical_sha256"):
        raise OracleVerificationError(
            "Oracle reference hash does not match its artifact manifest"
        )
    expected_shape = [1, len(expected_token_ids), ref_info.get("vocab_size")]
    if ref_info.get("shape") != expected_shape:
        raise OracleVerificationError("Oracle reference shape mismatch")
    if raw_artifact.get("shape") != expected_shape:
        raise OracleVerificationError("Oracle artifact shape mismatch")

    provenance = ref_report["provenance"]
    if provenance.get("run_id") != run_id:
        raise OracleVerificationError("Oracle provenance run ID mismatch")
    if provenance.get("schema_version") != GEMMA3N_ORACLE_SCHEMA_VERSION:
        raise OracleVerificationError("Oracle provenance schema mismatch")
    oracle_implementation = provenance.get("implementation")
    if not isinstance(oracle_implementation, dict):
        raise OracleVerificationError("Oracle implementation provenance is missing")
    _verify_implementation_identity(
        oracle_implementation,
        implementation_manifest,
    )

    return ref_report


def load_and_verify_reference_logits(
    reference_logits_path: Path,
    reference_report: dict[str, Any],
    *,
    expected_seq_len: int,
) -> np.ndarray:
    """Load and verify the standalone oracle logits NPY artifact.

    Uses :func:`evidence.verify_npy_artifact` for strict artifact
    verification, then checks shape and logits hash.

    Parameters
    ----------
    reference_logits_path : Path
        Path to the oracle logits ``.npy`` file.
    reference_report : dict
        The verified reference report.
    expected_seq_len : int
        Expected sequence length (number of prompt tokens).

    Returns
    -------
    np.ndarray
        Verified logits array, shape ``(1, T, V)``, dtype ``<f4``.

    Raises
    ------
    OracleVerificationError
        If any verification check fails.
    """
    ref_info = reference_report["reference"]
    expected_vocab = ref_info.get("vocab_size")
    expected_shape = [1, expected_seq_len, expected_vocab]

    # Build a manifest from the reference report's known shape/hash
    # We need to verify the file exists and matches expectations.
    if not reference_logits_path.exists():
        raise OracleVerificationError(
            f"Reference logits not found: {reference_logits_path}"
        )

    artifact_manifest = reference_report.get("raw_artifact")
    if not isinstance(artifact_manifest, dict):
        raise OracleVerificationError("Reference artifact manifest is missing")
    if artifact_manifest.get("basename") != reference_logits_path.name:
        raise OracleVerificationError(
            "Reference logits basename does not match oracle manifest"
        )
    if artifact_manifest.get("shape") != expected_shape:
        raise OracleVerificationError(
            f"Reference logits shape mismatch: "
            f"expected {expected_shape}, got {artifact_manifest.get('shape')}"
        )

    # Verify the file against the manifest signed by the oracle report.
    try:
        logits = verify_npy_artifact(reference_logits_path, artifact_manifest)
    except ArtifactVerificationError as exc:
        raise OracleVerificationError(
            f"Reference logits verification failed: {exc}"
        ) from exc

    # Verify logits hash matches the reference report
    actual_hash = hashlib.sha256(logits.tobytes()).hexdigest()
    expected_hash = ref_info.get("logits_sha256")
    if expected_hash and actual_hash != expected_hash:
        raise OracleVerificationError(
            f"Reference logits SHA-256 mismatch: "
            f"expected {expected_hash!r}, got {actual_hash!r}"
        )

    return logits


# ---------------------------------------------------------------------------
# MLX memory helpers for backend report
# ---------------------------------------------------------------------------


def get_mlx_memory_snapshot() -> dict[str, Any] | None:
    """Capture MLX allocator state (baseline/current/peak) if available.

    Returns ``None`` when MLX is not installed or the API is unavailable.

    Returns
    -------
    dict or None
        Keys: ``baseline_mib``, ``current_mib``, ``peak_mib``.
    """
    try:
        import mlx.core as mx

        try:
            mlx_version = importlib.metadata.version("mlx")
        except importlib.metadata.PackageNotFoundError:
            mlx_version = "unavailable"

        result: dict[str, Any] = {
            "version": mlx_version,
        }
        api_names: list[str] = []

        if hasattr(mx, "get_peak_memory"):
            api_names.append("mx.get_peak_memory")
            peak_bytes = int(mx.get_peak_memory())
            result["peak_bytes"] = peak_bytes
            result["peak_mib"] = round(peak_bytes / (1024 * 1024), 2)
        if hasattr(mx, "get_active_memory"):
            api_names.append("mx.get_active_memory")
            current_bytes = int(mx.get_active_memory())
            result["current_bytes"] = current_bytes
            result["current_mib"] = round(current_bytes / (1024 * 1024), 2)
        if hasattr(mx, "get_cache_memory"):
            api_names.append("mx.get_cache_memory")
            cache_bytes = int(mx.get_cache_memory())
            result["cache_bytes"] = cache_bytes
            result["cache_mib"] = round(cache_bytes / (1024 * 1024), 2)

        if not api_names:
            return None
        result["api"] = api_names
        return result
    except ImportError:
        return None


def reset_mlx_allocator() -> None:
    """Reset MLX peak accounting without clearing allocator caches."""
    try:
        import mlx.core as mx

        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Result sanitization
# ---------------------------------------------------------------------------


def sanitize_result(
    snapshot_dir: Path,
    config_dict: dict[str, Any],
    prompt_text: str,
    formatted_text: str,
    token_ids: list[int],
    backend: str,
    backend_info: dict[str, str],
    versions: dict[str, str],
    optional_statuses: dict[str, Any],
    manifest: list[dict[str, Any]],
    ref_result: dict[str, Any],
    pt_result: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    pub_thresholds: dict[str, Any] | None,
    timings: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, Any]:
    """Build a sanitized JSON result report.

    Parameters
    ----------
    snapshot_dir : Path
        Snapshot directory (only basename used).
    config_dict : dict
        Parsed config.json.
    prompt_text : str
        User prompt.
    formatted_text : str
        Chat-template-rendered prompt.
    token_ids : list[int]
        Prompt token IDs.
    backend : str
        Backend name.
    backend_info : dict
        Backend linker/mode info.
    versions : dict
        Package versions.
    optional_statuses : dict
        JAX/PyTensor-MLX statuses.
    manifest : list[dict]
        File manifest.
    ref_result : dict
        MLX-LM reference result.
    pt_result : dict or None
        PyTensor forward result (None if skipped).
    metrics : dict or None
        Comparison metrics (None if skipped).
    pub_thresholds : dict or None
        Publication threshold results.
    timings : dict
        Timing breakdown.
    memory : dict
        Memory metrics.

    Returns
    -------
    dict
        Sanitized report dictionary.
    """
    revision = detect_revision(snapshot_dir)

    report: dict[str, Any] = {
        "model": {
            "repo": EXPECTED_REPO,
            "revision": revision,
            "model_type": EXPECTED_MODEL_TYPE,
            "architecture": EXPECTED_ARCHITECTURE,
            "quantization": {
                "bits": EXPECTED_BITS,
                "group_size": EXPECTED_GROUP_SIZE,
            },
        },
        "prompt": {
            "text": prompt_text,
            "formatted": formatted_text,
            "token_ids": token_ids,
            "n_tokens": len(token_ids),
            "token_hash": hash_token_ids(token_ids),
        },
        "backend": backend_info,
        "versions": versions,
        "optional_statuses": optional_statuses,
        "device": get_device(),
        "file_manifest": manifest,
        "reference": {
            "runtime": "mlx_lm_native",
            "vocab_size": ref_result["vocab_size"],
            "seq_len": ref_result["seq_len"],
            "load_s": round(ref_result["load_s"], 3),
            "forward_s": round(ref_result["forward_s"], 3),
            "sync_s": round(ref_result["sync_s"], 3),
            "peak_memory_mib": ref_result["peak_memory_mib"],
            "logits_sha256": hashlib.sha256(
                np.asarray(ref_result["logits"], dtype="<f4").tobytes()
            ).hexdigest(),
            "artifact": ref_result.get("artifact"),
        },
        "timing": timings,
        "memory": memory,
    }

    if pt_result is not None:
        report["pytensor"] = {
            "layers_completed": pt_result["layers_completed"],
            "embed_s": round(pt_result["embed_s"], 3),
            "ple_s": round(pt_result.get("ple_s", 0.0), 3),
            "global_load_s": round(pt_result.get("global_load_s", 0.0), 3),
            "initial_s": round(pt_result.get("initial_s", 0.0), 3),
            "per_layer_proj_s": round(pt_result.get("per_layer_proj_s", 0.0), 3),
            "per_layer_s": [round(t, 4) for t in pt_result["per_layer_s"]],
            "final_s": round(pt_result.get("final_s", 0.0), 3),
            "logits_s": round(pt_result["logits_s"], 3),
            "total_s": round(pt_result["total_s"], 3),
            "layer_types_used": pt_result.get("layer_types_used", []),
            "rope_bases_used": pt_result.get("rope_bases_used", []),
            "sparse_layers_used": pt_result.get("sparse_layers_used", []),
            "chunks_processed": pt_result.get("chunks_processed", 0),
            "mlx_eval_s": round(pt_result.get("mlx_eval_s", 0.0), 4),
            "mlx_host_copy_s": round(pt_result.get("mlx_host_copy_s", 0.0), 4),
            "mlx_stages": pt_result.get("mlx_stages", []),
            "stage_count": pt_result.get("stage_count", 0),
            "logits_sha256": hashlib.sha256(
                np.asarray(pt_result["logits"], dtype="<f4").tobytes()
            ).hexdigest(),
            "artifact": pt_result.get("artifact"),
        }

    if metrics is not None:
        report["metrics"] = _round_metrics_for_report(metrics)

    if pub_thresholds is not None:
        report["publication_thresholds"] = _round_thresholds_for_report(
            pub_thresholds
        )

    return report


def _round_metrics_for_report(metrics: dict[str, Any]) -> dict[str, Any]:
    """Round metric floats for JSON without changing validation precision."""
    report = dict(metrics)
    report["per_position"] = []
    for position in metrics.get("per_position", []):
        rounded = {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in position.items()
        }
        report["per_position"].append(rounded)
    report["aggregate"] = dict(metrics.get("aggregate", {}))
    return report


def _round_thresholds_for_report(
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Copy threshold results without obscuring boundary-side precision."""
    report = dict(thresholds)
    report["checks"] = [dict(check) for check in thresholds.get("checks", [])]
    return report


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def atomic_write_json(data: dict[str, Any], dest: Path) -> None:
    """Write JSON atomically via the shared evidence writer.

    Delegates to :func:`cetagostini.utils.pytensor.evidence.atomic_write_json`
    for consistent atomic write semantics across all runner modules.

    Uses ``allow_nan=False`` to reject any NaN/Inf values that would
    produce invalid JSON.

    Parameters
    ----------
    data : dict
        Data to serialize.
    dest : Path
        Destination file path.

    Raises
    ------
    ValueError
        If ``data`` contains non-finite float values.
    """
    from cetagostini.utils.pytensor.evidence import (
        atomic_write_json as _evidence_atomic_write_json,
    )

    _evidence_atomic_write_json(data, Path(dest))


# ---------------------------------------------------------------------------
# Probe mode
# ---------------------------------------------------------------------------


def run_probe(snapshot_dir: Path | None) -> dict[str, Any]:
    """Run environment probe and return statuses.

    Parameters
    ----------
    snapshot_dir : Path or None
        Optional snapshot directory to validate.

    Returns
    -------
    dict
        Probe results.
    """
    result: dict[str, Any] = {
        "versions": collect_versions(),
        "optional_statuses": check_optional_statuses(),
        "device": get_device(),
        "peak_rss_mib": get_peak_rss_mib(),
        "valid_backends": list(VALID_BACKENDS),
        "publication_thresholds": {
            "cosine_min": PUB_COSINE_MIN,
            "pearson_min": PUB_PEARSON_MIN,
            "all_top1_match": PUB_ALL_TOP1_MATCH,
            "top10_overlap_mean_min": PUB_TOP10_OVERLAP_MEAN_MIN,
        },
    }

    if snapshot_dir is not None:
        try:
            config = validate_snapshot(snapshot_dir)
            result["snapshot"] = {
                "valid": True,
                "revision": detect_revision(snapshot_dir),
                "model_type": config.get("model_type"),
                "architecture": config.get("architectures", []),
            }
        except (FileNotFoundError, ValueError) as exc:
            result["snapshot"] = {
                "valid": False,
                "error": str(exc),
            }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_evidence_provenance(
    run_id: str,
    command: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the shared implementation manifest and bound run provenance."""
    repo_root = find_repo_root(Path(__file__))
    implementation = build_implementation_manifest(
        repo_root=repo_root,
        source_files=GEMMA3N_IMPLEMENTATION_SOURCE_FILES,
        require_clean=True,
        packages=GEMMA3N_PROVENANCE_PACKAGES,
        modules=GEMMA3N_PROVENANCE_MODULES,
        environment_yml_path=GEMMA3N_ENVIRONMENT_YML,
    )
    provenance = build_provenance_report(
        run_id=run_id,
        schema_version=GEMMA3N_ORACLE_SCHEMA_VERSION,
        implementation_manifest=implementation,
        command=command,
    )
    return implementation, provenance


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    The ``run`` subcommand consumes a pre-computed standalone oracle
    (report JSON + logits NPY) and runs only the backend forward pass.
    The backend never invokes the MLX-LM oracle.

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

    # Enforce offline mode before any HF/transformers import.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if args.command == "probe":
        result = run_probe(args.snapshot)
        print(json.dumps(result, indent=2, ensure_ascii=True))
        if args.snapshot is not None and not result["snapshot"]["valid"]:
            return 1
        return 0

    # ── run mode ─────────────────────────────────────────────────────
    snapshot_dir = args.snapshot.resolve()
    backend = args.backend
    run_id = args.run_id

    # Phase 1: Validate snapshot
    try:
        config_dict = validate_snapshot(snapshot_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Snapshot validated: {EXPECTED_ARCHITECTURE}, "
          f"{EXPECTED_BITS}-bit, group_size={EXPECTED_GROUP_SIZE}")

    manifest = build_file_manifest(snapshot_dir)
    versions = collect_versions()
    optional_statuses = check_optional_statuses()
    backend_info = get_backend_info(backend)

    # Phase 2: Tokenize (tokenizer-only, no model weights)
    print("Loading tokenizer from snapshot (tokenizer-only)...")
    t_tok_start = time.time()
    tokenizer = load_tokenizer_from_snapshot(snapshot_dir)
    formatted_text, token_ids = format_and_tokenize(tokenizer, args.prompt)
    try:
        validate_publication_prompt_tokens(args.prompt, token_ids)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    t_tokenize = time.time() - t_tok_start
    T = len(token_ids)
    print(f"  {T} tokens, formatted in {t_tokenize:.2f}s")
    print(f"  token_hash: {hash_token_ids(token_ids)}")

    # Bind the backend run to the same clean implementation as the oracle.
    command = list(sys.argv) if argv is None else [__name__, *argv]
    try:
        implementation_manifest, provenance = _build_evidence_provenance(
            run_id,
            command,
        )
    except RuntimeError as exc:
        print(f"ERROR: provenance failed: {exc}", file=sys.stderr)
        return 1

    # Phase 3: Load and verify standalone oracle artifacts
    print("Loading and verifying standalone oracle artifacts...")
    try:
        ref_report = load_and_verify_reference_report(
            args.reference_report,
            run_id=run_id,
            snapshot_dir=snapshot_dir,
            config_dict=config_dict,
            manifest=manifest,
            prompt_text=args.prompt,
            formatted_text=formatted_text,
            token_ids=token_ids,
            implementation_manifest=implementation_manifest,
        )
        print(f"  Reference report verified (repo={ref_report['model']['repo']})")

        ref_logits = load_and_verify_reference_logits(
            args.reference_logits,
            ref_report,
            expected_seq_len=T,
        )
        print(f"  Reference logits verified: shape={ref_logits.shape}, "
              f"dtype={ref_logits.dtype}")
    except OracleVerificationError as exc:
        print(f"ERROR: Oracle verification failed: {exc}", file=sys.stderr)
        return 1

    # Build a ref_result dict compatible with sanitize_result
    ref_result = {
        "logits": ref_logits,
        "load_s": ref_report.get("timing", {}).get("ref_load_s", 0.0),
        "forward_s": ref_report.get("timing", {}).get("ref_forward_s", 0.0),
        "sync_s": ref_report.get("timing", {}).get("ref_sync_s", 0.0),
        "peak_memory_mib": (
            ref_report.get("memory", {})
            .get("oracle_mlx", {})
            .get("peak_mib")
        ),
        "vocab_size": ref_report["reference"]["vocab_size"],
        "seq_len": ref_report["reference"]["seq_len"],
        "artifact": ref_report["raw_artifact"],
    }

    # Phase 4: Backend forward pass (no MLX-LM oracle)
    pt_result = None
    metrics = None
    pub_thresholds = None
    t_compile = 0.0
    t_pt_total = 0.0

    mlx_baseline_bytes: int | None = None

    print(f"Running backend forward pass (backend={backend})...")
    try:
        t_w_start = time.time()
        loader = _create_weight_loader(snapshot_dir)
        try:
            text_config = loader.config
            t_weights_load = time.time() - t_w_start
            print(f"  weight loader created in {t_weights_load:.2f}s")

            pt_config = _build_pytensor_config(text_config)

            t_c_start = time.time()
            compiled = _compile_graphs(pt_config, T, backend, text_config)
            t_compile = time.time() - t_c_start
            print(f"  compiled in {t_compile:.2f}s")

            # Isolate allocator accounting to the fully synchronized forward.
            if backend == "mlx":
                reset_mlx_allocator()
                baseline_snapshot = get_mlx_memory_snapshot()
                if baseline_snapshot is not None:
                    mlx_baseline_bytes = baseline_snapshot.get("current_bytes")

            pt_result = run_pytensor_forward(
                loader, compiled, token_ids, text_config, pt_config, backend,
            )
            t_pt_total = pt_result["total_s"]
            print(f"  {pt_result['layers_completed']} layers in {t_pt_total:.2f}s")
            print(f"  chunks_processed: {pt_result['chunks_processed']}")
            print(f"  stage_count: {pt_result['stage_count']}")
            print(f"  sparse_layers: {len(pt_result['sparse_layers_used'])}")
            del compiled
        finally:
            loader.close()
            gc.collect()

    except ImportError as exc:
        print(f"  ERROR (sibling module not available): {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"  ERROR during backend forward: {exc}", file=sys.stderr)
        return 1

    # Phase 5: Compute metrics
    if pt_result is not None:
        print("Computing all-position comparison metrics...")
        pt_logits = pt_result["logits"]

        if ref_logits.shape != pt_logits.shape:
            print(
                f"  WARNING: shape mismatch: ref={ref_logits.shape}, "
                f"pt={pt_logits.shape}",
                file=sys.stderr,
            )
        else:
            metrics = compute_all_position_metrics(ref_logits, pt_logits)
            if metrics["final_top1_ref"] is not None:
                metrics["final_top1_ref_text"] = decode_single_token(
                    tokenizer, metrics["final_top1_ref"]
                )
                metrics["final_top1_pt_text"] = decode_single_token(
                    tokenizer, metrics["final_top1_pt"]
                )
            else:
                metrics["final_top1_ref_text"] = None
                metrics["final_top1_pt_text"] = None
            pub_thresholds = check_publication_thresholds(metrics)
            agg = metrics.get("aggregate", {})
            print(f"  all_finite: ref={metrics['all_finite_ref']}, "
                  f"pt={metrics['all_finite_pt']}")
            print(f"  final top1 match: {metrics['final_top1_match']} "
                  f"(ref={metrics['final_top1_ref']}, "
                  f"pt={metrics['final_top1_pt']})")
            print(
                "  decoded next token: "
                f"{metrics['final_top1_pt_text']!r}"
            )
            if agg:
                print(f"  cosine_mean: {agg.get('cosine_mean', 'N/A')}")
                print(f"  pearson_mean: {agg.get('pearson_mean', 'N/A')}")
                print(f"  top10_overlap_mean: {agg.get('top10_overlap_mean', 'N/A')}")
            print(f"  publication thresholds: "
                  f"{'PASSED' if pub_thresholds['passed'] else 'FAILED'}")
            for check in pub_thresholds["checks"]:
                status = "PASS" if check["passed"] else "FAIL"
                print(f"    [{status}] {check['name']}: "
                      f"{check['actual']} (threshold: {check['threshold']})")

    # Phase 6: Build timings & memory
    timings: dict[str, Any] = {
        "tokenize_s": round(t_tokenize, 3),
        "ref_load_s": round(ref_result["load_s"], 3),
        "ref_forward_s": round(ref_result["forward_s"], 3),
        "ref_sync_s": round(ref_result["sync_s"], 3),
        "pt_compile_s": round(t_compile, 3),
        "pt_total_s": round(t_pt_total, 3),
    }
    if pt_result is not None:
        timings["pt_embed_s"] = round(pt_result["embed_s"], 3)
        timings["pt_ple_s"] = round(pt_result.get("ple_s", 0.0), 3)
        timings["pt_initial_s"] = round(pt_result.get("initial_s", 0.0), 3)
        timings["pt_per_layer_proj_s"] = round(pt_result.get("per_layer_proj_s", 0.0), 3)
        timings["pt_per_layer_s"] = [round(t, 4) for t in pt_result["per_layer_s"]]
        timings["pt_final_s"] = round(pt_result.get("final_s", 0.0), 3)
        timings["pt_logits_s"] = round(pt_result["logits_s"], 3)

    memory: dict[str, Any] = {
        "whole_process_peak_rss_mib": get_peak_rss_mib(),
    }

    # MLX-only memory section (after final sync)
    if backend == "mlx":
        mlx_mem = get_mlx_memory_snapshot()
        if mlx_mem is not None:
            mlx_mem["baseline_bytes"] = mlx_baseline_bytes
            mlx_mem["baseline_mib"] = (
                None
                if mlx_baseline_bytes is None
                else round(mlx_baseline_bytes / (1024 * 1024), 2)
            )
            memory["backend_mlx"] = mlx_mem

    # Persist backend logits before publishing the report so the validator can
    # independently recompute every metric and threshold decision.
    if pt_result is None or metrics is None:
        return 1
    if not metrics.get("all_finite_pt", False):
        print("ERROR: backend logits contain non-finite values", file=sys.stderr)
        return 1
    backend_logits = np.ascontiguousarray(
        pt_result["logits"],
        dtype=np.dtype("<f4"),
    )
    atomic_write_npy(backend_logits, args.logits_output)
    backend_artifact = build_npy_manifest(args.logits_output)
    try:
        verify_npy_artifact(args.logits_output, backend_artifact)
    except ArtifactVerificationError as exc:
        print(f"ERROR: backend logits verification failed: {exc}", file=sys.stderr)
        return 1
    pt_result["artifact"] = backend_artifact

    # Phase 7: Build report
    report = sanitize_result(
        snapshot_dir=snapshot_dir,
        config_dict=config_dict,
        prompt_text=args.prompt,
        formatted_text=formatted_text,
        token_ids=token_ids,
        backend=backend,
        backend_info=backend_info,
        versions=versions,
        optional_statuses=optional_statuses,
        manifest=manifest,
        ref_result=ref_result,
        pt_result=pt_result,
        metrics=metrics,
        pub_thresholds=pub_thresholds,
        timings=timings,
        memory=memory,
    )

    report["schema_version"] = GEMMA3N_ORACLE_SCHEMA_VERSION
    report["run_id"] = run_id
    report["provenance"] = provenance
    report["command"] = provenance["command"]

    # Phase 8: Write output
    if args.output:
        atomic_write_json(report, args.output)
        print(f"Report written to {args.output}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=True))

    if pt_result is None or pub_thresholds is None:
        return 1
    return 0 if pub_thresholds["passed"] else 1


# ---------------------------------------------------------------------------
# Cache-free generation
# ---------------------------------------------------------------------------


def run_cache_free_generation(
    loader: Any,
    tokenizer: Any,
    prompt_token_ids: list[int],
    first_token_id: int,
    max_tokens: int,
    stop_token_ids: frozenset[int],
    text_config: Any,
    pt_config: Any,
    backend: str,
    *,
    first_compile_s: float,
    first_forward_s: float,
    layer_weight_cache: Sequence[dict[str, np.ndarray]] | None,
    reference_forward: Callable[[Sequence[int]], dict[str, Any]] | None,
    first_reference_metrics: dict[str, Any] | None,
    first_reference_thresholds: dict[str, Any] | None,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    """Generate greedily by recompiling and re-evaluating each growing prefix.

    The initial token is supplied by the all-position validated prompt pass.
    Every later token is produced by a fresh PyTensor graph specialized to the
    prompt plus all previously generated tokens. No KV state is retained.

    Parameters
    ----------
    loader : Gemma3nWeightLoader
        Weight loader with mmap open.
    tokenizer : Any
        mlx_lm tokenizer wrapper.
    prompt_token_ids : list[int]
        Prompt token IDs.
    first_token_id : int
        First generated token from the validated prompt pass.
    max_tokens : int
        Maximum number of visible greedy tokens.
    stop_token_ids : frozenset[int]
        Token IDs that signal end-of-generation.
    text_config : Gemma3nTextConfig
        From gemma3n_weights.
    pt_config : Gemma3nConfig
        For gemma3n_pytensor.
    backend : str
        ``'c'`` or ``'numba'``.
    first_compile_s : float
        Compilation time from the first prompt pass.
    first_forward_s : float
        Forward pass time from the first prompt pass.
    layer_weight_cache : Sequence[dict[str, np.ndarray]] or None
        Optional cached expanded FP32 decoder weights.
    reference_forward : callable or None
        Optional MLX-LM reference forward pass for differential validation.
    first_reference_metrics : dict or None
        Metrics from the first prompt pass reference comparison.
    first_reference_thresholds : dict or None
        Publication thresholds from the first prompt pass.
    progress : callable or None
        Optional callback receiving progress messages.

    Returns
    -------
    dict
        Keys: ``generated_ids``, ``text``, ``stop_reason``, ``stop_token_id``,
        ``steps``, ``total_compile_s``, ``total_forward_s``,
        ``total_generation_wall_s``, ``reference_forward_s``,
        ``reference_sync_s``, ``all_reference_tokens_match``.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")

    generated_ids: list[int] = []
    steps: list[dict[str, Any]] = []
    current_ids = list(prompt_token_ids)
    stop_reason = "max_tokens"
    stop_token_id = None
    total_compile_s = first_compile_s
    total_forward_s = first_forward_s
    total_generation_wall_s = 0.0
    reference_forward_s = 0.0
    reference_sync_s = 0.0
    all_reference_tokens_match = True

    # Record the first step (from the validated prompt pass)
    first_step: dict[str, Any] = {
        "step": 1,
        "prefix_tokens": len(prompt_token_ids),
        "token_id": first_token_id,
        "compile_s": first_compile_s,
        "forward_s": first_forward_s,
        "validated_against_reference": first_reference_metrics is not None,
    }
    if first_reference_metrics is not None:
        first_step["reference_cache_mode"] = "initial_shared_prefill"
        first_step["reference_token_id"] = first_reference_metrics.get("final_top1_ref")
        first_step["reference_match"] = first_reference_metrics.get("final_top1_match")
        first_step["reference_forward_s"] = 0.0
        first_step["reference_sync_s"] = 0.0
        first_step["reference_metrics"] = first_reference_metrics
    steps.append(first_step)

    # Check if first token is a stop token
    if first_token_id in stop_token_ids:
        stop_reason = "stop_token"
        stop_token_id = first_token_id
        return {
            "generated_ids": generated_ids,
            "text": "",
            "stop_reason": stop_reason,
            "stop_token_ids": sorted(stop_token_ids),
            "stop_token_id": stop_token_id,
            "steps": steps,
            "total_compile_s": total_compile_s,
            "total_forward_s": total_forward_s,
            "total_generation_wall_s": total_generation_wall_s,
            "reference_forward_s": reference_forward_s,
            "reference_sync_s": reference_sync_s,
            "all_reference_tokens_match": all_reference_tokens_match,
        }

    generated_ids.append(first_token_id)
    current_ids.append(first_token_id)

    # Generate remaining tokens
    for step_idx in range(1, max_tokens):
        if progress:
            progress(f"Generating token {step_idx + 1}/{max_tokens} (prefix length {len(current_ids)})")

        t_step_start = time.time()

        # Compile fresh graph for this prefix length
        t_compile_start = time.time()
        compiled = _compile_graphs(pt_config, len(current_ids), backend, text_config)
        t_compile = time.time() - t_compile_start
        total_compile_s += t_compile

        # Run forward pass
        t_forward_start = time.time()
        pt_result = run_pytensor_forward(
            loader,
            compiled,
            current_ids,
            text_config,
            pt_config,
            backend,
            layer_weight_cache=layer_weight_cache,
        )
        t_forward = time.time() - t_forward_start
        total_forward_s += t_forward

        # Extract next token
        logits = pt_result["logits"][0, -1]  # Last position logits
        next_token_id = int(np.argmax(logits))

        # Optional reference validation
        ref_metrics = None
        ref_token_id = None
        ref_match = None
        if reference_forward is not None:
            t_ref_start = time.time()
            ref_result = reference_forward(current_ids)
            t_ref_forward = time.time() - t_ref_start
            reference_forward_s += t_ref_forward

            t_ref_sync_start = time.time()
            import mlx.core as mx
            mx.eval(ref_result["logits"])
            t_ref_sync = time.time() - t_ref_sync_start
            reference_sync_s += t_ref_sync

            ref_metrics = compute_all_position_metrics(
                ref_result["logits"], pt_result["logits"]
            )
            ref_token_id = ref_metrics.get("final_top1_ref")
            ref_match = ref_metrics.get("final_top1_match")
            if not ref_match:
                all_reference_tokens_match = False

        t_step = time.time() - t_step_start
        total_generation_wall_s += t_step

        step_record: dict[str, Any] = {
            "step": step_idx + 1,
            "prefix_tokens": len(current_ids),
            "token_id": next_token_id,
            "compile_s": t_compile,
            "forward_s": t_forward,
            "validated_against_reference": ref_metrics is not None,
        }
        if ref_metrics is not None:
            step_record["reference_cache_mode"] = "fresh_shared_prefix"
            step_record["reference_token_id"] = ref_token_id
            step_record["reference_match"] = ref_match
            step_record["reference_forward_s"] = t_ref_forward
            step_record["reference_sync_s"] = t_ref_sync
            step_record["reference_metrics"] = ref_metrics
        steps.append(step_record)

        # Check for stop token
        if next_token_id in stop_token_ids:
            stop_reason = "stop_token"
            stop_token_id = next_token_id
            break

        generated_ids.append(next_token_id)
        current_ids.append(next_token_id)

    # Decode generated text
    text = decode_token_ids(tokenizer, generated_ids)

    return {
        "generated_ids": generated_ids,
        "text": text,
        "stop_reason": stop_reason,
        "stop_token_ids": sorted(stop_token_ids),
        "stop_token_id": stop_token_id,
        "steps": steps,
        "total_compile_s": total_compile_s,
        "total_forward_s": total_forward_s,
        "total_generation_wall_s": total_generation_wall_s,
        "reference_forward_s": reference_forward_s,
        "reference_sync_s": reference_sync_s,
        "all_reference_tokens_match": all_reference_tokens_match,
    }


# ---------------------------------------------------------------------------
# Reusable inference entry point
# ---------------------------------------------------------------------------


def run_inference(
    snapshot_path: str | Path,
    *,
    prompt: str,
    backend: str,
    reference_only: bool,
    max_tokens: int,
    cache_layer_weights: bool,
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run cache-free Gemma 3n autoregressive inference from Python.

    Parameters
    ----------
    snapshot_path : str or Path
        Pinned local Hugging Face snapshot.
    prompt : str
        User prompt passed through Gemma's chat template.
    backend : str
        PyTensor linker, either ``"c"`` or ``"numba"``.
    reference_only : bool
        Skip the PyTensor path and retain only the MLX-LM oracle report.
    max_tokens : int
        Maximum number of visible greedy tokens. Every token after the first
        re-evaluates the complete growing prefix without a KV cache.
    cache_layer_weights : bool
        Retain expanded FP32 decoder weights across generation steps. This is
        faster but requires tens of GiB of memory.
    output_path : str, Path, or None
        Atomically persist the sanitized report here before returning.
    progress : callable or None
        Optional callback receiving progress messages.

    Returns
    -------
    dict
        Sanitized all-position inference and validation report.
    """
    snapshot_dir = Path(snapshot_path).resolve()

    # Enforce offline mode before any HF/transformers import.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if progress:
        progress("Validating snapshot")

    # Phase 1: Validate snapshot
    config_dict = validate_snapshot(snapshot_dir)
    manifest = build_file_manifest(snapshot_dir)
    versions = collect_versions()
    optional_statuses = check_optional_statuses()
    backend_info = get_backend_info(backend)

    # Phase 2: Tokenize
    if progress:
        progress("Loading tokenizer")
    t_tok_start = time.time()
    tokenizer = load_tokenizer_from_snapshot(snapshot_dir)
    formatted_text, token_ids = format_and_tokenize(tokenizer, prompt)
    t_tokenize = time.time() - t_tok_start
    T = len(token_ids)

    # Phase 3: MLX-LM reference forward pass
    if progress:
        progress("Running MLX-LM reference")
    ref_result = run_mlx_reference(snapshot_dir, token_ids)

    # Phase 4: PyTensor forward pass (skipped if --reference-only)
    pt_result = None
    metrics = None
    pub_thresholds = None
    t_compile = 0.0
    t_pt_total = 0.0
    generation_result = None

    if not reference_only:
        if progress:
            progress(f"Running PyTensor forward (backend={backend})")

        # Load weights
        t_w_start = time.time()
        loader = _create_weight_loader(snapshot_dir)
        try:
            text_config = loader.config
            t_weights_load = time.time() - t_w_start

            pt_config = _build_pytensor_config(text_config)

            # Compile and run first pass
            t_c_start = time.time()
            compiled = _compile_graphs(pt_config, T, backend, text_config)
            t_compile = time.time() - t_c_start

            pt_result = run_pytensor_forward(
                loader,
                compiled,
                token_ids,
                text_config,
                pt_config,
                backend,
                capture_layer_weights=cache_layer_weights,
            )
            t_pt_total = pt_result["total_s"]
            layer_weight_cache = pt_result.pop("_layer_weight_cache", None)

            # Compute metrics for first pass
            ref_logits = ref_result["logits"]
            pt_logits = pt_result["logits"]
            if ref_logits.shape == pt_logits.shape:
                metrics = compute_all_position_metrics(ref_logits, pt_logits)
                if metrics["final_top1_ref"] is not None:
                    metrics["final_top1_ref_text"] = decode_single_token(
                        tokenizer, metrics["final_top1_ref"]
                    )
                    metrics["final_top1_pt_text"] = decode_single_token(
                        tokenizer, metrics["final_top1_pt"]
                    )
                pub_thresholds = check_publication_thresholds(metrics)

            # Generate additional tokens if max_tokens > 1
            if max_tokens > 1 and metrics is not None:
                first_token_id = metrics["final_top1_pt"]
                if first_token_id is not None:
                    stop_token_ids = get_stop_token_ids(tokenizer)

                    # Build reference forward callable
                    def reference_forward(ids: Sequence[int]) -> dict[str, Any]:
                        return run_mlx_reference(snapshot_dir, list(ids))

                    if progress:
                        progress(f"Generating {max_tokens - 1} additional tokens")

                    generation_result = run_cache_free_generation(
                        loader=loader,
                        tokenizer=tokenizer,
                        prompt_token_ids=token_ids,
                        first_token_id=first_token_id,
                        max_tokens=max_tokens,
                        stop_token_ids=stop_token_ids,
                        text_config=text_config,
                        pt_config=pt_config,
                        backend=backend,
                        first_compile_s=t_compile,
                        first_forward_s=t_pt_total,
                        layer_weight_cache=layer_weight_cache,
                        reference_forward=reference_forward,
                        first_reference_metrics=metrics,
                        first_reference_thresholds=pub_thresholds,
                        progress=progress,
                    )

            del compiled
        finally:
            loader.close()
            gc.collect()

    # Phase 5: Build timings & memory
    timings: dict[str, Any] = {
        "tokenize_s": round(t_tokenize, 3),
        "ref_load_s": round(ref_result["load_s"], 3),
        "ref_forward_s": round(ref_result["forward_s"], 3),
        "ref_sync_s": round(ref_result["sync_s"], 3),
        "ref_generation_s": 0.0,
        "pt_compile_s": round(t_compile, 3),
        "pt_total_s": round(t_pt_total, 3),
    }
    if pt_result is not None:
        timings["pt_embed_s"] = round(pt_result["embed_s"], 3)
        timings["pt_ple_s"] = round(pt_result.get("ple_s", 0.0), 3)
        timings["pt_initial_s"] = round(pt_result.get("initial_s", 0.0), 3)
        timings["pt_per_layer_proj_s"] = round(pt_result.get("per_layer_proj_s", 0.0), 3)
        timings["pt_per_layer_s"] = [round(t, 4) for t in pt_result["per_layer_s"]]
        timings["pt_final_s"] = round(pt_result.get("final_s", 0.0), 3)
        timings["pt_logits_s"] = round(pt_result["logits_s"], 3)

    if generation_result is not None:
        timings["generation_additional_wall_s"] = round(
            generation_result["total_generation_wall_s"], 3
        )
        timings["generation_reference_forward_s"] = round(
            generation_result["reference_forward_s"], 3
        )
        timings["generation_reference_sync_s"] = round(
            generation_result["reference_sync_s"], 3
        )

    memory: dict[str, Any] = {
        "peak_rss_mib": get_peak_rss_mib(),
        "mlx_peak_memory_mib": ref_result["peak_memory_mib"],
    }

    # Phase 6: Build report
    report = sanitize_result(
        snapshot_dir=snapshot_dir,
        config_dict=config_dict,
        prompt_text=prompt,
        formatted_text=formatted_text,
        token_ids=token_ids,
        backend=backend,
        backend_info=backend_info,
        versions=versions,
        optional_statuses=optional_statuses,
        manifest=manifest,
        ref_result=ref_result,
        pt_result=pt_result,
        metrics=metrics,
        pub_thresholds=pub_thresholds,
        timings=timings,
        memory=memory,
    )

    # Add generation section if present
    if generation_result is not None:
        report["generation"] = {
            "mode": "cache_free_full_prefix",
            "max_tokens": max_tokens,
            "generated_ids": generation_result["generated_ids"],
            "generated_ids_sha256": hashlib.sha256(
                np.array(generation_result["generated_ids"], dtype=np.int32).tobytes()
            ).hexdigest(),
            "text": generation_result["text"],
            "n_tokens": len(generation_result["generated_ids"]),
            "stop_reason": generation_result["stop_reason"],
            "stop_token_ids": generation_result["stop_token_ids"],
            "stop_token_id": generation_result["stop_token_id"],
            "steps": generation_result["steps"],
            "validation_scope": "all_generated_prefixes_fresh_cache",
            "all_reference_tokens_match": generation_result["all_reference_tokens_match"],
            "n_prefix_graph_compilations": len(generation_result["steps"]),
            "additional_wall_s": generation_result["total_generation_wall_s"],
            "model_compile_s": generation_result["total_compile_s"],
            "model_forward_s": generation_result["total_forward_s"],
            "reference_forward_s": generation_result["reference_forward_s"],
            "reference_sync_s": generation_result["reference_sync_s"],
            "weight_mode": (
                "cached_fp32" if cache_layer_weights else "streamed_affine4"
            ),
        }

    # Phase 7: Write output
    if output_path is not None:
        atomic_write_json(report, Path(output_path))

    return report


if __name__ == "__main__":
    sys.exit(main())
