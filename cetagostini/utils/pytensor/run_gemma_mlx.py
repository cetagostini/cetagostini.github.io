#!/usr/bin/env python3
"""Run Gemma 3n E4B-it via native MLX-LM (reference oracle).

Standalone MLX-LM reference runner for Gemma 3n.  Loads a pinned local HF
snapshot, tokenizes a prompt via the snapshot's chat template, and runs a
direct MLX-LM forward pass with ``cache=None`` to obtain all-position
float32 reference logits.

This module serves as the oracle baseline for PyTensor comparisons and can
be used independently for reference-only evaluations.  When ``--logits-output``
is provided, it persists the full ``[1, T, vocab]`` logits as a C-contiguous
little-endian float32 ``.npy`` file and builds a verified artifact manifest
before atomically writing the JSON report.

Usage::

    python -m cetagostini.utils.pytensor.run_gemma_mlx --run-id run-001 --snapshot /path/to/snapshot
    python -m cetagostini.utils.pytensor.run_gemma_mlx --run-id run-001 --snapshot /path/to/snapshot --output results/gemma_mlx.json
    python -m cetagostini.utils.pytensor.run_gemma_mlx --run-id run-001 --snapshot /path/to/snapshot --logits-output logits.npy --output results/gemma_mlx.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# Reuse constants and helpers from the PyTensor runner
from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
    DEFAULT_PROMPT,
    EXPECTED_ARCHITECTURE,
    EXPECTED_BITS,
    EXPECTED_GROUP_SIZE,
    EXPECTED_MANIFEST,
    EXPECTED_MODEL_TYPE,
    EXPECTED_REPO,
    EXPECTED_REVISION,
    REQUIRED_FILES,
    _sha256_file,
    build_file_manifest,
    check_optional_statuses,
    collect_versions,
    decode_single_token,
    detect_revision,
    format_and_tokenize,
    get_device,
    get_mlx_peak_memory_mib,
    get_peak_rss_mib,
    hash_token_ids,
    load_tokenizer_from_snapshot,
    run_mlx_reference,
    validate_publication_prompt_tokens,
    validate_snapshot,
)

# Reuse evidence and provenance utilities
from cetagostini.utils.pytensor.evidence import (
    GEMMA3N_ORACLE_SCHEMA_VERSION,
    atomic_write_json,
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
# Repo root detection
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:
    """Return the Git worktree root containing this module."""
    return find_repo_root(Path(__file__))


# ---------------------------------------------------------------------------
# Tokenizer-only loading
# ---------------------------------------------------------------------------


def _load_tokenizer_only(snapshot_dir: Path) -> Any:
    """Load only the tokenizer without loading model weights when possible.

    Uses MLX-LM's public tokenizer-only loader. Falls back to Transformers'
    tokenizer-only loader when the pinned MLX-LM helper is unavailable.

    Parameters
    ----------
    snapshot_dir : Path
        Root directory of the local snapshot.

    Returns
    -------
    tokenizer
        The mlx_lm tokenizer wrapper.
    """
    try:
        from mlx_lm.utils import load_tokenizer
        return load_tokenizer(str(snapshot_dir))
    except (ImportError, AttributeError):
        return load_tokenizer_from_snapshot(snapshot_dir)


# ---------------------------------------------------------------------------
# Oracle forward pass
# ---------------------------------------------------------------------------


def run_oracle_forward(
    snapshot_dir: Path,
    token_ids: list[int],
) -> dict[str, Any]:
    """Run a direct MLX-LM forward pass for the oracle.

    Loads the model once, resets MLX peak memory immediately before the
    forward pass, runs the forward with ``cache=None``, evaluates and
    copies the output to an owning C-contiguous ``<f4`` array, then
    reads memory metrics.

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
        ``vocab_size``, ``seq_len``,
        ``mlx_api``, ``mlx_version``,
        ``mlx_baseline_mib``, ``mlx_peak_mib``, ``mlx_current_mib``.
    """
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    # Detect MLX core version
    try:
        mlx_version_core = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        mlx_version_core = "unavailable"

    # Load model (one legitimate model load)
    t_load_start = time.perf_counter()
    model, _tokenizer = mlx_load(str(snapshot_dir))
    t_load = time.perf_counter() - t_load_start

    # Detect memory API capabilities
    has_get_peak = hasattr(mx, "get_peak_memory")
    has_metal_peak = (
        hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory")
    )
    has_active = hasattr(mx, "get_active_memory")

    api_names: list[str] = []
    if has_get_peak:
        api_names.append("mx.get_peak_memory")
    elif has_metal_peak:
        api_names.append("mx.metal.get_peak_memory")
    if has_active:
        api_names.append("mx.get_active_memory")

    def _read_peak_bytes() -> int | None:
        if has_get_peak:
            return int(mx.get_peak_memory())
        if has_metal_peak:
            return int(mx.metal.get_peak_memory())
        return None

    def _read_current_bytes() -> int | None:
        if has_active:
            return int(mx.get_active_memory())
        return None

    def _to_mib(value: int | None) -> float | None:
        return None if value is None else round(value / (1024 * 1024), 2)

    # Reset MLX peak immediately before the oracle forward
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    baseline_bytes = _read_current_bytes()

    input_ids = mx.array(token_ids)[None]

    t_fwd_start = time.perf_counter()
    output = model(input_ids, cache=None).astype(mx.float32)
    t_fwd = time.perf_counter() - t_fwd_start

    # Evaluate and copy before reading metrics
    t_sync_start = time.perf_counter()
    mx.eval(output)
    logits_np = np.array(
        np.asarray(output, dtype=np.float32),
        dtype=np.dtype("<f4"),
        copy=True,
        order="C",
    )
    t_sync = time.perf_counter() - t_sync_start

    # Read memory metrics after eval+copy
    peak_bytes = _read_peak_bytes()
    current_bytes = _read_current_bytes()

    return {
        "logits": logits_np,
        "load_s": t_load,
        "forward_s": t_fwd,
        "sync_s": t_sync,
        "vocab_size": logits_np.shape[-1],
        "seq_len": logits_np.shape[1],
        "mlx_api": api_names,
        "mlx_version": mlx_version_core,
        "mlx_baseline_bytes": baseline_bytes,
        "mlx_peak_bytes": peak_bytes,
        "mlx_current_bytes": current_bytes,
        "mlx_baseline_mib": _to_mib(baseline_bytes),
        "mlx_peak_mib": _to_mib(peak_bytes),
        "mlx_current_mib": _to_mib(current_bytes),
    }


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
        description="Run Gemma 3n E4B-it via native MLX-LM (reference oracle).",
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        type=Path,
        help="Path to the local HF snapshot directory.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        type=str,
        help="Unique run identifier for this oracle execution.",
    )
    parser.add_argument(
        "--logits-output",
        required=True,
        type=Path,
        help="Path for the .npy logits artifact.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"User prompt text (default: {DEFAULT_PROMPT!r}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON oracle report.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Result sanitization
# ---------------------------------------------------------------------------


def sanitize_result(
    snapshot_dir: Path,
    config_dict: dict[str, Any],
    prompt_text: str,
    formatted_text: str,
    token_ids: list[int],
    versions: dict[str, str],
    optional_statuses: dict[str, Any],
    manifest: list[dict[str, Any]],
    ref_result: dict[str, Any],
    timings: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, Any]:
    """Build a sanitized JSON result report for MLX-LM reference.

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
    versions : dict
        Package versions.
    optional_statuses : dict
        JAX/PyTensor-MLX statuses.
    manifest : list[dict]
        File manifest.
    ref_result : dict
        MLX-LM reference result.
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

    # Compute top-10 tokens for reporting
    logits = ref_result["logits"][0, -1]  # Last position logits
    top10_indices = np.argsort(logits)[-10:][::-1]
    top10 = [
        {
            "id": int(idx),
            "logit": round(float(logits[idx]), 4),
            # Legacy report builder has no tokenizer argument; preserve the ID.
            "text": str(idx),
        }
        for idx in top10_indices
    ]

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
        "runtime": "mlx_lm_native",
        "versions": versions,
        "optional_statuses": optional_statuses,
        "device": get_device(),
        "file_manifest": manifest,
        "reference": {
            "vocab_size": ref_result["vocab_size"],
            "seq_len": ref_result["seq_len"],
            "load_s": round(ref_result["load_s"], 3),
            "forward_s": round(ref_result["forward_s"], 3),
            "sync_s": round(ref_result["sync_s"], 3),
            "peak_memory_mib": ref_result.get(
                "mlx_peak_mib",
                ref_result.get("peak_memory_mib"),
            ),
            "logits_sha256": hashlib.sha256(
                np.asarray(ref_result["logits"], dtype="<f4").tobytes()
            ).hexdigest(),
            "top10_final_position": top10,
        },
        "timing": timings,
        "memory": memory,
    }

    return report


# ---------------------------------------------------------------------------
# Oracle report builder
# ---------------------------------------------------------------------------


def build_oracle_report(
    *,
    run_id: str,
    snapshot_dir: Path,
    prompt_text: str,
    formatted_text: str,
    token_ids: list[int],
    oracle_result: dict[str, Any],
    npy_manifest: dict[str, Any],
    file_manifest: list[dict[str, Any]],
    versions: dict[str, str],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build the oracle JSON report.

    The report includes schema version, run identity, pinned model
    identity, prompt details, reference logits metadata, raw artifact
    manifest, versions, device, timing, memory (with separated MLX
    fields), and provenance.

    No absolute snapshot, temp, or output paths are recorded.

    Parameters
    ----------
    run_id : str
        Unique run identifier.
    snapshot_dir : Path
        Snapshot directory (only basename used via ``detect_revision``).
    prompt_text : str
        User prompt.
    formatted_text : str
        Chat-template-rendered prompt.
    token_ids : list[int]
        Prompt token IDs.
    oracle_result : dict
        Output of :func:`run_oracle_forward`.
    npy_manifest : dict
        Output of :func:`build_npy_manifest`.
    file_manifest : list[dict]
        Output of :func:`build_file_manifest`.
    versions : dict
        Package versions.
    provenance : dict
        Provenance report from :func:`build_provenance_report`.

    Returns
    -------
    dict
        Oracle report dictionary (JSON-serializable).
    """
    revision = detect_revision(snapshot_dir)
    logits = oracle_result["logits"]

    # Compute top-1 token at final position
    final_logits = logits[0, -1]
    top1_id = int(np.argmax(final_logits))

    # Compute logits hash over the full array bytes
    logits_sha256 = hashlib.sha256(
        np.asarray(logits, dtype="<f4").tobytes()
    ).hexdigest()

    report: dict[str, Any] = {
        "schema_version": GEMMA3N_ORACLE_SCHEMA_VERSION,
        "run_id": run_id,
        "model": {
            "repo": EXPECTED_REPO,
            "revision": revision,
            "model_type": EXPECTED_MODEL_TYPE,
            "architecture": EXPECTED_ARCHITECTURE,
            "quantization": {
                "bits": EXPECTED_BITS,
                "group_size": EXPECTED_GROUP_SIZE,
            },
            "manifest": file_manifest,
        },
        "prompt": {
            "text": prompt_text,
            "formatted": formatted_text,
            "token_ids": token_ids,
            "n_tokens": len(token_ids),
            "token_hash": hash_token_ids(token_ids),
        },
        "reference": {
            "shape": list(logits.shape),
            "logits_sha256": logits_sha256,
            "top1_id": top1_id,
            "vocab_size": oracle_result["vocab_size"],
            "seq_len": oracle_result["seq_len"],
        },
        "raw_artifact": npy_manifest,
        "runtime": "mlx_lm_native",
        "versions": versions,
        "device": get_device(),
        "timing": {
            "ref_load_s": round(oracle_result["load_s"], 3),
            "ref_forward_s": round(oracle_result["forward_s"], 3),
            "ref_sync_s": round(oracle_result["sync_s"], 3),
        },
        "memory": {
            "whole_process_peak_rss_mib": get_peak_rss_mib(),
            "oracle_mlx": {
                "api": oracle_result["mlx_api"],
                "version": oracle_result["mlx_version"],
                "baseline_bytes": oracle_result["mlx_baseline_bytes"],
                "current_bytes": oracle_result["mlx_current_bytes"],
                "peak_bytes": oracle_result["mlx_peak_bytes"],
                "baseline_mib": oracle_result["mlx_baseline_mib"],
                "current_mib": oracle_result["mlx_current_mib"],
                "peak_mib": oracle_result["mlx_peak_mib"],
            },
        },
        "provenance": provenance,
    }

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    # Enforce offline mode before any HF/transformers import.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    snapshot_dir = args.snapshot.resolve()

    # Phase 1: Validate snapshot
    try:
        config_dict = validate_snapshot(snapshot_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Snapshot validated: {EXPECTED_ARCHITECTURE}, "
          f"{EXPECTED_BITS}-bit, group_size={EXPECTED_GROUP_SIZE}")

    file_manifest = build_file_manifest(snapshot_dir)
    versions = collect_versions()

    # Phase 2: Tokenize (tokenizer-only loading where possible)
    print("Loading tokenizer from snapshot...")
    t_tok_start = time.time()
    tokenizer = _load_tokenizer_only(snapshot_dir)
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

    # Phase 3: Oracle forward pass (one legitimate model load)
    print("Running MLX-LM oracle forward pass (cache=None)...")
    try:
        oracle_result = run_oracle_forward(snapshot_dir, token_ids)
    except Exception as exc:
        print(f"ERROR during oracle forward: {exc}", file=sys.stderr)
        return 1
    print(f"  load: {oracle_result['load_s']:.2f}s, "
          f"forward: {oracle_result['forward_s']:.2f}s, "
          f"sync: {oracle_result['sync_s']:.2f}s")
    print(f"  shape: {oracle_result['logits'].shape}")
    print(f"  vocab_size: {oracle_result['vocab_size']}, "
          f"seq_len: {oracle_result['seq_len']}")

    # Decode top-1 token
    final_logits = oracle_result["logits"][0, -1]
    top1_id = int(np.argmax(final_logits))
    top1_text = decode_single_token(tokenizer, top1_id)
    print(f"  top1 token: {top1_id} ({top1_text!r})")

    # Phase 4: Strict nonfinite rejection
    logits_arr = oracle_result["logits"]
    if not np.all(np.isfinite(logits_arr)):
        print(
            "ERROR: logits contain non-finite values (NaN/Inf); "
            "aborting before publication",
            file=sys.stderr,
        )
        return 1

    # Phase 5: Persist .npy artifact FIRST (before manifest/JSON)
    logits_path = args.logits_output
    print(f"Writing logits to {logits_path}...")
    logits_arr = np.ascontiguousarray(logits_arr, dtype=np.dtype("<f4"))
    atomic_write_npy(logits_arr, logits_path)
    print(f"  shape: {logits_arr.shape}, dtype: {logits_arr.dtype}")

    # Phase 6: Build and verify artifact manifest (after .npy exists)
    print("Building and verifying artifact manifest...")
    npy_manifest = build_npy_manifest(logits_path)
    verified_arr = verify_npy_artifact(logits_path, npy_manifest)
    assert verified_arr.shape == logits_arr.shape
    print(f"  manifest verified: {npy_manifest['basename']}")
    print(f"  file_sha256: {npy_manifest['file_sha256'][:16]}...")
    print(f"  canonical_sha256: {npy_manifest['canonical_sha256'][:16]}...")

    # Phase 7: Build provenance from clean committed worktree
    print("Building provenance...")
    try:
        repo_root = _find_repo_root()
        impl_manifest = build_implementation_manifest(
            repo_root=repo_root,
            source_files=GEMMA3N_IMPLEMENTATION_SOURCE_FILES,
            require_clean=True,
            packages=GEMMA3N_PROVENANCE_PACKAGES,
            modules=GEMMA3N_PROVENANCE_MODULES,
            environment_yml_path=GEMMA3N_ENVIRONMENT_YML,
        )
        provenance = build_provenance_report(
            run_id=args.run_id,
            schema_version=GEMMA3N_ORACLE_SCHEMA_VERSION,
            implementation_manifest=impl_manifest,
            command=sys.argv,
        )
    except RuntimeError as exc:
        print(f"ERROR: provenance failed: {exc}", file=sys.stderr)
        return 1
    print(f"  git_commit: {provenance['implementation']['git_commit'][:12]}")
    print(f"  git_clean: {provenance['implementation']['git_clean']}")

    # Phase 8: Build oracle report
    report = build_oracle_report(
        run_id=args.run_id,
        snapshot_dir=snapshot_dir,
        prompt_text=args.prompt,
        formatted_text=formatted_text,
        token_ids=token_ids,
        oracle_result=oracle_result,
        npy_manifest=npy_manifest,
        file_manifest=file_manifest,
        versions=versions,
        provenance=provenance,
    )

    # Phase 9: Write JSON LAST (after .npy and manifest)
    if args.output:
        atomic_write_json(report, args.output)
        print(f"Oracle report written to {args.output}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
