#!/usr/bin/env python3
"""Run Gemma 3n E4B-it via native MLX-LM (reference oracle).

Standalone MLX-LM reference runner for Gemma 3n.  Loads a pinned local HF
snapshot, tokenizes a prompt via the snapshot's chat template, and runs a
direct MLX-LM forward pass with ``cache=None`` to obtain all-position
float32 reference logits.

This module serves as the oracle baseline for PyTensor comparisons and can
be used independently for reference-only evaluations.

Usage::

    python -m cetagostini.utils.pytensor.run_gemma_mlx --snapshot /path/to/snapshot
    python -m cetagostini.utils.pytensor.run_gemma_mlx --snapshot /path/to/snapshot --output results/gemma_mlx.json
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
    atomic_write_json,
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
    validate_snapshot,
)


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
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"User prompt text (default: {DEFAULT_PROMPT!r}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON result report.",
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
            "text": decode_single_token(None, int(idx)) if False else str(idx),
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
            "peak_memory_mib": ref_result["peak_memory_mib"],
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

    manifest = build_file_manifest(snapshot_dir)
    versions = collect_versions()
    optional_statuses = check_optional_statuses()

    # Phase 2: Tokenize
    print("Loading tokenizer from snapshot...")
    t_tok_start = time.time()
    tokenizer = load_tokenizer_from_snapshot(snapshot_dir)
    formatted_text, token_ids = format_and_tokenize(tokenizer, args.prompt)
    t_tokenize = time.time() - t_tok_start
    T = len(token_ids)
    print(f"  {T} tokens, formatted in {t_tokenize:.2f}s")
    print(f"  token_hash: {hash_token_ids(token_ids)}")

    # Phase 3: MLX-LM reference forward pass
    print("Running MLX-LM reference forward pass (cache=None)...")
    try:
        ref_result = run_mlx_reference(snapshot_dir, token_ids)
    except Exception as exc:
        print(f"ERROR during MLX-LM reference: {exc}", file=sys.stderr)
        return 1
    print(f"  load: {ref_result['load_s']:.2f}s, "
          f"forward: {ref_result['forward_s']:.2f}s, "
          f"sync: {ref_result['sync_s']:.2f}s")
    print(f"  vocab_size: {ref_result['vocab_size']}, "
          f"seq_len: {ref_result['seq_len']}")

    # Decode top-1 token
    final_logits = ref_result["logits"][0, -1]
    top1_id = int(np.argmax(final_logits))
    top1_text = decode_single_token(tokenizer, top1_id)
    print(f"  top1 token: {top1_id} ({top1_text!r})")

    # Phase 4: Build timings & memory
    timings: dict[str, Any] = {
        "tokenize_s": round(t_tokenize, 3),
        "ref_load_s": round(ref_result["load_s"], 3),
        "ref_forward_s": round(ref_result["forward_s"], 3),
        "ref_sync_s": round(ref_result["sync_s"], 3),
    }

    memory: dict[str, Any] = {
        "peak_rss_mib": get_peak_rss_mib(),
        "mlx_peak_memory_mib": ref_result["peak_memory_mib"],
    }

    # Phase 5: Build report
    report = sanitize_result(
        snapshot_dir=snapshot_dir,
        config_dict=config_dict,
        prompt_text=args.prompt,
        formatted_text=formatted_text,
        token_ids=token_ids,
        versions=versions,
        optional_statuses=optional_statuses,
        manifest=manifest,
        ref_result=ref_result,
        timings=timings,
        memory=memory,
    )

    # Phase 6: Write output
    if args.output:
        atomic_write_json(report, args.output)
        print(f"Report written to {args.output}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
