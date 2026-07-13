#!/usr/bin/env python3
"""Run SmolLM2-135M-Instruct via native MLX-LM (reference oracle).

Standalone MLX-LM reference runner for SmolLM2.  Loads the model via
``mlx_lm`` and runs greedy autoregressive generation as a reference
baseline for PyTensor comparisons.

This module serves as the oracle baseline for SmolLM2 PyTensor comparisons
and can be used independently for reference-only evaluations.

Usage::

    python -m cetagostini.utils.pytensor.run_smollm2_mlx --model /path/to/snapshot
    python -m cetagostini.utils.pytensor.run_smollm2_mlx --model /path/to/snapshot --output results/smollm2_mlx.json
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
import time
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = "What is 2 + 2? Answer with only the number."
DEFAULT_MAX_TOKENS = 16


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    """Argparse type that rejects values < 1."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(
            f"must be >= 1, got {ivalue}"
        )
    return ivalue


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
        description="Run SmolLM2-135M-Instruct via native MLX-LM (reference oracle).",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to the local HF snapshot directory or model path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"User prompt text (default: {DEFAULT_PROMPT!r}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS}, must be >= 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON result report.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# MLX-LM generation
# ---------------------------------------------------------------------------


def run_mlx_generation(
    model_path: Path,
    prompt_text: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Run MLX-LM greedy generation.

    Parameters
    ----------
    model_path : Path
        Path to the model snapshot.
    prompt_text : str
        User prompt text.
    max_tokens : int
        Maximum tokens to generate.

    Returns
    -------
    dict
        Keys: ``generated_ids``, ``generated_text``, ``prompt_ids``,
        ``prompt_text``, ``load_s``, ``generation_s``, ``peak_memory_mib``.
    """
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm import generate as mlx_generate

    t_load_start = time.time()
    model, tokenizer = mlx_load(str(model_path))
    t_load = time.time() - t_load_start

    mx.reset_peak_memory()

    # Format prompt using chat template
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    t_gen_start = time.time()
    # Use mlx_lm.generate for greedy generation
    response = mlx_generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        temp=0.0,  # Greedy
        verbose=False,
    )
    t_gen = time.time() - t_gen_start

    # Extract token IDs from the response
    # mlx_lm.generate returns a string; we need to re-tokenize to get IDs
    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    full_ids = tokenizer.encode(formatted + response, add_special_tokens=False)
    generated_ids = full_ids[len(prompt_ids):]

    peak_mem = _get_mlx_peak_memory_mib()

    return {
        "generated_ids": [int(t) for t in generated_ids],
        "generated_text": response,
        "prompt_ids": [int(t) for t in prompt_ids],
        "prompt_text": formatted,
        "load_s": t_load,
        "generation_s": t_gen,
        "peak_memory_mib": peak_mem,
    }


def _get_mlx_peak_memory_mib() -> float | None:
    """Get MLX peak memory in mebibytes, or None if unavailable."""
    try:
        import mlx.core as mx
        if hasattr(mx, "get_peak_memory"):
            return round(mx.get_peak_memory() / (1024 * 1024), 2)
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
            return round(mx.metal.get_peak_memory() / (1024 * 1024), 2)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Version & memory helpers
# ---------------------------------------------------------------------------


def collect_versions() -> dict[str, str]:
    """Collect package versions for the result report."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for mod_name, distribution_name in (
        ("mlx", "mlx"),
        ("mlx_lm", "mlx-lm"),
    ):
        try:
            versions[mod_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            versions[mod_name] = "unavailable"

    return versions


def get_peak_rss_mib() -> float:
    """Get peak RSS in mebibytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return round(rss / (1024 * 1024), 2)
    return round(rss / 1024, 2)


def get_device() -> str:
    """Return a short string describing the compute device."""
    machine = platform.machine()
    if platform.system() == "Darwin" and machine == "arm64":
        return "apple_silicon"
    return machine


# ---------------------------------------------------------------------------
# Result sanitization
# ---------------------------------------------------------------------------


def sanitize_result(
    model_path: Path,
    prompt_text: str,
    versions: dict[str, str],
    gen_result: dict[str, Any],
    timings: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, Any]:
    """Build a sanitized JSON result report for MLX-LM reference.

    Parameters
    ----------
    model_path : Path
        Model path (only basename used).
    prompt_text : str
        User prompt.
    versions : dict
        Package versions.
    gen_result : dict
        Generation result from ``run_mlx_generation``.
    timings : dict
        Timing breakdown.
    memory : dict
        Memory metrics.

    Returns
    -------
    dict
        Sanitized report dictionary.
    """
    report: dict[str, Any] = {
        "model": {
            "path": model_path.name,
        },
        "prompt": {
            "text": prompt_text,
            "formatted": gen_result["prompt_text"],
            "token_ids": gen_result["prompt_ids"],
            "n_tokens": len(gen_result["prompt_ids"]),
        },
        "runtime": "mlx_lm_native",
        "versions": versions,
        "device": get_device(),
        "generation": {
            "generated_ids": gen_result["generated_ids"],
            "text": gen_result["generated_text"],
            "n_tokens": len(gen_result["generated_ids"]),
        },
        "timing": timings,
        "memory": memory,
    }

    return report


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def atomic_write_json(data: dict[str, Any], dest: Path) -> None:
    """Write JSON atomically via a temporary file + rename.

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
    import tempfile

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(dest.parent), suffix=".tmp", prefix=".run_smollm2_mlx_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True, allow_nan=False)
            f.write("\n")
        os.replace(tmp_path, str(dest))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


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
    model_path = args.model.resolve()

    if not model_path.exists():
        print(f"ERROR: Model path not found: {model_path}", file=sys.stderr)
        return 1

    # Enforce offline mode before any HF/transformers import.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Phase 1: Run MLX-LM generation
    print(f"Running MLX-LM generation (max_tokens={args.max_tokens})...")
    try:
        gen_result = run_mlx_generation(
            model_path,
            args.prompt,
            args.max_tokens,
        )
    except Exception as exc:
        print(f"ERROR during MLX-LM generation: {exc}", file=sys.stderr)
        return 1

    print(f"  load: {gen_result['load_s']:.2f}s")
    print(f"  generation: {gen_result['generation_s']:.2f}s")
    print(f"  generated {len(gen_result['generated_ids'])} tokens")
    print(f"  text: {gen_result['generated_text']!r}")

    # Phase 2: Build timings & memory
    timings: dict[str, Any] = {
        "load_s": round(gen_result["load_s"], 3),
        "generation_s": round(gen_result["generation_s"], 3),
    }

    memory: dict[str, Any] = {
        "peak_rss_mib": get_peak_rss_mib(),
        "mlx_peak_memory_mib": gen_result["peak_memory_mib"],
    }

    # Phase 3: Build report
    versions = collect_versions()
    report = sanitize_result(
        model_path=model_path,
        prompt_text=args.prompt,
        versions=versions,
        gen_result=gen_result,
        timings=timings,
        memory=memory,
    )

    # Phase 4: Write output
    if args.output:
        atomic_write_json(report, args.output)
        print(f"Report written to {args.output}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
