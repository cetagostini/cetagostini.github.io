"""Validate pytensor MLX backend against NumPy reference for split/join, Linear, and MultiheadAttention.

Deterministic asymmetric nonzero weights and inputs are used throughout. The script compares
FAST_COMPILE (NumPy reference) and a custom MLX mode to a pure NumPy baseline, recording exact
success/failure for each check.

This script documents a **compatibility matrix** — expected backend defects (e.g., missing MLX
SplitDims conversion) are not treated as validator process failures. ``main()`` returns 0 only
when the actual matrix matches ``EXPECTED_OUTCOMES``; any unexpected pass, failure, or error
returns 1, signalling that the evidence record needs updating.

Usage
-----
    python validate_pytensor_mlx.py                  # print compact statuses to stdout
    python validate_pytensor_mlx.py --output out.json  # also write allowlisted JSON atomically
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

ALLOWED_KEYS = frozenset({
    "metadata",
    "split_join",
    "linear_rank2",
    "linear_rank3",
    "linear_rank4",
    "matrix_match",
    "discrepancies",
    "expected_outcomes",
    "multihead_attention",
})

EXPECTED_OUTCOMES = {
    "linear_rank2": {
        "fast_compile": True,
        "mlx": True,
    },
    "linear_rank3": {
        "fast_compile": True,
        "mlx": False,
    },
    "linear_rank4": {
        "fast_compile": True,
        "mlx": False,
    },
    "split_join": {
        "error": True,
    },
    "multihead_attention": {
        "error": True,
    },
}


def deterministic_array(
    shape: tuple[int, ...],
    seed: int,
    low: float = 0.1,
    high: float = 1.7,
) -> np.ndarray:
    """Return a deterministic, asymmetric, nonzero float32 array.

    Parameters
    ----------
    shape : tuple of int
        Output shape.
    seed : int
        Seed for the deterministic RNG.
    low : float
        Lower bound of the uniform draw (exclusive of zero).
    high : float
        Upper bound of the uniform draw.

    Returns
    -------
    numpy.ndarray
        A float32 array with no zero entries and no two rows identical.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(low, high, size=shape).astype(np.float32)
    ramp = np.arange(x.size, dtype=np.float32).reshape(shape) * 0.001
    return x + ramp


def make_mlx_mode():
    """Build the exact custom MLX mode used by pytensor_ml.

    Constructs an MLX mode from ``MLXLinker`` and a ``RewriteDatabaseQuery`` that includes
    ``['mlx']`` and excludes whatever ``MLX._optimizer`` excludes, matching the upstream
    convention.

    Returns
    -------
    pytensor.compile.mode.Mode
        A fresh MLX compilation mode.
    """
    import pytensor.compile.mode
    from pytensor.compile.mode import Mode
    from pytensor.graph.rewriting.db import RewriteDatabaseQuery
    from pytensor.link.mlx.linker import MLXLinker
    from pytensor.link.mlx import MLX

    return Mode(
        MLXLinker(),
        RewriteDatabaseQuery(include=["mlx"], exclude=MLX._optimizer.exclude),
    )


def compare_arrays(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict:
    """Compare two arrays and return a status dict.

    Parameters
    ----------
    actual : array-like
        The array produced by the backend under test.
    expected : numpy.ndarray
        The NumPy reference array.
    label : str
        Human-readable label for the comparison.
    atol : float
        Absolute tolerance for ``numpy.allclose``.
    rtol : float
        Relative tolerance for ``numpy.allclose``.

    Returns
    -------
    dict
        Keys: ``label``, ``pass`` (bool), ``expected_shape``, ``actual_shape``,
        ``max_abs_diff``, ``allclose``.
    """
    actual_np = np.asarray(actual, dtype=np.float32)
    expected_np = np.asarray(expected, dtype=np.float32)
    shapes_match = actual_np.shape == expected_np.shape
    if shapes_match:
        close = bool(np.allclose(actual_np, expected_np, atol=atol, rtol=rtol))
        diff = float(np.max(np.abs(actual_np - expected_np)))
    else:
        close = False
        diff = None

    return {
        "label": label,
        "pass": shapes_match and close,
        "expected_shape": list(expected_np.shape),
        "actual_shape": list(actual_np.shape),
        "max_abs_diff": diff,
        "allclose": close,
    }


def collect_metadata() -> dict:
    """Collect version and hardware metadata.

    Returns
    -------
    dict
        Keys: ``python``, ``pytensor``, ``pytensor_ml``, ``mlx``, ``platform``, ``machine``,
        ``mlx_device``.
    """
    import pytensor
    import pytensor_ml

    try:
        import mlx.core as mx
        mlx_version = mx.__version__
        mlx_device = str(mx.default_device())
    except Exception:
        mlx_version = "unavailable"
        mlx_device = "unavailable"

    return {
        "python": platform.python_version(),
        "pytensor": pytensor.__version__,
        "pytensor_ml": pytensor_ml.__version__,
        "mlx": mlx_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx_device": mlx_device,
    }


def _extract_backend_pass(entry: dict, backend: str) -> bool:
    """Extract the pass/fail status for a backend from a check result entry.

    Handles both linear/split_join-style (nested dict with ``"pass"`` key) and
    MHA-style (flat ``"fast_compile_pass"`` / ``"mlx_pass"`` keys) results.

    Parameters
    ----------
    entry : dict
        A single check result dict from :func:`run_all_checks`.
    backend : str
        Backend name, e.g. ``"fast_compile"`` or ``"mlx"``.

    Returns
    -------
    bool
        True if the backend passed, False otherwise.
    """
    if backend in entry and isinstance(entry[backend], dict):
        return bool(entry[backend].get("pass", False))
    mha_key = f"{backend}_pass"
    if mha_key in entry:
        return bool(entry[mha_key])
    return False


def matches_expected_outcomes(result: dict) -> tuple[bool, list[str]]:
    """Compare actual check results to the expected compatibility matrix.

    Parameters
    ----------
    result : dict
        Output of :func:`run_all_checks`.

    Returns
    -------
    tuple of (bool, list of str)
        ``(True, [])`` if all outcomes match ``EXPECTED_OUTCOMES``.
        ``(False, discrepancies)`` listing each mismatch.
    """
    discrepancies = []
    for check_name, expected in EXPECTED_OUTCOMES.items():
        actual = result.get(check_name, {})
        if expected.get("error"):
            if "error" not in actual:
                fast_pass = _extract_backend_pass(actual, "fast_compile")
                mlx_pass = _extract_backend_pass(actual, "mlx")
                discrepancies.append(
                    f"{check_name}: expected ERROR but got result "
                    f"(fast_compile={'PASS' if fast_pass else 'FAIL'}, "
                    f"mlx={'PASS' if mlx_pass else 'FAIL'}) — evidence update needed"
                )
            continue
        if "error" in actual:
            discrepancies.append(
                f"{check_name}: expected result but got ERROR: {actual['error']}"
            )
            continue
        for backend, expected_pass in expected.items():
            actual_pass = _extract_backend_pass(actual, backend)
            if actual_pass != expected_pass:
                discrepancies.append(
                    f"{check_name}/{backend}: expected "
                    f"{'PASS' if expected_pass else 'FAIL'}, got "
                    f"{'PASS' if actual_pass else 'FAIL'}"
                )
    return (len(discrepancies) == 0, discrepancies)


def check_split_join() -> dict:
    """Compare ``pt.split_dims`` / ``pt.join_dims`` under FAST_COMPILE and MLX to NumPy.

    Builds a graph that splits the trailing axis of a (2, 3, 12) tensor into (3, 4), swaps
    axes, then joins them back. Compares the output to a pure NumPy reference.

    Returns
    -------
    dict
        Keys: ``fast_compile`` and ``mlx``, each a comparison status dict.
        On error, contains ``error`` key instead.
    """
    import pytensor
    import pytensor.tensor as pt

    x_sym = pt.tensor("x", shape=(2, 3, 12), dtype="float32")
    split = pt.split_dims(x_sym, shape=(3, 4), axis=-1)
    swapped = split.swapaxes(-2, -1)
    joined = pt.join_dims(swapped, start_axis=-2, n_axes=2)
    x_val = deterministic_array((2, 3, 12), seed=42)

    # NumPy reference
    np_split = x_val.reshape(2, 3, 3, 4)
    np_swapped = np_split.swapaxes(-2, -1)
    np_ref = np_swapped.reshape(2, 3, 12)

    try:
        fast_fn = pytensor.function([x_sym], joined, mode="FAST_COMPILE")
        fast_out = fast_fn(x_val)
        fast_result = compare_arrays(fast_out, np_ref, "split_join/fast_compile")
    except Exception as exc:
        return {"error": str(exc), "pass": False}

    try:
        mlx_mode = make_mlx_mode()
        mlx_fn = pytensor.function([x_sym], joined, mode=mlx_mode)
        mlx_out = mlx_fn(x_val)
        mlx_result = compare_arrays(mlx_out, np_ref, "split_join/mlx")
    except Exception as exc:
        return {"error": str(exc), "pass": False}

    return {
        "fast_compile": fast_result,
        "mlx": mlx_result,
    }


def _numpy_linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pure NumPy linear: ``x @ W + b``."""
    return (x @ W) + b


def check_linear_rank(rank: int) -> dict:
    """Compare ``pytensor_ml.layers.Linear`` under FAST_COMPILE and MLX to NumPy.

    Parameters
    ----------
    rank : int
        Input tensor rank (2, 3, or 4).

    Returns
    -------
    dict
        Keys: ``rank``, ``fast_compile``, ``mlx``, ``numpy_ref_shape``,
        ``mlx_actual_shape``.
    """
    import pytensor
    import pytensor.tensor as pt
    from pytensor_ml.layers import Linear

    n_in, n_out = 4, 5
    if rank == 2:
        in_shape = (3, n_in)
    elif rank == 3:
        in_shape = (2, 3, n_in)
    elif rank == 4:
        in_shape = (2, 2, 3, n_in)
    else:
        raise ValueError(f"Unsupported rank {rank}")

    layer = Linear("test_linear", n_in, n_out, bias=True)
    x_sym = pt.tensor("x", shape=in_shape, dtype="float32")
    out = layer(x_sym)

    x_val = deterministic_array(in_shape, seed=100 + rank)
    W_val = deterministic_array((n_in, n_out), seed=200 + rank)
    b_val = deterministic_array((n_out,), seed=300 + rank)
    layer.W.set_value(W_val)
    layer.b.set_value(b_val)

    np_ref = _numpy_linear(x_val, W_val, b_val)

    fast_fn = pytensor.function([x_sym], out, mode="FAST_COMPILE")
    fast_out = fast_fn(x_val)

    mlx_mode = make_mlx_mode()
    mlx_fn = pytensor.function([x_sym], out, mode=mlx_mode)
    mlx_out = mlx_fn(x_val)

    return {
        "rank": rank,
        "numpy_ref_shape": list(np_ref.shape),
        "fast_compile": compare_arrays(
            fast_out, np_ref, f"linear_rank{rank}/fast_compile"
        ),
        "mlx": compare_arrays(
            mlx_out, np_ref, f"linear_rank{rank}/mlx"
        ),
    }


def check_multihead_attention() -> dict:
    """Compare ``pytensor_ml.layers.MultiheadAttention`` under FAST_COMPILE and MLX.

    Returns
    -------
    dict
        Keys: ``fast_compile_pass``, ``mlx_pass``, or ``error`` on failure.
    """
    import pytensor
    import pytensor.tensor as pt
    from pytensor_ml.layers import MultiheadAttention

    batch, seq, n_heads, head_dim = 2, 4, 2, 8
    n_in = n_heads * head_dim
    n_out = n_in

    layer = MultiheadAttention(
        "test_mha",
        n_in=n_in,
        n_heads=n_heads,
        head_dim=head_dim,
    )

    x_sym = pt.tensor("x", shape=(batch, seq, n_in), dtype="float32")
    out = layer(x_sym)

    x_val = deterministic_array((batch, seq, n_in), seed=500)

    try:
        fast_fn = pytensor.function([x_sym], out, mode="FAST_COMPILE")
        fast_out = fast_fn(x_val)
        fast_pass = fast_out.shape == (batch, seq, n_out)
    except Exception as exc:
        return {"error": str(exc), "pass": False}

    try:
        mlx_mode = make_mlx_mode()
        mlx_fn = pytensor.function([x_sym], out, mode=mlx_mode)
        mlx_out = mlx_fn(x_val)
        mlx_pass = mlx_out.shape == (batch, seq, n_out)
    except Exception as exc:
        return {"error": str(exc), "pass": False}

    return {
        "fast_compile_pass": fast_pass,
        "mlx_pass": mlx_pass,
    }


def run_all_checks() -> dict:
    """Run all compatibility checks and return the complete result.

    Returns
    -------
    dict
        Keys from ``ALLOWED_KEYS``: ``metadata``, ``split_join``,
        ``linear_rank2``, ``linear_rank3``, ``linear_rank4``,
        ``multihead_attention``, ``matrix_match``, ``discrepancies``,
        ``expected_outcomes``.
    """
    result = {
        "metadata": collect_metadata(),
    }

    # Split/join check
    try:
        result["split_join"] = check_split_join()
    except Exception as exc:
        result["split_join"] = {"error": str(exc), "pass": False}

    # Linear rank checks
    for rank in [2, 3, 4]:
        try:
            result[f"linear_rank{rank}"] = check_linear_rank(rank)
        except Exception as exc:
            result[f"linear_rank{rank}"] = {"error": str(exc), "pass": False}

    # Multihead attention check
    try:
        result["multihead_attention"] = check_multihead_attention()
    except Exception as exc:
        result["multihead_attention"] = {"error": str(exc), "pass": False}

    # Check matrix match
    matrix_match, discrepancies = matches_expected_outcomes(result)
    result["matrix_match"] = matrix_match
    result["discrepancies"] = discrepancies
    result["expected_outcomes"] = EXPECTED_OUTCOMES

    return result


def write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON data atomically using a temporary file.

    Parameters
    ----------
    path : Path
        Destination file path.
    data : dict
        Data to serialize as JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=path.stem
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def main(argv: list[str] | None = None) -> int:
    """Run the MLX compatibility validator and return exit code.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments. If None, uses sys.argv[1:].

    Returns
    -------
    int
        0 if the actual matrix matches EXPECTED_OUTCOMES, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Validate pytensor MLX backend compatibility matrix"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write results to JSON file",
    )
    args = parser.parse_args(argv)

    result = run_all_checks()

    # Print summary
    for check_name in ["split_join", "linear_rank2", "linear_rank3",
                       "linear_rank4", "multihead_attention"]:
        entry = result.get(check_name, {})
        if "error" in entry:
            print(f"{check_name:25s} ERROR")
        else:
            fast_pass = _extract_backend_pass(entry, "fast_compile")
            mlx_pass = _extract_backend_pass(entry, "mlx")
            print(
                f"{check_name:25s} "
                f"fast_compile={'PASS' if fast_pass else 'FAIL'} "
                f"mlx={'PASS' if mlx_pass else 'FAIL'}"
            )

    print(f"\n{'matrix_match':25s} {result['matrix_match']}")
    if result["discrepancies"]:
        for d in result["discrepancies"]:
            print(f"  DISCREPANCY: {d}")

    # Write output if requested
    if args.output:
        write_json_atomic(args.output, result)
        print(f"\nResults written to {args.output}")

    return 0 if result["matrix_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
