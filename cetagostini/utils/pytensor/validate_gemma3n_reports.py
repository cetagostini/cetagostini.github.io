"""Cross-report validation for Gemma 3n oracle and PyTensor backend reports.

Loads four strict JSON reports (oracle, C, Numba, MLX), verifies the raw
``.npy`` artifact through shared evidence helpers, validates every schema
field, and atomically emits a machine-readable validation/comparison report.

Every gate is collected as ``{name, expected, actual, passed}``.  The CLI
returns nonzero if any gate fails.

Usage::

    python -m cetagostini.utils.pytensor.validate_gemma3n_reports \\
        --run-id run-001 \\
        --oracle oracle.json --oracle-logits logits.npy \\
        --c c.json --numba numba.json --mlx mlx.json \\
        --output validation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from cetagostini.utils.pytensor.evidence import (
    GEMMA3N_ORACLE_SCHEMA_VERSION,
    ArtifactVerificationError,
    build_npy_manifest,
    verify_npy_artifact,
)
from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
    DEFAULT_PROMPT,
    EXPECTED_ARCHITECTURE,
    EXPECTED_BITS,
    EXPECTED_GROUP_SIZE,
    EXPECTED_MANIFEST,
    EXPECTED_MODEL_TYPE,
    EXPECTED_PROMPT_TOKEN_HASH,
    EXPECTED_PROMPT_TOKEN_IDS,
    EXPECTED_REPO,
    EXPECTED_REVISION,
    EXPECTED_VOCAB_SIZE,
    PUB_ALL_TOP1_MATCH,
    PUB_COSINE_MIN,
    PUB_PEARSON_MIN,
    PUB_TOP10_OVERLAP_MEAN_MIN,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALIDATION_SCHEMA_VERSION = "gemma3n-validation-v1"

EXPECTED_BACKEND_NAMES = {"c": "c", "numba": "numba", "mlx": "mlx"}

EXPECTED_N_POSITIONS = 20
EXPECTED_N_LAYERS = 35
EXPECTED_CHUNK_SIZE = 4096
EXPECTED_LAYER_TYPES = tuple(
    "full_attention" if (index + 1) % 5 == 0 else "sliding_attention"
    for index in range(EXPECTED_N_LAYERS)
)
EXPECTED_ROPE_BASES = tuple(
    "1M" if layer_type == "full_attention" else "10K"
    for layer_type in EXPECTED_LAYER_TYPES
)
EXPECTED_SPARSE_LAYERS = tuple(range(10))

EXPECTED_VERSIONS = {
    "python": "3.13.14",
    "numpy": "2.4.6",
    "pytensor": "3.1.2",
    "numba": "0.65.1",
    "mlx": "0.32.0",
    "mlx_lm": "0.31.3",
    "transformers": "5.12.1",
    "pytensor_ml": "0.0.5.dev24+gf6ecf81d5",
}

_ABS_PATH_RE = re.compile(r"(?:^|[\"\s:=])(/[a-zA-Z0-9_./ -]+)")
_HTTP_URL_RE = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Gate collector
# ---------------------------------------------------------------------------


class GateCollector:
    """Accumulate named gates with expected/actual/passed semantics."""

    def __init__(self) -> None:
        self._gates: list[dict[str, Any]] = []

    def add(
        self,
        name: str,
        expected: Any,
        actual: Any,
        passed: bool,
    ) -> None:
        self._gates.append({
            "name": name,
            "expected": expected,
            "actual": actual,
            "passed": bool(passed),
        })

    @property
    def gates(self) -> list[dict[str, Any]]:
        return list(self._gates)

    @property
    def all_passed(self) -> bool:
        return all(g["passed"] for g in self._gates)

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [g for g in self._gates if not g["passed"]]


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------


def load_strict_json(path: Path) -> dict[str, Any]:
    """Load a JSON file with strict parsing (no NaN/Inf).

    Parameters
    ----------
    path : Path
        Path to a JSON file.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    ValueError
        If the file is not valid JSON or contains non-object root.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"report not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text, parse_constant=_reject_nonfinite)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: root must be a JSON object")
    return data


def _reject_nonfinite(token: str) -> Any:
    """Reject NaN/Infinity tokens during JSON parsing."""
    raise ValueError(f"non-finite JSON token: {token!r}")


# ---------------------------------------------------------------------------
# Path leakage detection
# ---------------------------------------------------------------------------


def _contains_absolute_path(value: Any) -> list[str]:
    """Recursively scan a JSON value for absolute filesystem paths.

    Returns a list of detected absolute path strings.
    """
    found: list[str] = []
    if isinstance(value, str):
        scan_value = _HTTP_URL_RE.sub("", value)
        if _ABS_PATH_RE.search(scan_value):
            for m in _ABS_PATH_RE.finditer(scan_value):
                candidate = m.group(1).strip()
                if len(candidate) > 2 and "/" in candidate:
                    found.append(candidate)
        if scan_value.startswith("/") and len(scan_value) > 3:
            parts = scan_value.split("/")
            if len(parts) >= 3 and all(parts[i] for i in range(1, min(3, len(parts)))):
                found.append(scan_value)
    elif isinstance(value, dict):
        for v in value.values():
            found.extend(_contains_absolute_path(v))
    elif isinstance(value, list):
        for item in value:
            found.extend(_contains_absolute_path(item))
    return found


# ---------------------------------------------------------------------------
# Stage validation helpers
# ---------------------------------------------------------------------------


def _expected_stage_labels(
    n_layers: int,
    vocab_size: int,
    chunk_size: int,
) -> list[str]:
    """Compute the expected ordered stage labels for an MLX backend run.

    Parameters
    ----------
    n_layers : int
        Number of decoder layers.
    vocab_size : int
        Model vocabulary size.
    chunk_size : int
        Vocabulary chunk size for logit projection.

    Returns
    -------
    list of str
        Ordered stage labels.
    """
    labels: list[str] = []
    labels.append("initial_projections")
    labels.append("per_layer_projection")
    for i in range(n_layers):
        labels.append(f"layer_{i}")
    labels.append("final_unembed")
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        labels.append(f"vocab_chunk_{start}_{end}")
    return labels


def _expected_n_chunks(vocab_size: int, chunk_size: int) -> int:
    """Compute the expected number of vocabulary chunks."""
    n_full = vocab_size // chunk_size
    remainder = vocab_size % chunk_size
    return n_full + (1 if remainder else 0)


def _expected_n_stages(
    n_layers: int,
    vocab_size: int,
    chunk_size: int,
) -> int:
    """Compute the expected total number of stages."""
    return 2 + n_layers + 1 + _expected_n_chunks(vocab_size, chunk_size)


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------


def _check_common_schema_and_run_id(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
    expected_run_id: str,
) -> None:
    """Check schema_version and run_id across all reports."""
    for label, report in reports.items():
        sv = report.get("schema_version")
        gates.add(
            f"{label}/schema_version",
            GEMMA3N_ORACLE_SCHEMA_VERSION,
            sv,
            sv == GEMMA3N_ORACLE_SCHEMA_VERSION,
        )
        rid = report.get("run_id")
        gates.add(
            f"{label}/run_id",
            expected_run_id,
            rid,
            rid == expected_run_id,
        )


def _check_provenance(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
) -> None:
    """Check implementation provenance across reports.

    The oracle report nests provenance under ``provenance.implementation``.
    Backend reports may omit the full provenance section; this is reported
    as a blocker rather than silently weakened.
    """
    oracle = reports["oracle"]
    oracle_prov = oracle.get("provenance", {})
    oracle_impl = oracle_prov.get("implementation", {})

    oracle_commit = oracle_impl.get("git_commit")
    oracle_clean = oracle_impl.get("git_clean")
    oracle_manifest_sha = oracle_impl.get("implementation_manifest_sha256")
    oracle_env = oracle_impl.get("environment")
    oracle_source_hashes = oracle_impl.get("source_hashes")
    oracle_python_exec = oracle_impl.get("python_executable")

    gates.add(
        "oracle/provenance_git_clean",
        True,
        oracle_clean,
        oracle_clean is True,
    )

    for label in ("c", "numba", "mlx"):
        report = reports[label]
        prov = report.get("provenance")
        if prov is None:
            impl = None
        else:
            impl = prov.get("implementation")

        if impl is None:
            gates.add(
                f"{label}/provenance_implementation_present",
                True,
                False,
                False,
            )
            continue

        gates.add(
            f"{label}/provenance_implementation_present",
            True,
            True,
            True,
        )

        commit = impl.get("git_commit")
        if oracle_commit is not None and commit is not None:
            gates.add(
                f"{label}/git_commit_matches_oracle",
                oracle_commit,
                commit,
                commit == oracle_commit,
            )

        clean = impl.get("git_clean")
        gates.add(
            f"{label}/git_clean",
            True,
            clean,
            clean is True,
        )

        manifest_sha = impl.get("implementation_manifest_sha256")
        if oracle_manifest_sha is not None and manifest_sha is not None:
            gates.add(
                f"{label}/implementation_manifest_sha256_matches_oracle",
                oracle_manifest_sha,
                manifest_sha,
                manifest_sha == oracle_manifest_sha,
            )

        env = impl.get("environment")
        if oracle_env is not None and env is not None:
            gates.add(
                f"{label}/environment_matches_oracle",
                oracle_env,
                env,
                env == oracle_env,
            )

        source_hashes = impl.get("source_hashes")
        if oracle_source_hashes is not None and source_hashes is not None:
            gates.add(
                f"{label}/source_hashes_matches_oracle",
                oracle_source_hashes,
                source_hashes,
                source_hashes == oracle_source_hashes,
            )

        python_exec = impl.get("python_executable")
        if oracle_python_exec is not None and python_exec is not None:
            gates.add(
                f"{label}/python_executable_matches_oracle",
                oracle_python_exec,
                python_exec,
                python_exec == oracle_python_exec,
            )

        for key in (
            "environment_yml_sha256",
            "package_versions",
            "module_paths",
        ):
            expected = oracle_impl.get(key)
            actual = impl.get(key)
            gates.add(
                f"{label}/{key}_matches_oracle",
                expected,
                actual,
                expected is not None and actual == expected,
            )


def _check_model_identity(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
) -> None:
    """Check pinned model identity across all reports."""
    for label, report in reports.items():
        model = report.get("model", {})
        gates.add(
            f"{label}/model_repo",
            EXPECTED_REPO,
            model.get("repo"),
            model.get("repo") == EXPECTED_REPO,
        )
        gates.add(
            f"{label}/model_revision",
            EXPECTED_REVISION,
            model.get("revision"),
            model.get("revision") == EXPECTED_REVISION,
        )
        gates.add(
            f"{label}/model_type",
            EXPECTED_MODEL_TYPE,
            model.get("model_type"),
            model.get("model_type") == EXPECTED_MODEL_TYPE,
        )
        gates.add(
            f"{label}/model_architecture",
            EXPECTED_ARCHITECTURE,
            model.get("architecture"),
            model.get("architecture") == EXPECTED_ARCHITECTURE,
        )
        quant = model.get("quantization", {})
        gates.add(
            f"{label}/model_quantization_bits",
            EXPECTED_BITS,
            quant.get("bits"),
            quant.get("bits") == EXPECTED_BITS,
        )
        gates.add(
            f"{label}/model_quantization_group_size",
            EXPECTED_GROUP_SIZE,
            quant.get("group_size"),
            quant.get("group_size") == EXPECTED_GROUP_SIZE,
        )

    # Cross-report model manifest SHA consistency
    oracle_manifest = reports["oracle"].get("model", {}).get("manifest", [])
    if oracle_manifest:
        oracle_shas = {
            entry.get("name"): entry.get("sha256")
            for entry in oracle_manifest
        }
        for label in ("c", "numba", "mlx"):
            backend_manifest = reports[label].get("model", {}).get("manifest", [])
            if not backend_manifest:
                backend_manifest = reports[label].get("file_manifest", [])
            backend_shas = {
                entry.get("name"): entry.get("sha256")
                for entry in backend_manifest
            }
            gates.add(
                f"{label}/model_manifest_matches_oracle",
                oracle_shas,
                backend_shas,
                oracle_shas == backend_shas,
            )

        expected_model_sha = EXPECTED_MANIFEST["model.safetensors"]["sha256"]
        actual_model_sha = oracle_shas.get("model.safetensors")
        gates.add(
            "oracle/model_safetensors_sha256",
            expected_model_sha,
            actual_model_sha,
            actual_model_sha == expected_model_sha,
        )


def _check_prompt_identity(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
) -> None:
    """Check prompt, formatted text, token IDs, count, and hash."""
    oracle_prompt = reports["oracle"].get("prompt", {})
    oracle_text = oracle_prompt.get("text")
    oracle_formatted = oracle_prompt.get("formatted")
    oracle_token_ids = oracle_prompt.get("token_ids")
    oracle_n_tokens = oracle_prompt.get("n_tokens")
    oracle_token_hash = oracle_prompt.get("token_hash")

    gates.add(
        "oracle/prompt_text_is_pinned",
        DEFAULT_PROMPT,
        oracle_text,
        oracle_text == DEFAULT_PROMPT,
    )
    gates.add(
        "oracle/prompt_token_ids_are_pinned",
        list(EXPECTED_PROMPT_TOKEN_IDS),
        oracle_token_ids,
        oracle_token_ids == list(EXPECTED_PROMPT_TOKEN_IDS),
    )
    gates.add(
        "oracle/prompt_token_hash_is_pinned",
        EXPECTED_PROMPT_TOKEN_HASH,
        oracle_token_hash,
        oracle_token_hash == EXPECTED_PROMPT_TOKEN_HASH,
    )

    for label, report in reports.items():
        prompt = report.get("prompt", {})
        gates.add(
            f"{label}/prompt_text",
            oracle_text,
            prompt.get("text"),
            prompt.get("text") == oracle_text,
        )
        gates.add(
            f"{label}/prompt_formatted",
            oracle_formatted,
            prompt.get("formatted"),
            prompt.get("formatted") == oracle_formatted,
        )
        gates.add(
            f"{label}/prompt_token_ids",
            oracle_token_ids,
            prompt.get("token_ids"),
            prompt.get("token_ids") == oracle_token_ids,
        )
        gates.add(
            f"{label}/prompt_n_tokens",
            EXPECTED_N_POSITIONS,
            prompt.get("n_tokens"),
            prompt.get("n_tokens") == EXPECTED_N_POSITIONS,
        )
        gates.add(
            f"{label}/prompt_token_hash",
            oracle_token_hash,
            prompt.get("token_hash"),
            prompt.get("token_hash") == oracle_token_hash,
        )

    # Chat-template identity: formatted text must be identical across all
    formatted_values = [
        report.get("prompt", {}).get("formatted")
        for report in reports.values()
    ]
    all_same = len(set(repr(f) for f in formatted_values)) == 1
    gates.add(
        "cross_report/chat_template_identity",
        "all identical",
        f"{len(set(repr(f) for f in formatted_values))} distinct",
        all_same,
    )


def _check_oracle_artifact(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
    oracle_logits_path: Path,
) -> dict[str, Any] | None:
    """Verify oracle raw .npy artifact and cross-check with reports.

    Returns the verified npy manifest or None on failure.
    """
    oracle = reports["oracle"]
    raw_artifact = oracle.get("raw_artifact", {})

    # Verify the actual .npy file against the report's manifest
    if not oracle_logits_path.exists():
        gates.add(
            "oracle/raw_artifact_file_exists",
            True,
            False,
            False,
        )
        return None

    gates.add(
        "oracle/raw_artifact_file_exists",
        True,
        True,
        True,
    )

    try:
        actual_manifest = build_npy_manifest(oracle_logits_path)
    except (ValueError, FileNotFoundError) as exc:
        gates.add(
            "oracle/raw_artifact_manifest_build",
            "valid npy",
            str(exc),
            False,
        )
        return None

    gates.add(
        "oracle/raw_artifact_manifest_build",
        "valid npy",
        "valid npy",
        True,
    )

    # Verify artifact against report manifest
    try:
        verified_arr = verify_npy_artifact(oracle_logits_path, raw_artifact)
        gates.add(
            "oracle/raw_artifact_verify_against_report",
            True,
            True,
            True,
        )
    except (ArtifactVerificationError, KeyError) as exc:
        gates.add(
            "oracle/raw_artifact_verify_against_report",
            True,
            str(exc),
            False,
        )

    # Check basename
    gates.add(
        "oracle/raw_artifact_basename",
        raw_artifact.get("basename"),
        actual_manifest["basename"],
        actual_manifest["basename"] == raw_artifact.get("basename"),
    )

    # Check dtype
    gates.add(
        "oracle/raw_artifact_dtype",
        "<f4",
        actual_manifest["dtype"],
        actual_manifest["dtype"] == "<f4",
    )

    # Check shape
    expected_shape = raw_artifact.get("shape")
    gates.add(
        "oracle/raw_artifact_shape",
        expected_shape,
        actual_manifest["shape"],
        actual_manifest["shape"] == expected_shape,
    )

    # Check file SHA
    gates.add(
        "oracle/raw_artifact_file_sha256",
        raw_artifact.get("file_sha256"),
        actual_manifest["file_sha256"],
        actual_manifest["file_sha256"] == raw_artifact.get("file_sha256"),
    )

    # Check canonical hash matches raw file
    gates.add(
        "oracle/raw_artifact_canonical_sha256",
        raw_artifact.get("canonical_sha256"),
        actual_manifest["canonical_sha256"],
        actual_manifest["canonical_sha256"] == raw_artifact.get("canonical_sha256"),
    )

    # Cross-check: oracle reference.logits_sha256 matches canonical
    oracle_ref = oracle.get("reference", {})
    oracle_logits_sha = oracle_ref.get("logits_sha256")
    gates.add(
        "oracle/reference_logits_sha256_matches_canonical",
        oracle_logits_sha,
        actual_manifest["canonical_sha256"],
        oracle_logits_sha == actual_manifest["canonical_sha256"],
    )

    # Cross-check: each backend's reference.logits_sha256 matches oracle
    for label in ("c", "numba", "mlx"):
        backend_ref = reports[label].get("reference", {})
        backend_logits_sha = backend_ref.get("logits_sha256")
        gates.add(
            f"{label}/reference_logits_sha256_matches_oracle",
            oracle_logits_sha,
            backend_logits_sha,
            backend_logits_sha == oracle_logits_sha,
        )
        backend_artifact = backend_ref.get("artifact")
        gates.add(
            f"{label}/reference_artifact_matches_oracle",
            raw_artifact,
            backend_artifact,
            backend_artifact == raw_artifact,
        )

    return actual_manifest


def _check_backend_names(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
) -> None:
    """Check that backend names are exactly c/numba/mlx."""
    for label, expected_name in EXPECTED_BACKEND_NAMES.items():
        report = reports[label]
        backend = report.get("backend", {})
        actual_name = backend.get("name")
        gates.add(
            f"{label}/backend_name",
            expected_name,
            actual_name,
            actual_name == expected_name,
        )


def _check_reference_canonical_hash_consistency(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
) -> None:
    """All reports must reference the same canonical logits hash."""
    oracle_ref = reports["oracle"].get("reference", {})
    oracle_sha = oracle_ref.get("logits_sha256")

    shas: dict[str, str | None] = {"oracle": oracle_sha}
    for label in ("c", "numba", "mlx"):
        ref = reports[label].get("reference", {})
        shas[label] = ref.get("logits_sha256")

    all_same = len(set(s for s in shas.values() if s is not None)) == 1
    gates.add(
        "cross_report/reference_logits_sha256_consistent",
        "all identical",
        shas,
        all_same and all(s is not None for s in shas.values()),
    )


def _check_positions_and_finiteness(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
) -> None:
    """Check 20 positions, all finite, publication checks, top1 match."""
    for label in ("c", "numba", "mlx"):
        report = reports[label]
        metrics = report.get("metrics", {})

        n_pos = metrics.get("n_positions")
        gates.add(
            f"{label}/metrics_n_positions",
            EXPECTED_N_POSITIONS,
            n_pos,
            n_pos == EXPECTED_N_POSITIONS,
        )

        all_finite_ref = metrics.get("all_finite_ref")
        gates.add(
            f"{label}/metrics_all_finite_ref",
            True,
            all_finite_ref,
            all_finite_ref is True,
        )

        all_finite_pt = metrics.get("all_finite_pt")
        gates.add(
            f"{label}/metrics_all_finite_pt",
            True,
            all_finite_pt,
            all_finite_pt is True,
        )

        all_top1 = metrics.get("all_top1_match")
        gates.add(
            f"{label}/metrics_all_top1_match",
            True,
            all_top1,
            all_top1 is True,
        )

        positions = metrics.get("per_position", [])
        gates.add(
            f"{label}/metrics_per_position_count",
            EXPECTED_N_POSITIONS,
            len(positions),
            len(positions) == EXPECTED_N_POSITIONS,
        )
        for position, item in enumerate(positions):
            gates.add(
                f"{label}/position_{position}_index",
                position,
                item.get("position"),
                item.get("position") == position,
            )
            gates.add(
                f"{label}/position_{position}_finite_ref",
                True,
                item.get("finite_ref"),
                item.get("finite_ref") is True,
            )
            gates.add(
                f"{label}/position_{position}_finite_pt",
                True,
                item.get("finite_pt"),
                item.get("finite_pt") is True,
            )
            gates.add(
                f"{label}/position_{position}_top1_match",
                True,
                item.get("top1_match"),
                item.get("top1_match") is True,
            )

        # Publication threshold checks
        pub = report.get("publication_thresholds", {})
        pub_passed = pub.get("passed")
        gates.add(
            f"{label}/publication_thresholds_passed",
            True,
            pub_passed,
            pub_passed is True,
        )

        checks = pub.get("checks", [])
        expected_check_names = {
            "all_finite_ref",
            "all_finite_pt",
            "cosine_min",
            "pearson_min",
            "all_top1_match",
            "top10_overlap_mean",
        }
        actual_check_names = {check.get("name") for check in checks}
        gates.add(
            f"{label}/publication_check_names",
            sorted(expected_check_names),
            sorted(name for name in actual_check_names if name is not None),
            actual_check_names == expected_check_names,
        )
        for check in checks:
            check_name = check.get("name", "unknown")
            check_passed = check.get("passed")
            gates.add(
                f"{label}/pub_check_{check_name}",
                True,
                check_passed,
                check_passed is True,
            )


def _check_layers_and_chunks(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
) -> None:
    """Check each backend completes 35 layers and expected chunks."""
    for label in ("c", "numba", "mlx"):
        report = reports[label]
        pt = report.get("pytensor", {})

        layers = pt.get("layers_completed")
        gates.add(
            f"{label}/pytensor_layers_completed",
            EXPECTED_N_LAYERS,
            layers,
            layers == EXPECTED_N_LAYERS,
        )

        chunks = pt.get("chunks_processed")
        # Derive expected chunks from vocab_size if available
        oracle_ref = reports["oracle"].get("reference", {})
        vocab_size = oracle_ref.get("vocab_size")
        if vocab_size is not None:
            expected_chunks = _expected_n_chunks(vocab_size, EXPECTED_CHUNK_SIZE)
        else:
            expected_chunks = None

        gates.add(
            f"{label}/pytensor_chunks_processed",
            expected_chunks,
            chunks,
            chunks == expected_chunks if expected_chunks is not None else False,
        )

        layer_types = pt.get("layer_types_used")
        gates.add(
            f"{label}/pytensor_layer_types",
            list(EXPECTED_LAYER_TYPES),
            layer_types,
            layer_types == list(EXPECTED_LAYER_TYPES),
        )
        rope_bases = pt.get("rope_bases_used")
        gates.add(
            f"{label}/pytensor_rope_bases",
            list(EXPECTED_ROPE_BASES),
            rope_bases,
            rope_bases == list(EXPECTED_ROPE_BASES),
        )
        sparse_layers = pt.get("sparse_layers_used")
        gates.add(
            f"{label}/pytensor_sparse_layers",
            list(EXPECTED_SPARSE_LAYERS),
            sparse_layers,
            sparse_layers == list(EXPECTED_SPARSE_LAYERS),
        )

        timing_per_layer = report.get("timing", {}).get("pt_per_layer_s")
        gates.add(
            f"{label}/timing_per_layer_count",
            EXPECTED_N_LAYERS,
            None if not isinstance(timing_per_layer, list) else len(timing_per_layer),
            isinstance(timing_per_layer, list)
            and len(timing_per_layer) == EXPECTED_N_LAYERS,
        )
        pytensor_per_layer = pt.get("per_layer_s")
        gates.add(
            f"{label}/pytensor_per_layer_count",
            EXPECTED_N_LAYERS,
            None if not isinstance(pytensor_per_layer, list) else len(pytensor_per_layer),
            isinstance(pytensor_per_layer, list)
            and len(pytensor_per_layer) == EXPECTED_N_LAYERS,
        )
        logits_sha = pt.get("logits_sha256")
        valid_sha = (
            isinstance(logits_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", logits_sha) is not None
        )
        gates.add(
            f"{label}/pytensor_logits_sha256",
            "64 lowercase hexadecimal characters",
            logits_sha,
            valid_sha,
        )


def _check_mlx_stages(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
) -> None:
    """Check MLX has configuration-derived exact ordered stages."""
    mlx_report = reports["mlx"]
    pt = mlx_report.get("pytensor", {})
    stages = pt.get("mlx_stages", [])
    stage_count = pt.get("stage_count")

    oracle_ref = reports["oracle"].get("reference", {})
    vocab_size = oracle_ref.get("vocab_size")

    gates.add(
        "oracle/reference_vocab_size",
        EXPECTED_VOCAB_SIZE,
        vocab_size,
        vocab_size == EXPECTED_VOCAB_SIZE,
    )

    if vocab_size is None:
        gates.add(
            "mlx/stages_vocab_size_available",
            True,
            False,
            False,
        )
        return

    gates.add(
        "mlx/stages_vocab_size_available",
        True,
        True,
        True,
    )

    expected_labels = _expected_stage_labels(
        EXPECTED_N_LAYERS, vocab_size, EXPECTED_CHUNK_SIZE,
    )
    expected_count = len(expected_labels)

    # Stage count
    gates.add(
        "mlx/stages_count",
        expected_count,
        stage_count,
        stage_count == expected_count,
    )

    # Stage list length
    gates.add(
        "mlx/stages_list_length",
        expected_count,
        len(stages),
        len(stages) == expected_count,
    )

    # Stage label order
    actual_labels = [s.get("label") for s in stages]
    gates.add(
        "mlx/stages_label_order",
        expected_labels,
        actual_labels,
        actual_labels == expected_labels,
    )

    # Initial/per-layer labels present
    has_initial = any(s.get("label") == "initial_projections" for s in stages)
    gates.add("mlx/stages_has_initial_projections", True, has_initial, has_initial)

    has_per_layer = any(s.get("label") == "per_layer_projection" for s in stages)
    gates.add("mlx/stages_has_per_layer_projection", True, has_per_layer, has_per_layer)

    # Layer labels 0..34
    for i in range(EXPECTED_N_LAYERS):
        has_layer = any(s.get("label") == f"layer_{i}" for s in stages)
        gates.add(f"mlx/stages_has_layer_{i}", True, has_layer, has_layer)

    # Final label
    has_final = any(s.get("label") == "final_unembed" for s in stages)
    gates.add("mlx/stages_has_final_unembed", True, has_final, has_final)

    # Vocab chunk ranges
    chunk_labels = [s.get("label", "") for s in stages if "vocab_chunk" in s.get("label", "")]
    expected_chunk_labels = [l for l in expected_labels if l.startswith("vocab_chunk_")]
    gates.add(
        "mlx/stages_vocab_chunk_ranges",
        expected_chunk_labels,
        chunk_labels,
        chunk_labels == expected_chunk_labels,
    )

    # Positive timing totals
    for stage in stages:
        eval_s = stage.get("eval_s", 0.0)
        host_copy_s = stage.get("host_copy_s", 0.0)
        label = stage.get("label", "unknown")
        gates.add(
            f"mlx/stage_{label}_eval_s_nonneg",
            ">= 0",
            eval_s,
            isinstance(eval_s, (int, float)) and eval_s >= 0,
        )
        gates.add(
            f"mlx/stage_{label}_host_copy_s_nonneg",
            ">= 0",
            host_copy_s,
            isinstance(host_copy_s, (int, float)) and host_copy_s >= 0,
        )


def _check_c_numba_no_backend_mlx(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
) -> None:
    """C and Numba reports must not have backend_mlx in memory."""
    for label in ("c", "numba"):
        memory = reports[label].get("memory", {})
        has_backend_mlx = "backend_mlx" in memory
        gates.add(
            f"{label}/memory_no_backend_mlx",
            False,
            has_backend_mlx,
            not has_backend_mlx,
        )


def _check_mlx_separate_allocator_memory(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
) -> None:
    """MLX report must have separate allocator memory section."""
    mlx_memory = reports["mlx"].get("memory", {})
    has_backend_mlx = "backend_mlx" in mlx_memory
    gates.add(
        "mlx/memory_has_backend_mlx",
        True,
        has_backend_mlx,
        has_backend_mlx,
    )

    if has_backend_mlx:
        mlx_mem = mlx_memory["backend_mlx"]
        api = mlx_mem.get("api")
        gates.add(
            "mlx/memory_backend_mlx_api_list",
            "non-empty list",
            api,
            isinstance(api, list) and bool(api),
        )
        for key in ("version", "baseline_mib", "peak_mib", "current_mib"):
            val = mlx_mem.get(key)
            gates.add(
                f"mlx/memory_backend_mlx_{key}_present",
                "present",
                val,
                val is not None,
            )
        for key in ("baseline_bytes", "peak_bytes", "current_bytes", "cache_bytes"):
            val = mlx_mem.get(key)
            gates.add(
                f"mlx/memory_backend_mlx_{key}",
                "non-negative integer",
                val,
                isinstance(val, int) and val >= 0,
            )


def _check_oracle_separate_mlx_memory(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
) -> None:
    """Oracle report must have separate oracle_mlx memory section."""
    oracle_memory = reports["oracle"].get("memory", {})
    has_oracle_mlx = "oracle_mlx" in oracle_memory
    gates.add(
        "oracle/memory_has_oracle_mlx",
        True,
        has_oracle_mlx,
        has_oracle_mlx,
    )

    if has_oracle_mlx:
        oracle_mlx = oracle_memory["oracle_mlx"]
        api = oracle_mlx.get("api")
        gates.add(
            "oracle/memory_oracle_mlx_api_list",
            "non-empty list",
            api,
            isinstance(api, list) and bool(api),
        )
        for key in ("version", "baseline_mib", "current_mib", "peak_mib"):
            val = oracle_mlx.get(key)
            gates.add(
                f"oracle/memory_oracle_mlx_{key}_present",
                "present",
                val,
                val is not None,
            )
        for key in ("baseline_bytes", "current_bytes", "peak_bytes"):
            val = oracle_mlx.get(key)
            gates.add(
                f"oracle/memory_oracle_mlx_{key}",
                "non-negative integer",
                val,
                isinstance(val, int) and val >= 0,
            )


def _check_whole_process_rss(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
) -> None:
    """All reports must have whole-process RSS memory field."""
    for label, report in reports.items():
        memory = report.get("memory", {})
        rss_key = "whole_process_peak_rss_mib"
        has_rss = rss_key in memory
        gates.add(
            f"{label}/memory_has_whole_process_peak_rss_mib",
            True,
            has_rss,
            has_rss,
        )
        if has_rss:
            rss_val = memory[rss_key]
            gates.add(
                f"{label}/memory_whole_process_peak_rss_mib_positive",
                "> 0",
                rss_val,
                isinstance(rss_val, (int, float)) and rss_val > 0,
            )


def _check_no_absolute_paths(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
) -> None:
    """No absolute snapshot/temp paths in published command/report fields."""
    for label, report in reports.items():
        # Check command field
        if label == "oracle":
            command = report.get("provenance", {}).get("command", [])
        else:
            command = report.get("command", [])

        command_str = json.dumps(command)
        abs_paths = _contains_absolute_path(command)
        gates.add(
            f"{label}/command_no_absolute_paths",
            "no absolute paths",
            abs_paths if abs_paths else "clean",
            len(abs_paths) == 0,
        )

        # Check report_path or output_path fields if present
        for path_key in ("report_path", "output_path", "logits_path"):
            if path_key in report:
                path_val = report[path_key]
                if isinstance(path_val, str) and path_val.startswith("/"):
                    gates.add(
                        f"{label}/{path_key}_no_absolute_path",
                        "relative or basename",
                        path_val,
                        False,
                    )


def _check_package_versions_consistent(
    gates: GateCollector,
    reports: dict[str, dict[str, Any]],
) -> None:
    """Package versions must be consistent across reports."""
    oracle_versions = reports["oracle"].get("versions", {})

    for key, expected in EXPECTED_VERSIONS.items():
        actual = oracle_versions.get(key)
        gates.add(
            f"oracle/versions_{key}_is_pinned",
            expected,
            actual,
            actual == expected,
        )

    for label in ("c", "numba", "mlx"):
        backend_versions = reports[label].get("versions", {})

        # Check shared keys match
        shared_keys = set(oracle_versions.keys()) & set(backend_versions.keys())
        for key in sorted(shared_keys):
            oracle_val = oracle_versions[key]
            backend_val = backend_versions[key]
            gates.add(
                f"{label}/versions_{key}_matches_oracle",
                oracle_val,
                backend_val,
                oracle_val == backend_val,
            )


# ---------------------------------------------------------------------------
# Comparison rows
# ---------------------------------------------------------------------------


def _build_comparison_rows(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build compact comparison rows for timing, memory, and aggregate metrics.

    Does not invent or round source values — reports them as-is.
    """
    timing_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    # Timing comparison
    timing_keys = [
        "tokenize_s", "ref_load_s", "ref_forward_s", "ref_sync_s",
        "pt_compile_s", "pt_total_s",
        "pt_embed_s", "pt_ple_s", "pt_initial_s",
        "pt_per_layer_proj_s", "pt_final_s", "pt_logits_s",
    ]
    for key in timing_keys:
        row: dict[str, Any] = {"metric": key}
        for label in ("c", "numba", "mlx"):
            timing = reports[label].get("timing", {})
            val = timing.get(key)
            if val is not None:
                row[label] = val
        if len(row) > 1:
            timing_rows.append(row)

    # Per-layer timing summary
    for label in ("c", "numba", "mlx"):
        timing = reports[label].get("timing", {})
        per_layer = timing.get("pt_per_layer_s")
        if per_layer and isinstance(per_layer, list) and len(per_layer) > 0:
            timing_rows.append({
                "metric": "pt_per_layer_s_mean",
                label: sum(per_layer) / len(per_layer),
            })
            timing_rows.append({
                "metric": "pt_per_layer_s_max",
                label: max(per_layer),
            })

    # Memory comparison
    memory_keys = ["whole_process_peak_rss_mib"]
    for key in memory_keys:
        row = {"metric": key}
        for label, report in reports.items():
            memory = report.get("memory", {})
            val = memory.get(key)
            if val is not None:
                row[label] = val
        if len(row) > 1:
            memory_rows.append(row)

    # MLX-specific memory
    for label in ("oracle", "mlx"):
        memory = reports[label].get("memory", {})
        if label == "oracle":
            mlx_mem = memory.get("oracle_mlx", {})
        else:
            mlx_mem = memory.get("backend_mlx", {})
        if mlx_mem:
            for key in ("peak_mib", "current_mib", "baseline_mib"):
                val = mlx_mem.get(key)
                if val is not None:
                    memory_rows.append({
                        "metric": f"{label}_mlx_{key}",
                        label: val,
                    })

    # Aggregate metrics comparison
    agg_keys = [
        "cosine_min", "cosine_mean",
        "pearson_min", "pearson_mean",
        "max_abs_diff_max", "max_abs_diff_mean",
        "mean_abs_diff_max", "mean_abs_diff_mean",
        "rmse_max", "rmse_mean",
        "top10_overlap_min", "top10_overlap_mean",
    ]
    for key in agg_keys:
        row = {"metric": key}
        for label in ("c", "numba", "mlx"):
            metrics = reports[label].get("metrics", {})
            agg = metrics.get("aggregate", {})
            val = agg.get(key)
            if val is not None:
                row[label] = val
        if len(row) > 1:
            aggregate_rows.append(row)

    return {
        "timing": timing_rows,
        "memory": memory_rows,
        "aggregate": aggregate_rows,
    }


# ---------------------------------------------------------------------------
# Atomic output
# ---------------------------------------------------------------------------


def atomic_write_validation_report(
    report: dict[str, Any],
    dest: Path,
) -> None:
    """Write the validation report atomically.

    Uses the shared evidence atomic writer for consistency.
    """
    from cetagostini.utils.pytensor.evidence import atomic_write_json

    atomic_write_json(report, Path(dest))


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------


def validate_reports(
    *,
    run_id: str,
    oracle_path: Path,
    oracle_logits_path: Path,
    c_path: Path,
    numba_path: Path,
    mlx_path: Path,
) -> dict[str, Any]:
    """Run the full cross-report validation pipeline.

    Parameters
    ----------
    run_id : str
        Expected run identifier.
    oracle_path : Path
        Path to the oracle JSON report.
    oracle_logits_path : Path
        Path to the oracle logits ``.npy`` artifact.
    c_path : Path
        Path to the C backend JSON report.
    numba_path : Path
        Path to the Numba backend JSON report.
    mlx_path : Path
        Path to the MLX backend JSON report.

    Returns
    -------
    dict
        Validation report with gates, comparison, and pass/fail status.
    """
    gates = GateCollector()

    # Phase 1: Load all reports
    reports: dict[str, dict[str, Any]] = {}
    load_errors: list[str] = []
    for label, path in [
        ("oracle", oracle_path),
        ("c", c_path),
        ("numba", numba_path),
        ("mlx", mlx_path),
    ]:
        try:
            reports[label] = load_strict_json(path)
        except (FileNotFoundError, ValueError) as exc:
            load_errors.append(f"{label}: {exc}")

    if load_errors:
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "run_id": run_id,
            "load_errors": load_errors,
            "gates": [],
            "all_passed": False,
            "comparison": {},
        }

    # Phase 2: Common schema and run ID
    _check_common_schema_and_run_id(gates, reports, run_id)

    # Phase 3: Provenance
    _check_provenance(gates, reports)

    # Phase 4: Model identity
    _check_model_identity(gates, reports)

    # Phase 5: Prompt identity
    _check_prompt_identity(gates, reports)

    # Phase 6: Oracle artifact
    _check_oracle_artifact(gates, reports, oracle_logits_path)

    # Phase 7: Backend names
    _check_backend_names(gates, reports)

    # Phase 8: Reference canonical hash consistency
    _check_reference_canonical_hash_consistency(gates, reports)

    # Phase 9: Positions, finiteness, publication, top1
    _check_positions_and_finiteness(gates, reports)

    # Phase 10: Layers and chunks
    _check_layers_and_chunks(gates, reports)

    # Phase 11: MLX stages
    _check_mlx_stages(gates, reports)

    # Phase 12: C/Numba no backend_mlx
    _check_c_numba_no_backend_mlx(gates, reports)

    # Phase 13: MLX separate allocator memory
    _check_mlx_separate_allocator_memory(gates, reports)

    # Phase 14: Oracle separate mlx memory
    _check_oracle_separate_mlx_memory(gates, reports)

    # Phase 15: Whole-process RSS
    _check_whole_process_rss(gates, reports)

    # Phase 16: No absolute paths
    _check_no_absolute_paths(gates, reports)

    # Phase 17: Package versions
    _check_package_versions_consistent(gates, reports)

    # Phase 18: Comparison rows
    comparison = _build_comparison_rows(reports)

    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "run_id": run_id,
        "gates": gates.gates,
        "all_passed": gates.all_passed,
        "n_gates": len(gates.gates),
        "n_passed": sum(1 for g in gates.gates if g["passed"]),
        "n_failed": sum(1 for g in gates.gates if not g["passed"]),
        "failed_gates": gates.failed,
        "comparison": comparison,
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
        description="Cross-validate Gemma 3n oracle and PyTensor backend reports.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        type=str,
        help="Expected run identifier.",
    )
    parser.add_argument(
        "--oracle",
        required=True,
        type=Path,
        help="Path to the oracle JSON report.",
    )
    parser.add_argument(
        "--oracle-logits",
        required=True,
        type=Path,
        help="Path to the oracle logits .npy artifact.",
    )
    parser.add_argument(
        "--c",
        required=True,
        type=Path,
        dest="c_report",
        help="Path to the C backend JSON report.",
    )
    parser.add_argument(
        "--numba",
        required=True,
        type=Path,
        help="Path to the Numba backend JSON report.",
    )
    parser.add_argument(
        "--mlx",
        required=True,
        type=Path,
        help="Path to the MLX backend JSON report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the validation report JSON output.",
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
        Exit code (0 if all gates pass, 1 otherwise).
    """
    args = parse_args(argv)

    result = validate_reports(
        run_id=args.run_id,
        oracle_path=args.oracle,
        oracle_logits_path=args.oracle_logits,
        c_path=args.c_report,
        numba_path=args.numba,
        mlx_path=args.mlx,
    )

    if args.output:
        atomic_write_validation_report(result, args.output)
        print(f"Validation report written to {args.output}")

    n_gates = result.get("n_gates", 0)
    n_passed = result.get("n_passed", 0)
    n_failed = result.get("n_failed", 0)
    all_passed = result.get("all_passed", False)

    print(f"Gates: {n_passed}/{n_gates} passed, {n_failed} failed")

    if not all_passed:
        for gate in result.get("failed_gates", []):
            print(
                f"  FAIL: {gate['name']}: "
                f"expected={gate['expected']!r}, actual={gate['actual']!r}"
            )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
