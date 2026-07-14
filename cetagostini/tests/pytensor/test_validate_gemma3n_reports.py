"""Tests for validate_gemma3n_reports cross-report validator.

Covers:
- Complete passing four-report fixture matching actual schemas
- One mutation test per gate family (run ID, provenance, model SHA,
  prompt/token, artifact SHA/shape, wrong/missing backend, nonfinite,
  thresholds, top1, layers/chunks, stage count/order/ranges, memory
  leakage, package mismatch, absolute path)
- CLI success/failure
- Strict JSON / atomic output
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cetagostini.utils.pytensor import validate_gemma3n_reports as validator_module

from cetagostini.utils.pytensor.evidence import (
    GEMMA3N_ORACLE_SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_npy,
    build_npy_manifest,
)
from cetagostini.utils.pytensor.validate_gemma3n_reports import (
    VALIDATION_SCHEMA_VERSION,
    GateCollector,
    _check_backend_names,
    _check_c_numba_no_backend_mlx,
    _check_common_schema_and_run_id,
    _check_layers_and_chunks,
    _check_mlx_separate_allocator_memory,
    _check_mlx_stages,
    _check_model_identity,
    _check_no_absolute_paths,
    _check_oracle_artifact,
    _check_oracle_separate_mlx_memory,
    _check_package_versions_consistent,
    _check_positions_and_finiteness,
    _check_prompt_identity,
    _check_provenance,
    _check_reference_canonical_hash_consistency,
    _check_whole_process_rss,
    _contains_absolute_path,
    _expected_n_chunks,
    _expected_n_stages,
    _expected_stage_labels,
    load_strict_json,
    main,
    parse_args,
    validate_reports,
)


# ---------------------------------------------------------------------------
# Constants for fixtures
# ---------------------------------------------------------------------------

RUN_ID = "validation-run-001"
VOCAB_SIZE = 128
SEQ_LEN = 20
N_LAYERS = 35
CHUNK_SIZE = 4096
GIT_COMMIT = "a" * 40
IMPL_MANIFEST_SHA = "b" * 64
ENV_YML_SHA = "c" * 64

EXPECTED_REPO = "mlx-community/gemma-3n-E4B-it-lm-4bit"
EXPECTED_REVISION = "00b5ecdc79ba872a9b4cd32f4327e263bab5936c"
EXPECTED_MODEL_TYPE = "gemma3n"
EXPECTED_ARCHITECTURE = "Gemma3nForConditionalGeneration"
EXPECTED_BITS = 4
EXPECTED_GROUP_SIZE = 64

PROMPT_TEXT = "Explain in two sentences what a symbolic tensor graph is."
FORMATTED_PROMPT = "<bos><start_of_turn>user\nExplain in two sentences what a symbolic tensor graph is.<end_of_turn>\n<start_of_turn>model\n"
TOKEN_IDS = [
    2, 105, 2364, 107, 155122, 528, 1156, 23974, 1144, 496,
    42988, 18441, 3753, 563, 236761, 106, 107, 105, 4368, 107,
]
TOKEN_HASH = "bec5926dff4bdc1ae70cb754a2078ad616f830aa1a31fcd6fdc5b72512299545"


@pytest.fixture(autouse=True)
def _use_small_vocab_for_unit_fixtures(monkeypatch):
    """Keep mutation tests small while production remains pinned to 262400."""
    monkeypatch.setattr(validator_module, "EXPECTED_VOCAB_SIZE", VOCAB_SIZE)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_versions() -> dict[str, str]:
    return {
        "python": "3.13.14",
        "numpy": "2.4.6",
        "pytensor": "3.1.2",
        "numba": "0.65.1",
        "mlx": "0.32.0",
        "mlx_lm": "0.31.3",
        "transformers": "5.12.1",
        "pytensor_ml": "0.0.5.dev24+gf6ecf81d5",
    }


def _make_model_manifest() -> list[dict[str, Any]]:
    return [
        {"name": "config.json", "size_bytes": 124435, "sha256": "d" * 64},
        {
            "name": "model.safetensors",
            "size_bytes": 3863598176,
            "sha256": "94401d496aa8a68c0d853adcbb0acea9635e71e390afeb24678acd0dbf530007",
        },
        {"name": "tokenizer.json", "size_bytes": 33442553, "sha256": "f" * 64},
        {"name": "tokenizer_config.json", "size_bytes": 1202305, "sha256": "1" * 64},
        {"name": "chat_template.jinja", "size_bytes": 1626, "sha256": "2" * 64},
    ]


def _make_provenance() -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "schema_version": GEMMA3N_ORACLE_SCHEMA_VERSION,
        "implementation": {
            "git_commit": GIT_COMMIT,
            "git_clean": True,
            "environment_yml_sha256": ENV_YML_SHA,
            "source_hashes": [
                {
                    "path": "cetagostini/utils/pytensor/run_gemma_mlx.py",
                    "sha256": "3" * 64,
                },
                {
                    "path": "cetagostini/utils/pytensor/evidence.py",
                    "sha256": "4" * 64,
                },
                {
                    "path": "cetagostini/utils/pytensor/provenance.py",
                    "sha256": "5" * 64,
                },
            ],
            "implementation_manifest_sha256": IMPL_MANIFEST_SHA,
            "timestamp_utc": "2026-07-13T00:00:00+00:00",
            "python_executable": "/usr/bin/python3",
            "environment": {
                "python_version": "3.13.14",
                "python_executable": "/usr/bin/python3",
                "platform_system": "Darwin",
                "platform_machine": "arm64",
                "platform_release": "24.0.0",
                "platform_version": "Darwin Kernel Version 24.0.0",
            },
            "package_versions": _make_versions(),
            "module_paths": {
                "numpy": "/usr/lib/python3/numpy/__init__.py",
                "pytensor": "/usr/lib/python3/pytensor/__init__.py",
            },
        },
        "command": [
            "run_gemma_mlx.py",
            "--snapshot",
            EXPECTED_REVISION,
            "--run-id",
            RUN_ID,
            "--logits-output",
            "oracle_logits.npy",
            "--output",
            "oracle.json",
        ],
    }


def _make_oracle_report(
    logits_arr: np.ndarray,
    npy_manifest: dict[str, Any],
) -> dict[str, Any]:
    logits_sha = hashlib.sha256(logits_arr.tobytes()).hexdigest()
    top1_id = int(np.argmax(logits_arr[0, -1]))

    return {
        "schema_version": GEMMA3N_ORACLE_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "model": {
            "repo": EXPECTED_REPO,
            "revision": EXPECTED_REVISION,
            "model_type": EXPECTED_MODEL_TYPE,
            "architecture": EXPECTED_ARCHITECTURE,
            "quantization": {
                "bits": EXPECTED_BITS,
                "group_size": EXPECTED_GROUP_SIZE,
            },
            "manifest": _make_model_manifest(),
        },
        "prompt": {
            "text": PROMPT_TEXT,
            "formatted": FORMATTED_PROMPT,
            "token_ids": TOKEN_IDS,
            "n_tokens": len(TOKEN_IDS),
            "token_hash": TOKEN_HASH,
        },
        "reference": {
            "shape": list(logits_arr.shape),
            "logits_sha256": logits_sha,
            "top1_id": top1_id,
            "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN,
        },
        "raw_artifact": npy_manifest,
        "runtime": "mlx_lm_native",
        "versions": _make_versions(),
        "device": "apple_silicon",
        "timing": {
            "ref_load_s": 2.345,
            "ref_forward_s": 0.678,
            "ref_sync_s": 0.123,
        },
        "memory": {
            "whole_process_peak_rss_mib": 8192.0,
            "oracle_mlx": {
                "api": ["mx.get_peak_memory", "mx.get_active_memory"],
                "version": "0.32.0",
                "baseline_bytes": 0,
                "current_bytes": 4096 * 1024 * 1024,
                "peak_bytes": 6144 * 1024 * 1024,
                "baseline_mib": 0.0,
                "current_mib": 4096.0,
                "peak_mib": 6144.0,
            },
        },
        "provenance": _make_provenance(),
    }


def _make_mlx_stages() -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    stages.append({"label": "initial_projections", "eval_s": 0.001, "host_copy_s": 0.0})
    stages.append({"label": "per_layer_projection", "eval_s": 0.002, "host_copy_s": 0.0})
    for i in range(N_LAYERS):
        stages.append({"label": f"layer_{i}", "eval_s": 0.01, "host_copy_s": 0.0})
    stages.append({"label": "final_unembed", "eval_s": 0.003, "host_copy_s": 0.0})
    for start in range(0, VOCAB_SIZE, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, VOCAB_SIZE)
        stages.append({
            "label": f"vocab_chunk_{start}_{end}",
            "eval_s": 0.005,
            "host_copy_s": 0.002,
        })
    return stages


def _make_per_layer_times() -> list[float]:
    return [round(0.5 + i * 0.01, 4) for i in range(N_LAYERS)]


def _make_metrics(logits_ref: np.ndarray, logits_pt: np.ndarray) -> dict[str, Any]:
    """Build a passing metrics section from two logit arrays."""
    T = logits_ref.shape[1]
    per_position: list[dict[str, Any]] = []
    all_cosine: list[float] = []
    all_pearson: list[float] = []
    all_top10: list[int] = []
    all_max_abs: list[float] = []
    all_mean_abs: list[float] = []
    all_rmse: list[float] = []

    for pos in range(T):
        r = logits_ref[0, pos].astype(np.float64)
        p = logits_pt[0, pos].astype(np.float64)
        abs_diff = np.abs(r - p)
        max_abs = float(np.max(abs_diff))
        mean_abs = float(np.mean(abs_diff))
        rmse = float(np.sqrt(np.mean(abs_diff ** 2)))
        r_norm = float(np.linalg.norm(r))
        p_norm = float(np.linalg.norm(p))
        cosine = float(np.clip(np.dot(r, p) / (r_norm * p_norm), -1, 1))
        r_std = float(np.std(r))
        p_std = float(np.std(p))
        pearson = float(np.corrcoef(r, p)[0, 1]) if r_std > 0 and p_std > 0 else 0.0
        ref_top10 = set(np.argsort(r)[-10:].tolist())
        pt_top10 = set(np.argsort(p)[-10:].tolist())
        overlap = len(ref_top10 & pt_top10)
        top1_match = int(np.argmax(r)) == int(np.argmax(p))

        per_position.append({
            "position": pos,
            "finite_ref": True,
            "finite_pt": True,
            "max_abs_diff": round(max_abs, 6),
            "mean_abs_diff": round(mean_abs, 6),
            "rmse": round(rmse, 6),
            "cosine": round(cosine, 6),
            "pearson": round(pearson, 6),
            "top10_overlap": overlap,
            "top1_match": top1_match,
        })
        all_cosine.append(cosine)
        all_pearson.append(pearson)
        all_top10.append(overlap)
        all_max_abs.append(max_abs)
        all_mean_abs.append(mean_abs)
        all_rmse.append(rmse)

    final_ref = int(np.argmax(logits_ref[0, -1]))
    final_pt = int(np.argmax(logits_pt[0, -1]))

    return {
        "all_finite_ref": True,
        "all_finite_pt": True,
        "n_positions": T,
        "per_position": per_position,
        "aggregate": {
            "max_abs_diff_max": max(all_max_abs),
            "max_abs_diff_mean": float(np.mean(all_max_abs)),
            "mean_abs_diff_max": max(all_mean_abs),
            "mean_abs_diff_mean": float(np.mean(all_mean_abs)),
            "rmse_max": max(all_rmse),
            "rmse_mean": float(np.mean(all_rmse)),
            "cosine_min": min(all_cosine),
            "cosine_mean": float(np.mean(all_cosine)),
            "pearson_min": min(all_pearson),
            "pearson_mean": float(np.mean(all_pearson)),
            "top10_overlap_min": min(all_top10),
            "top10_overlap_mean": float(np.mean(all_top10)),
        },
        "final_top1_ref": final_ref,
        "final_top1_pt": final_pt,
        "final_top1_match": final_ref == final_pt,
        "all_top1_match": True,
    }


def _make_publication_thresholds(metrics: dict[str, Any]) -> dict[str, Any]:
    agg = metrics.get("aggregate", {})
    checks: list[dict[str, Any]] = [
        {"name": "all_finite_ref", "threshold": True, "actual": True, "passed": True},
        {"name": "all_finite_pt", "threshold": True, "actual": True, "passed": True},
        {
            "name": "cosine_min",
            "threshold": 0.99,
            "actual": agg.get("cosine_min", 1.0),
            "passed": agg.get("cosine_min", 1.0) >= 0.99,
        },
        {
            "name": "pearson_min",
            "threshold": 0.99,
            "actual": agg.get("pearson_min", 1.0),
            "passed": agg.get("pearson_min", 1.0) >= 0.99,
        },
        {
            "name": "all_top1_match",
            "threshold": True,
            "actual": True,
            "passed": True,
        },
        {
            "name": "top10_overlap_mean",
            "threshold": 8.0,
            "actual": agg.get("top10_overlap_mean", 10.0),
            "passed": agg.get("top10_overlap_mean", 10.0) >= 8.0,
        },
    ]
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def _make_backend_report(
    backend_name: str,
    logits_ref: np.ndarray,
    logits_pt: np.ndarray,
    *,
    include_provenance: bool = False,
    artifact_manifest: dict[str, Any] | None = None,
    reference_artifact_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a backend report matching the actual sanitize_result schema."""
    logits_sha = hashlib.sha256(logits_ref.tobytes()).hexdigest()
    pt_logits_sha = hashlib.sha256(logits_pt.tobytes()).hexdigest()
    metrics = _make_metrics(logits_ref, logits_pt)
    pub_thresholds = _make_publication_thresholds(metrics)
    stages = _make_mlx_stages() if backend_name == "mlx" else []
    per_layer_s = _make_per_layer_times()

    backend_info = {
        "c": {"name": "c", "linker": "cvm", "mode": "o2"},
        "numba": {"name": "numba", "linker": "numba", "mode": "fast_compile"},
        "mlx": {"name": "mlx", "linker": "mlx", "mode": "fast_run+mlx"},
    }[backend_name]

    memory: dict[str, Any] = {
        "whole_process_peak_rss_mib": 10240.0,
    }
    if backend_name == "mlx":
        memory["backend_mlx"] = {
            "api": ["mx.get_peak_memory", "mx.get_active_memory"],
            "version": "0.32.0",
            "baseline_bytes": 2048 * 1024 * 1024,
            "current_bytes": 3072 * 1024 * 1024,
            "peak_bytes": 5120 * 1024 * 1024,
            "cache_bytes": 1024 * 1024 * 1024,
            "baseline_mib": 2048.0,
            "peak_mib": 5120.0,
            "current_mib": 3072.0,
            "cache_mib": 1024.0,
        }

    report: dict[str, Any] = {
        "schema_version": GEMMA3N_ORACLE_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "command": [
            "run_gemma3n_pytensor.py",
            "run",
            "--snapshot",
            EXPECTED_REVISION,
            "--run-id",
            RUN_ID,
            "--backend",
            backend_name,
        ],
        "model": {
            "repo": EXPECTED_REPO,
            "revision": EXPECTED_REVISION,
            "model_type": EXPECTED_MODEL_TYPE,
            "architecture": EXPECTED_ARCHITECTURE,
            "quantization": {
                "bits": EXPECTED_BITS,
                "group_size": EXPECTED_GROUP_SIZE,
            },
        },
        "prompt": {
            "text": PROMPT_TEXT,
            "formatted": FORMATTED_PROMPT,
            "token_ids": TOKEN_IDS,
            "n_tokens": len(TOKEN_IDS),
            "token_hash": TOKEN_HASH,
        },
        "backend": backend_info,
        "versions": _make_versions(),
        "optional_statuses": {
            "jax_installed": False,
            "jax_version": None,
            "pytensor_ml_installed": True,
            "pytensor_ml_version": "0.3.0",
        },
        "device": "apple_silicon",
        "file_manifest": _make_model_manifest(),
        "reference": {
            "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN,
            "load_s": 2.345,
            "forward_s": 0.678,
            "sync_s": 0.123,
            "peak_memory_mib": 6144.0,
            "logits_sha256": logits_sha,
            "artifact": reference_artifact_manifest,
        },
        "timing": {
            "tokenize_s": 0.045,
            "ref_load_s": 2.345,
            "ref_forward_s": 0.678,
            "ref_sync_s": 0.123,
            "pt_compile_s": 3.456,
            "pt_total_s": 12.345,
            "pt_embed_s": 0.012,
            "pt_ple_s": 0.008,
            "pt_initial_s": 0.234,
            "pt_per_layer_proj_s": 0.056,
            "pt_per_layer_s": per_layer_s,
            "pt_final_s": 0.123,
            "pt_logits_s": 1.234,
        },
        "memory": memory,
        "pytensor": {
            "layers_completed": N_LAYERS,
            "embed_s": 0.012,
            "ple_s": 0.008,
            "global_load_s": 0.034,
            "initial_s": 0.234,
            "per_layer_proj_s": 0.056,
            "per_layer_s": per_layer_s,
            "final_s": 0.123,
            "logits_s": 1.234,
            "total_s": 12.345,
            "layer_types_used": [
                "full_attention" if (index + 1) % 5 == 0 else "sliding_attention"
                for index in range(N_LAYERS)
            ],
            "rope_bases_used": [
                "1M" if (index + 1) % 5 == 0 else "10K"
                for index in range(N_LAYERS)
            ],
            "sparse_layers_used": list(range(10)),
            "chunks_processed": _expected_n_chunks(VOCAB_SIZE, CHUNK_SIZE),
            "mlx_eval_s": 0.5 if backend_name == "mlx" else 0.0,
            "mlx_host_copy_s": 0.3 if backend_name == "mlx" else 0.0,
            "mlx_stages": stages,
            "stage_count": len(stages),
            "logits_sha256": pt_logits_sha,
            "artifact": artifact_manifest,
        },
        "metrics": metrics,
        "publication_thresholds": pub_thresholds,
    }

    if include_provenance:
        report["provenance"] = _make_provenance()

    return report


def _make_all_fixtures(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, np.ndarray]:
    """Create all four report files and the logits .npy.

    Returns paths: (oracle_json, logits_npy, c_json, numba_json, mlx_json, logits_arr).
    """
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((1, SEQ_LEN, VOCAB_SIZE)).astype(np.float32)
    # Make PT logits very close to oracle (small perturbation)
    pt_logits = logits + rng.standard_normal(logits.shape).astype(np.float32) * 1e-6

    logits_path = tmp_path / "oracle_logits.npy"
    atomic_write_npy(
        np.ascontiguousarray(logits, dtype=np.dtype("<f4")),
        logits_path,
    )
    npy_manifest = build_npy_manifest(logits_path)

    oracle_report = _make_oracle_report(logits, npy_manifest)
    oracle_path = tmp_path / "oracle.json"
    atomic_write_json(oracle_report, oracle_path)

    c_logits_path = tmp_path / "c_logits.npy"
    atomic_write_npy(np.ascontiguousarray(pt_logits, dtype="<f4"), c_logits_path)
    c_manifest = build_npy_manifest(c_logits_path)
    c_report = _make_backend_report(
        "c", logits, pt_logits,
        include_provenance=True,
        artifact_manifest=c_manifest,
        reference_artifact_manifest=npy_manifest,
    )
    c_path = tmp_path / "c.json"
    atomic_write_json(c_report, c_path)

    numba_logits_path = tmp_path / "numba_logits.npy"
    atomic_write_npy(
        np.ascontiguousarray(pt_logits, dtype="<f4"),
        numba_logits_path,
    )
    numba_manifest = build_npy_manifest(numba_logits_path)
    numba_report = _make_backend_report(
        "numba", logits, pt_logits,
        include_provenance=True,
        artifact_manifest=numba_manifest,
        reference_artifact_manifest=npy_manifest,
    )
    numba_path = tmp_path / "numba.json"
    atomic_write_json(numba_report, numba_path)

    mlx_logits_path = tmp_path / "mlx_logits.npy"
    atomic_write_npy(
        np.ascontiguousarray(pt_logits, dtype="<f4"),
        mlx_logits_path,
    )
    mlx_manifest = build_npy_manifest(mlx_logits_path)
    mlx_report = _make_backend_report(
        "mlx", logits, pt_logits,
        include_provenance=True,
        artifact_manifest=mlx_manifest,
        reference_artifact_manifest=npy_manifest,
    )
    mlx_path = tmp_path / "mlx.json"
    atomic_write_json(mlx_report, mlx_path)

    return oracle_path, logits_path, c_path, numba_path, mlx_path, logits


# ---------------------------------------------------------------------------
# Tests: Full passing fixture
# ---------------------------------------------------------------------------


class TestFullPassingFixture:
    """Complete passing four-report fixture matching actual schemas."""

    def test_all_gates_pass(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        assert result["all_passed"], (
            f"Expected all gates to pass. Failed: "
            f"{[g['name'] for g in result.get('failed_gates', [])]}"
        )
        assert result["n_failed"] == 0
        assert result["n_gates"] > 0

    def test_schema_version(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        assert result["schema_version"] == VALIDATION_SCHEMA_VERSION

    def test_comparison_rows_present(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        comp = result["comparison"]
        assert "timing" in comp
        assert "memory" in comp
        assert "aggregate" in comp
        assert len(comp["timing"]) > 0
        assert len(comp["memory"]) > 0
        assert len(comp["aggregate"]) > 0


# ---------------------------------------------------------------------------
# Tests: Gate family mutations
# ---------------------------------------------------------------------------


class TestRunIDMutation:
    def test_wrong_run_id_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        result = validate_reports(
            run_id="wrong-run-id",
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        assert not result["all_passed"]
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("run_id" in n for n in failed_names)


class TestProvenanceMutation:
    def test_missing_provenance_in_backend_is_blocker(self, tmp_path):
        """Backend reports without provenance.implementation are flagged."""
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        # Strip provenance from C report to test blocker detection
        c_data = json.loads(c.read_text())
        del c_data["provenance"]
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("provenance_object" in name for name in failed_names)

    def test_mismatched_git_commit_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        # Modify C report to include provenance with wrong commit
        c_data = json.loads(c.read_text())
        c_data["provenance"] = _make_provenance()
        c_data["provenance"]["implementation"]["git_commit"] = "x" * 40
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("git_commit_matches_oracle" in n for n in failed_names)

    @pytest.mark.parametrize(
        "field",
        [
            "git_commit",
            "implementation_manifest_sha256",
            "source_hashes",
            "environment",
            "python_executable",
        ],
    )
    def test_missing_provenance_field_fails(self, tmp_path, field):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        c_data = json.loads(c.read_text())
        del c_data["provenance"]["implementation"][field]
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [gate["name"] for gate in result["failed_gates"]]
        assert any(f"provenance_{field}_present" in name for name in failed_names)


class TestModelSHAMutation:
    def test_wrong_model_repo_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        oracle_data = json.loads(oracle.read_text())
        oracle_data["model"]["repo"] = "wrong/repo"
        oracle.write_text(json.dumps(oracle_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("model_repo" in n for n in failed_names)

    def test_wrong_revision_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        c_data = json.loads(c.read_text())
        c_data["model"]["revision"] = "wrong_revision"
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("model_revision" in n for n in failed_names)

    def test_model_manifest_mismatch_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        c_data = json.loads(c.read_text())
        c_data["file_manifest"] = [
            {"name": "config.json", "size_bytes": 1, "sha256": "wrong" * 12 + "w"}
        ]
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("model_manifest_matches_oracle" in n for n in failed_names)


class TestPromptTokenMutation:
    def test_wrong_token_hash_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        c_data = json.loads(c.read_text())
        c_data["prompt"]["token_hash"] = "0" * 64
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("prompt_token_hash" in n for n in failed_names)

    def test_wrong_n_tokens_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        numba_data = json.loads(numba.read_text())
        numba_data["prompt"]["n_tokens"] = 15
        numba.write_text(json.dumps(numba_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("prompt_n_tokens" in n for n in failed_names)


class TestArtifactSHAMutation:
    def test_missing_logits_file_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        missing_path = tmp_path / "missing.npy"
        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=missing_path,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("raw_artifact_file_exists" in n for n in failed_names)

    def test_wrong_shape_in_report_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        oracle_data = json.loads(oracle.read_text())
        oracle_data["raw_artifact"]["shape"] = [1, 10, 100]
        oracle.write_text(json.dumps(oracle_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("raw_artifact_shape" in n or "raw_artifact_verify" in n for n in failed_names)


class TestWrongMissingBackend:
    def test_wrong_backend_name_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        c_data = json.loads(c.read_text())
        c_data["backend"]["name"] = "wrong_backend"
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("backend_name" in n for n in failed_names)

    def test_missing_report_file_returns_load_error(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=tmp_path / "nonexistent.json",
            numba_path=numba,
            mlx_path=mlx,
        )
        assert not result["all_passed"]
        assert len(result.get("load_errors", [])) > 0


class TestNonfiniteMutation:
    def test_nonfinite_logits_detected(self, tmp_path):
        oracle, logits, c, numba, mlx, logits_arr = _make_all_fixtures(tmp_path)
        # Rebuild C report with good PT logits but mark all_finite_pt=False
        rng = np.random.default_rng(42)
        pt_logits = logits_arr + rng.standard_normal(logits_arr.shape).astype(np.float32) * 1e-6
        c_data = _make_backend_report("c", logits_arr, pt_logits, include_provenance=True)
        # Override the finiteness flag to simulate nonfinite detection
        c_data["metrics"]["all_finite_pt"] = False
        c_data["publication_thresholds"]["passed"] = False
        for check in c_data["publication_thresholds"]["checks"]:
            if check["name"] == "all_finite_pt":
                check["passed"] = False
                check["actual"] = False
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("all_finite_pt" in n for n in failed_names)


class TestThresholdsMutation:
    def test_failing_publication_threshold(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        c_data = json.loads(c.read_text())
        c_data["publication_thresholds"]["passed"] = False
        c_data["publication_thresholds"]["checks"][0]["passed"] = False
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("publication_thresholds_passed" in n for n in failed_names)

    def test_forged_passing_metrics_fail_recomputation(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        c_data = json.loads(c.read_text())
        c_data["metrics"]["aggregate"]["cosine_min"] = -1.0
        cosine_check = next(
            check
            for check in c_data["publication_thresholds"]["checks"]
            if check["name"] == "cosine_min"
        )
        cosine_check.update(actual=-1.0, passed=True)
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [gate["name"] for gate in result["failed_gates"]]
        assert any("metrics_recomputed" in name for name in failed_names)


class TestTop1Mutation:
    def test_all_top1_match_false_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        numba_data = json.loads(numba.read_text())
        numba_data["metrics"]["all_top1_match"] = False
        numba.write_text(json.dumps(numba_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("all_top1_match" in n for n in failed_names)


class TestLayersChunksMutation:
    def test_wrong_layers_completed_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        mlx_data = json.loads(mlx.read_text())
        mlx_data["pytensor"]["layers_completed"] = 30
        mlx.write_text(json.dumps(mlx_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("pytensor_layers_completed" in n for n in failed_names)

    def test_wrong_chunks_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        c_data = json.loads(c.read_text())
        c_data["pytensor"]["chunks_processed"] = 50
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("pytensor_chunks_processed" in n for n in failed_names)

    @pytest.mark.parametrize(
        "field,bad_value,expected_gate",
        [
            ("sparse_layers_used", [0], "pytensor_sparse_layers"),
            ("layer_types_used", ["full_attention"] * 35, "pytensor_layer_types"),
            ("rope_bases_used", ["1M"] * 35, "pytensor_rope_bases"),
            ("logits_sha256", "not-a-hash", "pytensor_logits_sha256"),
        ],
    )
    def test_execution_shape_mutations_fail(
        self, tmp_path, field, bad_value, expected_gate,
    ):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        c_data = json.loads(c.read_text())
        c_data["pytensor"][field] = bad_value
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any(expected_gate in name for name in failed_names)

    def test_per_layer_timing_length_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        c_data = json.loads(c.read_text())
        c_data["timing"]["pt_per_layer_s"] = [0.1]
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("timing_per_layer_count" in name for name in failed_names)


class TestStageCountOrderRangesMutation:
    def test_wrong_stage_count_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        mlx_data = json.loads(mlx.read_text())
        mlx_data["pytensor"]["stage_count"] = 50
        mlx_data["pytensor"]["mlx_stages"] = mlx_data["pytensor"]["mlx_stages"][:50]
        mlx.write_text(json.dumps(mlx_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("stages_count" in n or "stages_list_length" in n for n in failed_names)

    def test_wrong_stage_order_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        mlx_data = json.loads(mlx.read_text())
        stages = mlx_data["pytensor"]["mlx_stages"]
        # Swap first two stages
        stages[0], stages[1] = stages[1], stages[0]
        mlx_data["pytensor"]["mlx_stages"] = stages
        mlx.write_text(json.dumps(mlx_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("stages_label_order" in n for n in failed_names)

    def test_missing_layer_stage_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        mlx_data = json.loads(mlx.read_text())
        stages = mlx_data["pytensor"]["mlx_stages"]
        # Remove layer_17
        stages = [s for s in stages if s.get("label") != "layer_17"]
        mlx_data["pytensor"]["mlx_stages"] = stages
        mlx_data["pytensor"]["stage_count"] = len(stages)
        mlx.write_text(json.dumps(mlx_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("stages_has_layer_17" in n for n in failed_names)


class TestMemoryLeakageMutation:
    def test_c_has_backend_mlx_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        c_data = json.loads(c.read_text())
        c_data["memory"]["backend_mlx"] = {"peak_mib": 100.0}
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("memory_no_backend_mlx" in n for n in failed_names)

    def test_mlx_missing_backend_mlx_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        mlx_data = json.loads(mlx.read_text())
        del mlx_data["memory"]["backend_mlx"]
        mlx.write_text(json.dumps(mlx_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("memory_has_backend_mlx" in n for n in failed_names)

    def test_oracle_missing_oracle_mlx_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        oracle_data = json.loads(oracle.read_text())
        del oracle_data["memory"]["oracle_mlx"]
        oracle.write_text(json.dumps(oracle_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("memory_has_oracle_mlx" in n for n in failed_names)

    def test_mlx_missing_raw_allocator_bytes_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        mlx_data = json.loads(mlx.read_text())
        del mlx_data["memory"]["backend_mlx"]["current_bytes"]
        mlx.write_text(json.dumps(mlx_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("current_bytes" in name for name in failed_names)


class TestPackageMismatchMutation:
    def test_version_mismatch_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        c_data = json.loads(c.read_text())
        c_data["versions"]["numpy"] = "1.0.0"
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("versions_numpy_matches_oracle" in n for n in failed_names)


class TestAbsolutePathMutation:
    @pytest.mark.parametrize(
        "value",
        [
            "https://example.com/model.json",
            "http://example.com/x",
            "--url=https://example.com/x?q=/tmp/not-a-file",
        ],
    )
    def test_http_urls_are_not_paths(self, value):
        assert _contains_absolute_path(value) == []

    def test_real_posix_path_is_detected(self):
        assert _contains_absolute_path("--output=/tmp/output.json")

    def test_windows_path_is_detected(self):
        assert _contains_absolute_path(r"C:\Users\me\output.json")


class TestMalformedSections:
    @pytest.mark.parametrize(
        "report_name,section,value",
        [
            ("oracle", "raw_artifact", "bad"),
            ("c", "pytensor", "bad"),
            ("c", "metrics", ["bad"]),
        ],
    )
    def test_non_object_section_fails_without_crashing(
        self, tmp_path, report_name, section, value,
    ):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        paths = {"oracle": oracle, "c": c, "numba": numba, "mlx": mlx}
        path = paths[report_name]
        report = json.loads(path.read_text())
        report[section] = value
        path.write_text(json.dumps(report))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )

        assert result["all_passed"] is False
        assert result["n_failed"] >= 1

    @pytest.mark.parametrize("basename", [None, 123, "../outside.npy"])
    def test_invalid_backend_artifact_basename_fails_without_crashing(
        self, tmp_path, basename,
    ):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        report = json.loads(c.read_text())
        report["pytensor"]["artifact"]["basename"] = basename
        c.write_text(json.dumps(report))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )

        failed_names = [gate["name"] for gate in result["failed_gates"]]
        assert any("backend_artifact_basename" in name for name in failed_names)

    def test_absolute_path_in_command_fails(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        c_data = json.loads(c.read_text())
        c_data["command"] = [
            "python",
            "/Users/test/snapshots/model.safetensors",
            "--output",
            "/tmp/results/output.json",
        ]
        c.write_text(json.dumps(c_data))

        result = validate_reports(
            run_id=RUN_ID,
            oracle_path=oracle,
            oracle_logits_path=logits,
            c_path=c,
            numba_path=numba,
            mlx_path=mlx,
        )
        failed_names = [g["name"] for g in result["failed_gates"]]
        assert any("no_absolute_paths" in n for n in failed_names)


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_success_exit_zero(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        output = tmp_path / "validation.json"
        rc = main([
            "--run-id", RUN_ID,
            "--oracle", str(oracle),
            "--oracle-logits", str(logits),
            "--c", str(c),
            "--numba", str(numba),
            "--mlx", str(mlx),
            "--output", str(output),
        ])
        assert rc == 0
        assert output.exists()
        report = json.loads(output.read_text())
        assert report["all_passed"]

    def test_cli_failure_exit_nonzero(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        # Corrupt C report
        c_data = json.loads(c.read_text())
        c_data["run_id"] = "wrong"
        c.write_text(json.dumps(c_data))

        rc = main([
            "--run-id", RUN_ID,
            "--oracle", str(oracle),
            "--oracle-logits", str(logits),
            "--c", str(c),
            "--numba", str(numba),
            "--mlx", str(mlx),
        ])
        assert rc == 1

    def test_cli_missing_file_returns_load_error(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        rc = main([
            "--run-id", RUN_ID,
            "--oracle", str(oracle),
            "--oracle-logits", str(logits),
            "--c", str(tmp_path / "missing.json"),
            "--numba", str(numba),
            "--mlx", str(mlx),
        ])
        assert rc == 1


class TestCLIParsing:
    def test_parse_args_minimal(self, tmp_path):
        args = parse_args([
            "--run-id", "r1",
            "--oracle", str(tmp_path / "o.json"),
            "--oracle-logits", str(tmp_path / "l.npy"),
            "--c", str(tmp_path / "c.json"),
            "--numba", str(tmp_path / "n.json"),
            "--mlx", str(tmp_path / "m.json"),
        ])
        assert args.run_id == "r1"
        assert args.output is None

    def test_parse_args_with_output(self, tmp_path):
        args = parse_args([
            "--run-id", "r1",
            "--oracle", str(tmp_path / "o.json"),
            "--oracle-logits", str(tmp_path / "l.npy"),
            "--c", str(tmp_path / "c.json"),
            "--numba", str(tmp_path / "n.json"),
            "--mlx", str(tmp_path / "m.json"),
            "--output", str(tmp_path / "out.json"),
        ])
        assert args.output == tmp_path / "out.json"

    def test_parse_args_requires_all(self):
        with pytest.raises(SystemExit):
            parse_args(["--run-id", "r1"])


# ---------------------------------------------------------------------------
# Tests: Strict JSON
# ---------------------------------------------------------------------------


class TestStrictJSON:
    def test_rejects_nan_in_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"value": NaN}')
        with pytest.raises(ValueError, match="non-finite"):
            load_strict_json(bad)

    def test_rejects_non_object_root(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('[1, 2, 3]')
        with pytest.raises(ValueError, match="root must be"):
            load_strict_json(bad)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_strict_json(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# Tests: Atomic output
# ---------------------------------------------------------------------------


class TestAtomicOutput:
    def test_output_is_valid_json(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        output = tmp_path / "validation.json"
        main([
            "--run-id", RUN_ID,
            "--oracle", str(oracle),
            "--oracle-logits", str(logits),
            "--c", str(c),
            "--numba", str(numba),
            "--mlx", str(mlx),
            "--output", str(output),
        ])
        data = json.loads(output.read_text())
        assert isinstance(data, dict)
        assert "gates" in data
        assert "all_passed" in data

    def test_output_no_partial_on_success(self, tmp_path):
        oracle, logits, c, numba, mlx, _ = _make_all_fixtures(tmp_path)
        output = tmp_path / "validation.json"
        main([
            "--run-id", RUN_ID,
            "--oracle", str(oracle),
            "--oracle-logits", str(logits),
            "--c", str(c),
            "--numba", str(numba),
            "--mlx", str(mlx),
            "--output", str(output),
        ])
        # No temp files left
        temps = list(tmp_path.glob(".evidence_*.tmp"))
        assert temps == []


# ---------------------------------------------------------------------------
# Tests: Stage helpers
# ---------------------------------------------------------------------------


class TestStageHelpers:
    def test_expected_n_chunks_with_remainder(self):
        assert _expected_n_chunks(262272, 4096) == 65

    def test_expected_n_chunks_exact(self):
        assert _expected_n_chunks(262144, 4096) == 64

    def test_expected_n_stages(self):
        # 2 + 35 + 1 + 65 = 103
        assert _expected_n_stages(35, 262272, 4096) == 103

    def test_expected_stage_labels_count(self):
        labels = _expected_stage_labels(35, 262272, 4096)
        assert len(labels) == 103

    def test_expected_stage_labels_first_last(self):
        labels = _expected_stage_labels(35, 262272, 4096)
        assert labels[0] == "initial_projections"
        assert labels[1] == "per_layer_projection"
        assert labels[2] == "layer_0"
        assert labels[36] == "layer_34"
        assert labels[37] == "final_unembed"
        assert labels[38] == "vocab_chunk_0_4096"
        assert labels[-1] == "vocab_chunk_262144_262272"


# ---------------------------------------------------------------------------
# Tests: GateCollector
# ---------------------------------------------------------------------------


class TestGateCollector:
    def test_empty_collector_passes(self):
        gc = GateCollector()
        assert gc.all_passed
        assert gc.failed == []

    def test_all_pass(self):
        gc = GateCollector()
        gc.add("g1", "a", "a", True)
        gc.add("g2", "b", "b", True)
        assert gc.all_passed
        assert len(gc.gates) == 2

    def test_one_fail(self):
        gc = GateCollector()
        gc.add("g1", "a", "a", True)
        gc.add("g2", "b", "c", False)
        assert not gc.all_passed
        assert len(gc.failed) == 1
        assert gc.failed[0]["name"] == "g2"
