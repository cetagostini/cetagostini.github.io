"""Tests for the reusable Python-first inference API."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cetagostini.utils.pytensor import Gemma3n, SmolLM2, inference
from cetagostini.utils.pytensor import api
from cetagostini.utils.pytensor.reports import InferenceReport, InferenceResult


def _smollm2_report():
    return {
        "model": {"repo": "bartowski/SmolLM2-135M-Instruct-GGUF"},
        "prompt": {"text": "What is 2 + 2?", "token_ids": [1, 10, 2], "n_tokens": 3},
        "generation": {
            "text": "4",
            "generated_ids": [36],
            "n_tokens": 1,
            "stop_reason": "eos",
        },
        "timing": {
            "load_dequant_s": 1.0,
            "mlx_convert_s": 0.2,
            "compile_s": 0.3,
            "prefill_s": 0.1,
            "decode_total_s": 0.4,
        },
        "memory": {"peak_rss_mib": 100.0},
        "reference": {
            "argmax_match": True,
            "top10_overlap": 10,
            "pearson": 0.999,
            "mean_abs_diff": 0.1,
        },
    }


def _gemma3n_report():
    return {
        "model": {"repo": "mlx-community/gemma-3n-E4B-it-lm-4bit"},
        "prompt": {
            "text": "Explain a symbolic graph.",
            "token_ids": [2, 4, 6],
            "n_tokens": 3,
        },
        "backend": {"name": "numba", "linker": "numba", "mode": "fast_compile"},
        "metrics": {
            "final_top1_pt": 2717,
            "final_top1_pt_text": "```",
            "final_top1_match": True,
            "all_top1_match": True,
            "aggregate": {"pearson_mean": 0.998605, "top10_overlap_mean": 9.7},
        },
        "publication_thresholds": {"passed": True},
        "timing": {"pt_compile_s": 0.9, "pt_total_s": 55.6},
        "memory": {"peak_rss_mib": 8445.0},
    }


def test_inference_dispatches_to_smollm2_runner(monkeypatch, tmp_path):
    calls = {}

    def fake_run_inference(**kwargs):
        calls.update(kwargs)
        return _smollm2_report()

    monkeypatch.setattr(
        api,
        "_load_smollm2_runner",
        lambda: SimpleNamespace(run_inference=fake_run_inference),
    )

    model = SmolLM2.from_gguf(tmp_path / "model.gguf", cache_capacity=128)
    result = inference(model, input="What is 2 + 2?", max_tokens=8, reference=True)

    assert isinstance(result, InferenceResult)
    assert result.output == "4"
    assert result.output_token_ids == (36,)
    assert result.report.mode == "generation"
    assert result.report.validation["argmax_match"] is True
    assert result.report.total_s == 2.0
    assert calls["model_path"] == (tmp_path / "model.gguf")
    assert calls["prompt"] == "What is 2 + 2?"
    assert calls["cache_capacity"] == 128
    assert calls["max_tokens"] == 8
    assert calls["reference"] is True


def test_inference_dispatches_to_gemma_runner(monkeypatch, tmp_path):
    calls = {}

    def fake_run_inference(**kwargs):
        calls.update(kwargs)
        return _gemma3n_report()

    monkeypatch.setattr(
        api,
        "_load_gemma3n_runner",
        lambda: SimpleNamespace(run_inference=fake_run_inference),
    )

    model = Gemma3n.from_snapshot(tmp_path / "snapshot", backend="numba")
    result = inference(model, input="Explain a symbolic graph.", max_tokens=1)

    assert result.output == "```"
    assert result.output_token_ids == (2717,)
    assert result.report.backend == "numba"
    assert result.report.mode == "next_token"
    assert result.report.validation["thresholds_passed"] is True
    assert calls["snapshot_path"] == (tmp_path / "snapshot")
    assert calls["backend"] == "numba"
    assert calls["reference_only"] is False
    assert calls["max_tokens"] == 1


@pytest.mark.parametrize("invalid_input", ["", "  ", None])
def test_inference_rejects_empty_input(invalid_input):
    model = SmolLM2.from_gguf("model.gguf")
    with pytest.raises(ValueError, match="non-empty"):
        inference(model, input=invalid_input, max_tokens=1)


def test_inference_rejects_invalid_model():
    with pytest.raises(TypeError, match="InferenceModel"):
        inference(object(), input="hello", max_tokens=1)


def test_model_descriptors_validate_backend_and_capacity():
    with pytest.raises(ValueError, match="cache_capacity"):
        SmolLM2.from_gguf("model.gguf", cache_capacity=0)

    with pytest.raises(ValueError, match="only"):
        SmolLM2.from_gguf("model.gguf", backend="c")

    with pytest.raises(ValueError, match="'c' or 'numba'"):
        Gemma3n.from_snapshot("snapshot", backend="jax")


def test_gemma_accepts_multi_token_generation(monkeypatch):
    calls = {}

    def fake_run_inference(**kwargs):
        calls.update(kwargs)
        report = _gemma3n_report()
        report["generation"] = {
            "generated_ids": [2717, 42],
            "text": "Causal inference",
            "stop_reason": "max_tokens",
            "validation_scope": "first_step_all_prompt_positions",
        }
        return report

    monkeypatch.setattr(
        api,
        "_load_gemma3n_runner",
        lambda: SimpleNamespace(run_inference=fake_run_inference),
    )

    model = Gemma3n.from_snapshot("snapshot")
    result = inference(model, input="hello", max_tokens=2)

    assert calls["max_tokens"] == 2
    assert result.output == "Causal inference"
    assert result.output_token_ids == (2717, 42)
    assert result.report.mode == "cache_free_generation"


def test_result_report_copy_is_detached():
    result = InferenceResult.from_smollm2_report(_smollm2_report())
    copied = result.to_dict()
    copied["generation"]["text"] = "changed"

    assert result.output == "4"
    assert copied["generation"]["text"] == "changed"


def test_low_level_package_modules_expose_validated_implementations():
    from cetagostini.utils.pytensor import models, weights

    assert models.SmolLM2Config is weights.SmolLM2Config
    assert callable(models.compile_prefill_layer)
    assert callable(weights.load_smollm2_weights)
    assert callable(weights.dequantize_affine4)


def test_cache_free_report_total_includes_additional_generation():
    report = InferenceReport(
        model="gemma",
        backend="numba",
        mode="cache_free_generation",
        prompt_tokens=10,
        output_tokens=3,
        stop_reason="max_tokens",
        timing_s={
            "tokenize_s": 1.0,
            "ref_load_s": 2.0,
            "ref_forward_s": 0.1,
            "ref_sync_s": 0.2,
            "pt_compile_s": 0.3,
            "pt_total_s": 4.0,
            "generation_additional_wall_s": 8.0,
        },
    )

    assert report.total_s == 15.6
