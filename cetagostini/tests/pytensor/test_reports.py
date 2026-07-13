"""Focused tests for the InferenceReport and InferenceResult adapters."""

from __future__ import annotations

from copy import deepcopy

import pytest

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


def _gemma3n_next_token_report():
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


def _gemma3n_generation_report():
    report = _gemma3n_next_token_report()
    report["generation"] = {
        "generated_ids": [2717, 42],
        "text": "Causal inference",
        "stop_reason": "max_tokens",
        "validation_scope": "all_generated_prefixes_fresh_cache",
        "all_reference_tokens_match": False,
    }
    return report


class TestInferenceReportTotalS:
    def test_generation_mode_sums_all_keys(self):
        report = InferenceReport(
            model="smollm2",
            backend="mlx",
            mode="generation",
            prompt_tokens=3,
            output_tokens=1,
            stop_reason="eos",
            timing_s={
                "load_dequant_s": 1.0,
                "mlx_convert_s": 0.2,
                "compile_s": 0.3,
                "prefill_s": 0.1,
                "decode_total_s": 0.4,
            },
        )
        assert report.total_s == 2.0

    def test_cache_free_generation_includes_additional_wall(self):
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

    def test_next_token_mode_uses_default_keys(self):
        report = InferenceReport(
            model="gemma",
            backend="numba",
            mode="next_token",
            prompt_tokens=3,
            output_tokens=1,
            stop_reason="next_token",
            timing_s={
                "tokenize_s": 0.1,
                "ref_load_s": 0.2,
                "ref_forward_s": 0.3,
                "ref_sync_s": 0.4,
                "pt_compile_s": 0.5,
                "pt_total_s": 0.6,
            },
        )
        assert report.total_s == 2.1

    def test_returns_none_when_no_numeric_values(self):
        report = InferenceReport(
            model="gemma",
            backend="numba",
            mode="next_token",
            prompt_tokens=3,
            output_tokens=1,
            stop_reason="next_token",
            timing_s={},
        )
        assert report.total_s is None


class TestInferenceReportToDict:
    def test_returns_deep_copy(self):
        raw = {"key": {"nested": "value"}}
        report = InferenceReport(
            model="test",
            backend="test",
            mode="generation",
            prompt_tokens=1,
            output_tokens=1,
            stop_reason="eos",
            raw=raw,
        )
        copied = report.to_dict()
        copied["key"]["nested"] = "changed"
        assert report.raw["key"]["nested"] == "value"


class TestInferenceResultFromSmolLM2:
    def test_extracts_text_and_tokens(self):
        result = InferenceResult.from_smollm2_report(_smollm2_report())
        assert result.input == "What is 2 + 2?"
        assert result.output == "4"
        assert result.input_token_ids == (1, 10, 2)
        assert result.output_token_ids == (36,)

    def test_report_has_correct_metadata(self):
        result = InferenceResult.from_smollm2_report(_smollm2_report())
        assert result.report.model == "bartowski/SmolLM2-135M-Instruct-GGUF"
        assert result.report.backend == "mlx"
        assert result.report.mode == "generation"
        assert result.report.prompt_tokens == 3
        assert result.report.output_tokens == 1
        assert result.report.stop_reason == "eos"

    def test_validation_extracts_reference_keys(self):
        result = InferenceResult.from_smollm2_report(_smollm2_report())
        assert result.report.validation["argmax_match"] is True
        assert result.report.validation["top10_overlap"] == 10
        assert result.report.validation["pearson"] == 0.999
        assert result.report.validation["mean_abs_diff"] == 0.1

    def test_to_dict_returns_deep_copy(self):
        result = InferenceResult.from_smollm2_report(_smollm2_report())
        copied = result.to_dict()
        copied["generation"]["text"] = "changed"
        assert result.output == "4"


class TestInferenceResultFromGemma3n:
    def test_next_token_mode(self):
        result = InferenceResult.from_gemma3n_report(_gemma3n_next_token_report())
        assert result.input == "Explain a symbolic graph."
        assert result.output == "```"
        assert result.input_token_ids == (2, 4, 6)
        assert result.output_token_ids == (2717,)
        assert result.report.mode == "next_token"
        assert result.report.stop_reason == "next_token"
        assert result.report.backend == "numba"

    def test_cache_free_generation_mode(self):
        result = InferenceResult.from_gemma3n_report(_gemma3n_generation_report())
        assert result.output == "Causal inference"
        assert result.output_token_ids == (2717, 42)
        assert result.report.mode == "cache_free_generation"
        assert result.report.stop_reason == "max_tokens"
        assert result.report.output_tokens == 2

    def test_validation_filters_none_values(self):
        result = InferenceResult.from_gemma3n_report(_gemma3n_next_token_report())
        assert "final_top1_match" in result.report.validation
        assert result.report.validation["final_top1_match"] is True
        assert "all_generated_tokens_match" not in result.report.validation

    def test_generation_validation_includes_scope(self):
        result = InferenceResult.from_gemma3n_report(_gemma3n_generation_report())
        assert result.report.validation["scope"] == "all_generated_prefixes_fresh_cache"
        assert result.report.validation["all_generated_tokens_match"] is False


class TestInferenceResultRepr:
    def test_repr_includes_output_and_report(self):
        result = InferenceResult.from_smollm2_report(_smollm2_report())
        text = repr(result)
        assert "InferenceResult(" in text
        assert "output=" in text
        assert "output_token_ids=" in text
        assert "report=" in text


class TestInferenceReportRepr:
    def test_repr_includes_key_fields(self):
        report = InferenceReport(
            model="test-model",
            backend="numba",
            mode="generation",
            prompt_tokens=5,
            output_tokens=3,
            stop_reason="eos",
            validation={"argmax_match": True},
        )
        text = repr(report)
        assert "InferenceReport(" in text
        assert "test-model" in text
        assert "numba" in text
        assert "generation" in text
