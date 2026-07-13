"""Tests for run_inference and run_cache_free_generation integration.

These tests verify that the runner modules expose the correct symbols and
signatures required by the api.Gemma3n and api.SmolLM2 descriptors.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestGemma3nRunnerSymbols:
    """Verify run_gemma3n_pytensor exposes required symbols."""

    def test_run_inference_exists(self):
        from cetagostini.utils.pytensor import run_gemma3n_pytensor

        assert hasattr(run_gemma3n_pytensor, "run_inference")
        assert callable(run_gemma3n_pytensor.run_inference)

    def test_run_inference_signature(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import run_inference

        sig = inspect.signature(run_inference)
        params = list(sig.parameters.keys())

        # Required positional
        assert "snapshot_path" in params

        # Required keyword-only
        assert "prompt" in params
        assert "backend" in params
        assert "reference_only" in params
        assert "max_tokens" in params
        assert "cache_layer_weights" in params

        # Optional keyword-only
        assert "output_path" in params
        assert "progress" in params

        # Verify keyword-only
        for name in ("prompt", "backend", "reference_only", "max_tokens",
                     "cache_layer_weights", "output_path", "progress"):
            assert sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY

    def test_run_cache_free_generation_exists(self):
        from cetagostini.utils.pytensor import run_gemma3n_pytensor

        assert hasattr(run_gemma3n_pytensor, "run_cache_free_generation")
        assert callable(run_gemma3n_pytensor.run_cache_free_generation)

    def test_decode_token_ids_exists(self):
        from cetagostini.utils.pytensor import run_gemma3n_pytensor

        assert hasattr(run_gemma3n_pytensor, "decode_token_ids")
        assert callable(run_gemma3n_pytensor.decode_token_ids)

    def test_get_stop_token_ids_exists(self):
        from cetagostini.utils.pytensor import run_gemma3n_pytensor

        assert hasattr(run_gemma3n_pytensor, "get_stop_token_ids")
        assert callable(run_gemma3n_pytensor.get_stop_token_ids)

    def test_decode_token_ids_empty(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import decode_token_ids

        tokenizer = MagicMock()
        assert decode_token_ids(tokenizer, []) == ""

    def test_decode_token_ids_normalizes_bytes(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import decode_token_ids

        tokenizer = MagicMock()
        tokenizer.decode.return_value = b"hello"
        assert decode_token_ids(tokenizer, [1, 2, 3]) == "hello"

    def test_decode_token_ids_normalizes_non_string(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import decode_token_ids

        tokenizer = MagicMock()
        tokenizer.decode.return_value = 42
        assert decode_token_ids(tokenizer, [1]) == "42"

    def test_get_stop_token_ids_from_eos_token_ids(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import get_stop_token_ids

        tokenizer = SimpleNamespace(eos_token_ids=[1, 106])
        result = get_stop_token_ids(tokenizer)
        assert result == frozenset([1, 106])

    def test_get_stop_token_ids_from_eos_token_id(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import get_stop_token_ids

        tokenizer = SimpleNamespace(eos_token_id=2)
        result = get_stop_token_ids(tokenizer)
        assert result == frozenset([2])

    def test_get_stop_token_ids_empty(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import get_stop_token_ids

        tokenizer = SimpleNamespace()
        result = get_stop_token_ids(tokenizer)
        assert result == frozenset()


class TestSmolLM2RunnerSymbols:
    """Verify run_smollm2_pytensor exposes required symbols."""

    def test_run_inference_exists(self):
        from cetagostini.utils.pytensor import run_smollm2_pytensor

        assert hasattr(run_smollm2_pytensor, "run_inference")
        assert callable(run_smollm2_pytensor.run_inference)

    def test_run_inference_signature(self):
        from cetagostini.utils.pytensor.run_smollm2_pytensor import run_inference

        sig = inspect.signature(run_inference)
        params = list(sig.parameters.keys())

        # Required positional
        assert "model_path" in params

        # Required keyword-only
        assert "prompt" in params
        assert "max_tokens" in params
        assert "cache_capacity" in params
        assert "reference" in params
        assert "verify_hash" in params

        # Optional keyword-only
        assert "progress" in params

        # Verify keyword-only
        for name in ("prompt", "max_tokens", "cache_capacity", "reference",
                     "verify_hash", "progress"):
            assert sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY


class TestGemma3nAPIIntegration:
    """Test that api.Gemma3n.infer() can call run_inference."""

    def test_gemma3n_infer_calls_run_inference(self, tmp_path, monkeypatch):
        from cetagostini.utils.pytensor import api

        calls = {}

        def fake_run_inference(**kwargs):
            calls.update(kwargs)
            return {
                "model": {"repo": "mlx-community/gemma-3n-E4B-it-lm-4bit"},
                "prompt": {
                    "text": "test",
                    "token_ids": [1, 2, 3],
                    "n_tokens": 3,
                },
                "backend": {"name": "c"},
                "metrics": {
                    "final_top1_pt": 42,
                    "final_top1_pt_text": "hello",
                    "final_top1_match": True,
                    "all_top1_match": True,
                    "aggregate": {"pearson_mean": 0.999, "top10_overlap_mean": 9.5},
                },
                "publication_thresholds": {"passed": True},
                "timing": {"pt_compile_s": 1.0, "pt_total_s": 2.0},
                "memory": {"peak_rss_mib": 100.0},
            }

        monkeypatch.setattr(
            api,
            "_load_gemma3n_runner",
            lambda: SimpleNamespace(run_inference=fake_run_inference),
        )

        model = api.Gemma3n.from_snapshot(tmp_path / "snapshot", backend="c")
        result = api.inference(model, input="test", max_tokens=1)

        assert calls["snapshot_path"] == (tmp_path / "snapshot")
        assert calls["prompt"] == "test"
        assert calls["backend"] == "c"
        assert calls["reference_only"] is False
        assert calls["max_tokens"] == 1
        assert calls["cache_layer_weights"] is False
        assert result.output == "hello"
        assert result.output_token_ids == (42,)

    def test_gemma3n_infer_multi_token(self, tmp_path, monkeypatch):
        from cetagostini.utils.pytensor import api

        calls = {}

        def fake_run_inference(**kwargs):
            calls.update(kwargs)
            return {
                "model": {"repo": "mlx-community/gemma-3n-E4B-it-lm-4bit"},
                "prompt": {
                    "text": "test",
                    "token_ids": [1, 2, 3],
                    "n_tokens": 3,
                },
                "backend": {"name": "numba"},
                "metrics": {
                    "final_top1_pt": 42,
                    "final_top1_pt_text": "hello",
                    "final_top1_match": True,
                    "all_top1_match": True,
                    "aggregate": {"pearson_mean": 0.999, "top10_overlap_mean": 9.5},
                },
                "publication_thresholds": {"passed": True},
                "generation": {
                    "generated_ids": [42, 43, 44],
                    "text": "hello world test",
                    "stop_reason": "max_tokens",
                    "validation_scope": "first_step_all_prompt_positions",
                },
                "timing": {"pt_compile_s": 1.0, "pt_total_s": 2.0},
                "memory": {"peak_rss_mib": 100.0},
            }

        monkeypatch.setattr(
            api,
            "_load_gemma3n_runner",
            lambda: SimpleNamespace(run_inference=fake_run_inference),
        )

        model = api.Gemma3n.from_snapshot(tmp_path / "snapshot", backend="numba")
        result = api.inference(model, input="test", max_tokens=3)

        assert calls["max_tokens"] == 3
        assert result.output == "hello world test"
        assert result.output_token_ids == (42, 43, 44)
        assert result.report.mode == "cache_free_generation"


class TestSmolLM2APIIntegration:
    """Test that api.SmolLM2.infer() can call run_inference."""

    def test_smollm2_infer_calls_run_inference(self, tmp_path, monkeypatch):
        from cetagostini.utils.pytensor import api

        calls = {}

        def fake_run_inference(**kwargs):
            calls.update(kwargs)
            return {
                "model": {"repo": "bartowski/SmolLM2-135M-Instruct-GGUF"},
                "prompt": {"text": "test", "token_ids": [1, 2], "n_tokens": 2},
                "generation": {
                    "text": "output",
                    "generated_ids": [10, 20],
                    "n_tokens": 2,
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

        monkeypatch.setattr(
            api,
            "_load_smollm2_runner",
            lambda: SimpleNamespace(run_inference=fake_run_inference),
        )

        model = api.SmolLM2.from_gguf(tmp_path / "model.gguf", cache_capacity=128)
        result = api.inference(model, input="test", max_tokens=2, reference=True)

        assert calls["model_path"] == (tmp_path / "model.gguf")
        assert calls["prompt"] == "test"
        assert calls["max_tokens"] == 2
        assert calls["cache_capacity"] == 128
        assert calls["reference"] is True
        assert calls["verify_hash"] is True
        assert result.output == "output"
        assert result.output_token_ids == (10, 20)


class TestRunCacheFreeGeneration:
    """Test run_cache_free_generation with mocked dependencies."""

    def test_max_tokens_zero_raises(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import run_cache_free_generation

        with pytest.raises(ValueError, match="max_tokens must be at least 1"):
            run_cache_free_generation(
                loader=MagicMock(),
                tokenizer=MagicMock(),
                prompt_token_ids=[1, 2, 3],
                first_token_id=42,
                max_tokens=0,
                stop_token_ids=frozenset(),
                text_config=MagicMock(),
                pt_config=MagicMock(),
                backend="c",
                first_compile_s=1.0,
                first_forward_s=2.0,
                layer_weight_cache=None,
                reference_forward=None,
                first_reference_metrics=None,
                first_reference_thresholds=None,
                progress=None,
            )

    def test_first_token_is_stop_token(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import run_cache_free_generation

        tokenizer = MagicMock()
        tokenizer.decode.return_value = ""

        result = run_cache_free_generation(
            loader=MagicMock(),
            tokenizer=tokenizer,
            prompt_token_ids=[1, 2, 3],
            first_token_id=106,
            max_tokens=5,
            stop_token_ids=frozenset([1, 106]),
            text_config=MagicMock(),
            pt_config=MagicMock(),
            backend="c",
            first_compile_s=1.0,
            first_forward_s=2.0,
            layer_weight_cache=None,
            reference_forward=None,
            first_reference_metrics=None,
            first_reference_thresholds=None,
            progress=None,
        )

        assert result["generated_ids"] == []
        assert result["text"] == ""
        assert result["stop_reason"] == "stop_token"
        assert result["stop_token_id"] == 106
        assert len(result["steps"]) == 1
        assert result["steps"][0]["step"] == 1
        assert result["steps"][0]["token_id"] == 106

    def test_generates_multiple_tokens(self):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import (
            run_cache_free_generation,
            _compile_graphs,
            run_pytensor_forward,
        )

        tokenizer = MagicMock()
        tokenizer.decode.return_value = "hello world"

        # Mock the compilation and forward pass
        mock_compiled = MagicMock()

        # Create a side effect that returns different logits for each call
        call_count = [0]

        def mock_forward(*args, **kwargs):
            call_count[0] += 1
            logits = np.zeros((1, 4, 100), dtype=np.float32)
            if call_count[0] == 1:
                logits[0, -1, 43] = 10.0  # Second generated token
            else:
                logits[0, -1, 44] = 10.0  # Third generated token

            return {
                "logits": logits,
                "embed_s": 0.01,
                "ple_s": 0.01,
                "global_load_s": 0.01,
                "initial_s": 0.01,
                "per_layer_proj_s": 0.01,
                "per_layer_s": [0.1, 0.1],
                "final_s": 0.01,
                "logits_s": 0.01,
                "total_s": 1.0,
                "layers_completed": 2,
                "layer_types_used": ["sliding", "full"],
                "rope_bases_used": ["10K", "1M"],
                "sparse_layers_used": [0],
                "chunks_processed": 1,
            }

        with patch("cetagostini.utils.pytensor.run_gemma3n_pytensor._compile_graphs",
                   return_value=mock_compiled):
            with patch("cetagostini.utils.pytensor.run_gemma3n_pytensor.run_pytensor_forward",
                       side_effect=mock_forward):
                result = run_cache_free_generation(
                    loader=MagicMock(),
                    tokenizer=tokenizer,
                    prompt_token_ids=[1, 2, 3],
                    first_token_id=42,
                    max_tokens=3,
                    stop_token_ids=frozenset([1]),
                    text_config=MagicMock(),
                    pt_config=MagicMock(),
                    backend="c",
                    first_compile_s=1.0,
                    first_forward_s=2.0,
                    layer_weight_cache=None,
                    reference_forward=None,
                    first_reference_metrics=None,
                    first_reference_thresholds=None,
                    progress=None,
                )

        assert result["generated_ids"] == [42, 43, 44]
        assert result["text"] == "hello world"
        assert result["stop_reason"] == "max_tokens"
        assert len(result["steps"]) == 3  # First step + 2 generated
        assert result["steps"][0]["token_id"] == 42
        assert result["steps"][1]["token_id"] == 43
        assert result["steps"][2]["token_id"] == 44


class TestRunInferenceOutputPath:
    """Test that run_inference atomically writes to output_path."""

    def test_output_path_writes_json(self, tmp_path, monkeypatch):
        from cetagostini.utils.pytensor.run_gemma3n_pytensor import run_inference

        # Mock all dependencies
        monkeypatch.setattr(
            "cetagostini.utils.pytensor.run_gemma3n_pytensor.validate_snapshot",
            lambda x: {"model_type": "gemma3n"},
        )
        monkeypatch.setattr(
            "cetagostini.utils.pytensor.run_gemma3n_pytensor.build_file_manifest",
            lambda x: [],
        )
        monkeypatch.setattr(
            "cetagostini.utils.pytensor.run_gemma3n_pytensor.collect_versions",
            lambda: {"python": "3.11"},
        )
        monkeypatch.setattr(
            "cetagostini.utils.pytensor.run_gemma3n_pytensor.check_optional_statuses",
            lambda: {},
        )
        monkeypatch.setattr(
            "cetagostini.utils.pytensor.run_gemma3n_pytensor.get_backend_info",
            lambda x: {"name": "c"},
        )
        monkeypatch.setattr(
            "cetagostini.utils.pytensor.run_gemma3n_pytensor.load_tokenizer_from_snapshot",
            lambda x: MagicMock(),
        )
        monkeypatch.setattr(
            "cetagostini.utils.pytensor.run_gemma3n_pytensor.format_and_tokenize",
            lambda t, p: ("<formatted>", [1, 2, 3]),
        )
        monkeypatch.setattr(
            "cetagostini.utils.pytensor.run_gemma3n_pytensor.run_mlx_reference",
            lambda s, t: {
                "logits": np.zeros((1, 3, 100), dtype=np.float32),
                "load_s": 1.0,
                "forward_s": 0.5,
                "sync_s": 0.1,
                "peak_memory_mib": 100.0,
                "vocab_size": 100,
                "seq_len": 3,
            },
        )
        monkeypatch.setattr(
            "cetagostini.utils.pytensor.run_gemma3n_pytensor.get_peak_rss_mib",
            lambda: 200.0,
        )
        monkeypatch.setattr(
            "cetagostini.utils.pytensor.run_gemma3n_pytensor.detect_revision",
            lambda x: "00b5ecdc79ba872a9b4cd32f4327e263bab5936c",
        )

        output_file = tmp_path / "result.json"

        result = run_inference(
            snapshot_path=tmp_path / "snapshot",
            prompt="test",
            backend="c",
            reference_only=True,
            max_tokens=1,
            cache_layer_weights=False,
            output_path=output_file,
            progress=None,
        )

        assert output_file.exists()
        import json
        loaded = json.loads(output_file.read_text())
        assert loaded["model"]["repo"] == "mlx-community/gemma-3n-E4B-it-lm-4bit"
        assert loaded["prompt"]["text"] == "test"
