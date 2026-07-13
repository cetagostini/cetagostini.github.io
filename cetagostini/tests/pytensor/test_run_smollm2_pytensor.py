"""Tests for run_smollm2_pytensor runtime.

Focused unit tests covering:
- Mask builders (write positions, attention patterns)
- Generation state sequence (prefill token then decode)
- EOS / capacity / max_tokens stopping conditions
- Result sanitization (no absolute paths, no env dump)
- CLI parsing

One real integration test is gated by the ``SMOLLM2_GGUF`` environment
variable and exercises the full pipeline end-to-end.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cetagostini.utils.pytensor.run_smollm2_pytensor import (
    DEFAULT_PROMPT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_CACHE_CAPACITY,
    atomic_write_json,
    build_attention_mask,
    build_write_mask,
    collect_versions,
    compare_logits,
    compute_n_ctx,
    format_chat_prompt,
    layer_weight_args,
    parse_args,
    run_generation,
    run_reference,
    sanitize_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_small_config():
    """Create a small SmolLM2Config-like namespace for testing."""
    return SimpleNamespace(
        vocab_size=100,
        hidden_size=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        intermediate_size=64,
        context_length=32,
        rms_eps=1e-5,
        rope_theta=100_000.0,
        bos=1,
        eos=2,
    )


def _make_mock_mlx():
    """Create a mock mlx.core module."""
    mock_mlx = MagicMock()
    mock_mlx.array = lambda x: np.asarray(x)
    mock_mlx.zeros = lambda shape: np.zeros(shape, dtype=np.float32)
    mock_mlx.concatenate = lambda arrays, axis: np.concatenate(arrays, axis=axis)
    mock_mlx.eval = lambda *args: None
    mock_mlx.float32 = np.float32
    return mock_mlx


def _make_mock_weights(config):
    """Create mock MLX weights container."""
    rng = np.random.default_rng(42)
    return {
        "emb": rng.standard_normal((config.vocab_size, config.hidden_size)).astype(np.float32),
        "final_norm": np.ones(config.hidden_size, dtype=np.float32),
        "layers": [
            {
                "wq": rng.standard_normal((config.hidden_size, config.n_heads * config.head_dim)).astype(np.float32),
                "wk": rng.standard_normal((config.hidden_size, config.n_kv_heads * config.head_dim)).astype(np.float32),
                "wv": rng.standard_normal((config.hidden_size, config.n_kv_heads * config.head_dim)).astype(np.float32),
                "wo": rng.standard_normal((config.n_heads * config.head_dim, config.hidden_size)).astype(np.float32),
                "w_gate": rng.standard_normal((config.hidden_size, config.intermediate_size)).astype(np.float32),
                "w_up": rng.standard_normal((config.hidden_size, config.intermediate_size)).astype(np.float32),
                "w_down": rng.standard_normal((config.intermediate_size, config.hidden_size)).astype(np.float32),
                "attn_norm": np.ones(config.hidden_size, dtype=np.float32),
                "ffn_norm": np.ones(config.hidden_size, dtype=np.float32),
            }
            for _ in range(config.n_layers)
        ],
    }


def _make_mock_embed(config, seq_len):
    """Create a mock embedding function."""
    def embed_fn(token_ids, emb_table):
        B = token_ids.shape[0]
        return np.zeros((B, seq_len, config.hidden_size), dtype=np.float32)
    return embed_fn


def _make_mock_prefill_layer(config):
    """Create a mock prefill layer function."""
    def layer_fn(hidden, q_w, k_w, v_w, o_w, gate_w, up_w, down_w, in_gamma, post_gamma, cos, sin):
        B, T, H = hidden.shape
        n_kv = config.n_kv_heads
        hd = config.head_dim
        k_rot = np.zeros((B, n_kv, T, hd), dtype=np.float32)
        v_raw = np.zeros((B, n_kv, T, hd), dtype=np.float32)
        return hidden, k_rot, v_raw
    return layer_fn


def _make_mock_decode_layer(config):
    """Create a mock decode layer function."""
    def layer_fn(hidden, q_w, k_w, v_w, o_w, gate_w, up_w, down_w, in_gamma, post_gamma,
                 old_k, old_v, write_mask, attn_mask, cos, sin):
        return hidden, old_k, old_v
    return layer_fn


def _make_mock_logits(config, token_sequence):
    """Create a mock logits function that returns tokens from sequence."""
    call_count = [0]
    def logits_fn(hidden, final_gamma, emb_table):
        B = np.asarray(hidden).shape[0]
        V = config.vocab_size
        result = np.full((B, V), -10.0, dtype=np.float32)
        idx = min(call_count[0], len(token_sequence) - 1)
        result[0, token_sequence[idx]] = 10.0
        call_count[0] += 1
        return result
    return logits_fn


# ---------------------------------------------------------------------------
# Tests: build_write_mask
# ---------------------------------------------------------------------------


class TestWriteMask:
    """Tests for the one-hot write mask builder."""

    def test_shape(self):
        mask = build_write_mask(5, 16)
        assert mask.shape == (1, 1, 16, 1)
        assert mask.dtype == np.float32

    def test_one_hot_sum_is_one(self):
        for pos in range(8):
            mask = build_write_mask(pos, 8)
            assert np.sum(mask) == 1.0, f"pos={pos}"

    def test_correct_position(self):
        mask = build_write_mask(3, 8)
        assert mask[0, 0, 3, 0] == 1.0
        for i in range(8):
            if i != 3:
                assert mask[0, 0, i, 0] == 0.0

    def test_position_zero(self):
        mask = build_write_mask(0, 4)
        assert mask[0, 0, 0, 0] == 1.0
        assert mask[0, 0, 1, 0] == 0.0

    def test_last_position(self):
        mask = build_write_mask(7, 8)
        assert mask[0, 0, 7, 0] == 1.0
        assert mask[0, 0, 6, 0] == 0.0

    def test_capacity_zero_raises(self):
        with pytest.raises(ValueError, match="capacity must be > 0"):
            build_write_mask(0, 0)

    def test_negative_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity must be > 0"):
            build_write_mask(0, -1)

    def test_position_out_of_range_raises(self):
        with pytest.raises(ValueError, match="pos must be in"):
            build_write_mask(8, 8)

    def test_negative_position_raises(self):
        with pytest.raises(ValueError, match="pos must be in"):
            build_write_mask(-1, 8)


# ---------------------------------------------------------------------------
# Tests: build_attention_mask
# ---------------------------------------------------------------------------


class TestAttentionMask:
    """Tests for the additive attention mask builder."""

    def test_shape(self):
        mask = build_attention_mask(5, 16)
        assert mask.shape == (1, 1, 1, 16)
        assert mask.dtype == np.float32

    def test_valid_positions_are_zero(self):
        mask = build_attention_mask(3, 8)
        for i in range(4):  # 0, 1, 2, 3
            assert mask[0, 0, 0, i] == 0.0, f"pos={i}"

    def test_invalid_positions_are_neg_inf(self):
        mask = build_attention_mask(3, 8)
        for i in range(4, 8):  # 4, 5, 6, 7
            assert mask[0, 0, 0, i] == -np.inf, f"pos={i}"

    def test_position_zero(self):
        mask = build_attention_mask(0, 4)
        assert mask[0, 0, 0, 0] == 0.0
        for i in range(1, 4):
            assert mask[0, 0, 0, i] == -np.inf

    def test_last_position(self):
        mask = build_attention_mask(7, 8)
        for i in range(8):
            assert mask[0, 0, 0, i] == 0.0

    def test_capacity_zero_raises(self):
        with pytest.raises(ValueError, match="capacity must be > 0"):
            build_attention_mask(0, 0)

    def test_position_out_of_range_raises(self):
        with pytest.raises(ValueError, match="pos must be in"):
            build_attention_mask(8, 8)


# ---------------------------------------------------------------------------
# Tests: layer_weight_args
# ---------------------------------------------------------------------------


class TestLayerWeightArgs:
    """Tests for layer weight unpacking."""

    def test_returns_9_elements(self):
        layer = {
            "wq": np.zeros((1, 1)),
            "wk": np.zeros((1, 1)),
            "wv": np.zeros((1, 1)),
            "wo": np.zeros((1, 1)),
            "w_gate": np.zeros((1, 1)),
            "w_up": np.zeros((1, 1)),
            "w_down": np.zeros((1, 1)),
            "attn_norm": np.zeros(1),
            "ffn_norm": np.zeros(1),
        }
        args = layer_weight_args(layer)
        assert len(args) == 9

    def test_order_matches_signature(self):
        layer = {
            "wq": np.array([1]),
            "wk": np.array([2]),
            "wv": np.array([3]),
            "wo": np.array([4]),
            "w_gate": np.array([5]),
            "w_up": np.array([6]),
            "w_down": np.array([7]),
            "attn_norm": np.array([8]),
            "ffn_norm": np.array([9]),
        }
        args = layer_weight_args(layer)
        assert args[0] == layer["wq"]
        assert args[1] == layer["wk"]
        assert args[2] == layer["wv"]
        assert args[3] == layer["wo"]
        assert args[4] == layer["w_gate"]
        assert args[5] == layer["w_up"]
        assert args[6] == layer["w_down"]
        assert args[7] == layer["attn_norm"]
        assert args[8] == layer["ffn_norm"]


# ---------------------------------------------------------------------------
# Tests: compare_logits
# ---------------------------------------------------------------------------


class TestCompareLogits:
    """Tests for logit comparison metrics."""

    def test_identical_logits(self):
        logits = np.random.default_rng(42).standard_normal(100).astype(np.float32)
        result = compare_logits(logits, logits.copy())
        assert result["argmax_match"] is True
        assert result["top10_overlap"] == 10
        assert result["pearson"] == pytest.approx(1.0, abs=1e-5)
        assert result["centered_cosine"] == pytest.approx(1.0, abs=1e-5)
        assert result["max_abs_diff"] == 0.0
        assert result["mean_abs_diff"] == 0.0

    def test_different_argmax(self):
        pt = np.zeros(100, dtype=np.float32)
        ref = np.zeros(100, dtype=np.float32)
        pt[10] = 10.0
        ref[20] = 10.0
        result = compare_logits(pt, ref)
        assert result["argmax_match"] is False
        assert result["pt_argmax"] == 10
        assert result["ref_argmax"] == 20

    def test_top10_overlap_partial(self):
        pt = np.zeros(100, dtype=np.float32)
        ref = np.zeros(100, dtype=np.float32)
        # Same top 5, different next 5
        for i in range(5):
            pt[90 + i] = 10.0 - i
            ref[90 + i] = 10.0 - i
        for i in range(5, 10):
            pt[80 + i] = 5.0 - (i - 5)
            ref[70 + i] = 5.0 - (i - 5)
        result = compare_logits(pt, ref)
        assert result["top10_overlap"] == 5


# ---------------------------------------------------------------------------
# Tests: compute_n_ctx
# ---------------------------------------------------------------------------


class TestComputeNCtx:
    """Tests for the compute_n_ctx helper."""

    def test_default_config_small_cache(self):
        """Default config (context_length=8192), cache=256 → n_ctx=512."""
        config = SimpleNamespace(context_length=8192)
        assert compute_n_ctx(config, 256) == 512

    def test_default_config_large_cache(self):
        """Default config, cache=1024 → n_ctx=1024."""
        config = SimpleNamespace(context_length=8192)
        assert compute_n_ctx(config, 1024) == 1024

    def test_cache_exceeds_context(self):
        """cache > context_length → n_ctx = context_length."""
        config = SimpleNamespace(context_length=512)
        assert compute_n_ctx(config, 1024) == 512

    def test_minimum_512(self):
        """Even with cache=1, n_ctx >= 512."""
        config = SimpleNamespace(context_length=8192)
        assert compute_n_ctx(config, 1) == 512

    def test_exact_512(self):
        """cache=512 → n_ctx=512."""
        config = SimpleNamespace(context_length=8192)
        assert compute_n_ctx(config, 512) == 512

    def test_small_context_caps(self):
        """Small context_length caps n_ctx even with large cache."""
        config = SimpleNamespace(context_length=256)
        assert compute_n_ctx(config, 4096) == 256


# ---------------------------------------------------------------------------
# Tests: CLI parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    """Tests for CLI argument parsing."""

    def test_minimal(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        args = parse_args(["--model", str(model)])
        assert args.model == model
        assert args.prompt == DEFAULT_PROMPT
        assert args.max_tokens == DEFAULT_MAX_TOKENS
        assert args.cache_capacity == DEFAULT_CACHE_CAPACITY
        assert args.output is None
        assert args.reference is False

    def test_all_options(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        out = tmp_path / "result.json"
        args = parse_args([
            "--model", str(model),
            "--prompt", "Hello",
            "--max-tokens", "32",
            "--cache-capacity", "512",
            "--output", str(out),
            "--reference",
        ])
        assert args.prompt == "Hello"
        assert args.max_tokens == 32
        assert args.cache_capacity == 512
        assert args.output == out
        assert args.reference is True

    def test_requires_model(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_positive_int_rejects_zero(self):
        with pytest.raises(SystemExit):
            parse_args(["--model", "/tmp/m.gguf", "--max-tokens", "0"])

    def test_positive_int_rejects_negative(self):
        with pytest.raises(SystemExit):
            parse_args(["--model", "/tmp/m.gguf", "--max-tokens", "-1"])


# ---------------------------------------------------------------------------
# Tests: run_generation
# ---------------------------------------------------------------------------


class TestGenerationBasic:
    """Basic tests for run_generation."""

    def test_first_token_from_prefill(self):
        config = _make_small_config()
        mock_mlx = _make_mock_mlx()
        mlx_weights = _make_mock_weights(config)

        cos_table = np.ones((20, config.head_dim // 2), dtype=np.float32)
        sin_table = np.zeros((20, config.head_dim // 2), dtype=np.float32)

        # Prefill logits → token 42; decode logits → EOS immediately
        logits_prefill = _make_mock_logits(config, [42])
        logits_decode = _make_mock_logits(config, [2])  # EOS

        modules = {"mlx": MagicMock(), "mlx.core": mock_mlx}
        with patch.dict("sys.modules", modules):
            result = run_generation(
                embed_fn_prefill=_make_mock_embed(config, 3),
                layer_fn_prefill=_make_mock_prefill_layer(config),
                logits_fn_prefill=logits_prefill,
                embed_fn_decode=_make_mock_embed(config, 1),
                layer_fn_decode=_make_mock_decode_layer(config),
                logits_fn_decode=logits_decode,
                token_ids=[1, 2, 3],
                mlx_weights=mlx_weights,
                cos_table=cos_table,
                sin_table=sin_table,
                config=config,
                cache_capacity=32,
                max_tokens=5,
                eos_id=2,
            )

        assert result["first_token"] == 42
        assert result["generated_ids"] == [42]
        assert result["eos_emitted"] is True
        assert result["stop_reason"] == "eos"
        assert result["model_eval_steps"] == 2  # 1 prefill + 1 decode (EOS step)


class TestGenerationEOSStopping:
    """Test that EOS stops generation immediately and is excluded from output."""

    def test_eos_from_prefill_excluded(self):
        """If prefill argmax is EOS, generated_ids is empty, eos_emitted=True."""
        config = _make_small_config()
        mock_mlx = _make_mock_mlx()
        mlx_weights = _make_mock_weights(config)

        cos_table = np.ones((20, config.head_dim // 2), dtype=np.float32)
        sin_table = np.zeros((20, config.head_dim // 2), dtype=np.float32)

        logits_prefill = _make_mock_logits(config, [2])  # EOS from prefill
        logits_decode = _make_mock_logits(config, [42])  # Should not be called

        modules = {"mlx": MagicMock(), "mlx.core": mock_mlx}
        with patch.dict("sys.modules", modules):
            result = run_generation(
                embed_fn_prefill=_make_mock_embed(config, 3),
                layer_fn_prefill=_make_mock_prefill_layer(config),
                logits_fn_prefill=logits_prefill,
                embed_fn_decode=_make_mock_embed(config, 1),
                layer_fn_decode=_make_mock_decode_layer(config),
                logits_fn_decode=logits_decode,
                token_ids=[1, 2, 3],
                mlx_weights=mlx_weights,
                cos_table=cos_table,
                sin_table=sin_table,
                config=config,
                cache_capacity=32,
                max_tokens=5,
                eos_id=2,
            )

        # EOS is excluded from generated_ids
        assert result["generated_ids"] == []
        assert result["first_token"] == 2
        assert result["eos_emitted"] is True
        assert result["stop_reason"] == "eos"
        assert len(result["decode_timings"]) == 0
        assert result["model_eval_steps"] == 1  # only prefill


class TestGenerationMaxTokens:
    """Test that max_tokens limits generation."""

    def test_max_tokens_stops(self):
        config = _make_small_config()
        mock_mlx = _make_mock_mlx()
        mlx_weights = _make_mock_weights(config)

        cos_table = np.ones((20, config.head_dim // 2), dtype=np.float32)
        sin_table = np.zeros((20, config.head_dim // 2), dtype=np.float32)

        # Never emit EOS
        logits_prefill = _make_mock_logits(config, [42])
        logits_decode = _make_mock_logits(config, [43])

        modules = {"mlx": MagicMock(), "mlx.core": mock_mlx}
        with patch.dict("sys.modules", modules):
            result = run_generation(
                embed_fn_prefill=_make_mock_embed(config, 3),
                layer_fn_prefill=_make_mock_prefill_layer(config),
                logits_fn_prefill=logits_prefill,
                embed_fn_decode=_make_mock_embed(config, 1),
                layer_fn_decode=_make_mock_decode_layer(config),
                logits_fn_decode=logits_decode,
                token_ids=[1, 2, 3],
                mlx_weights=mlx_weights,
                cos_table=cos_table,
                sin_table=sin_table,
                config=config,
                cache_capacity=32,
                max_tokens=4,
                eos_id=2,
            )

        # 1 from prefill + 3 from decode = 4 total
        assert len(result["generated_ids"]) == 4
        assert result["generated_ids"][0] == 42
        assert result["stop_reason"] == "max_tokens"
        assert result["eos_emitted"] is False

    def test_max_tokens_one(self):
        """max_tokens=1 yields exactly the prefill token."""
        config = _make_small_config()
        mock_mlx = _make_mock_mlx()
        mlx_weights = _make_mock_weights(config)

        cos_table = np.ones((20, config.head_dim // 2), dtype=np.float32)
        sin_table = np.zeros((20, config.head_dim // 2), dtype=np.float32)

        logits_prefill = _make_mock_logits(config, [42])
        logits_decode = _make_mock_logits(config, [43])

        modules = {"mlx": MagicMock(), "mlx.core": mock_mlx}
        with patch.dict("sys.modules", modules):
            result = run_generation(
                embed_fn_prefill=_make_mock_embed(config, 3),
                layer_fn_prefill=_make_mock_prefill_layer(config),
                logits_fn_prefill=logits_prefill,
                embed_fn_decode=_make_mock_embed(config, 1),
                layer_fn_decode=_make_mock_decode_layer(config),
                logits_fn_decode=logits_decode,
                token_ids=[1, 2, 3],
                mlx_weights=mlx_weights,
                cos_table=cos_table,
                sin_table=sin_table,
                config=config,
                cache_capacity=32,
                max_tokens=1,
                eos_id=2,
            )

        assert result["generated_ids"] == [42]
        assert result["stop_reason"] == "max_tokens"
        assert result["model_eval_steps"] == 1  # only prefill

    def test_max_tokens_zero_raises(self):
        """max_tokens=0 raises ValueError."""
        config = _make_small_config()
        mock_mlx = _make_mock_mlx()
        mlx_weights = _make_mock_weights(config)

        cos_table = np.ones((20, config.head_dim // 2), dtype=np.float32)
        sin_table = np.zeros((20, config.head_dim // 2), dtype=np.float32)

        modules = {"mlx": MagicMock(), "mlx.core": mock_mlx}
        with patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="max_tokens must be >= 1"):
                run_generation(
                    embed_fn_prefill=_make_mock_embed(config, 3),
                    layer_fn_prefill=_make_mock_prefill_layer(config),
                    logits_fn_prefill=_make_mock_logits(config, [42]),
                    embed_fn_decode=_make_mock_embed(config, 1),
                    layer_fn_decode=_make_mock_decode_layer(config),
                    logits_fn_decode=_make_mock_logits(config, [43]),
                    token_ids=[1, 2, 3],
                    mlx_weights=mlx_weights,
                    cos_table=cos_table,
                    sin_table=sin_table,
                    config=config,
                    cache_capacity=32,
                    max_tokens=0,
                    eos_id=2,
                )


class TestGenerationValidation:
    """Test upfront validation in run_generation."""

    def test_empty_token_ids_raises(self):
        """Empty token_ids raises ValueError."""
        config = _make_small_config()
        mock_mlx = _make_mock_mlx()
        mlx_weights = _make_mock_weights(config)

        cos_table = np.ones((20, config.head_dim // 2), dtype=np.float32)
        sin_table = np.zeros((20, config.head_dim // 2), dtype=np.float32)

        modules = {"mlx": MagicMock(), "mlx.core": mock_mlx}
        with patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="token_ids must be nonempty"):
                run_generation(
                    embed_fn_prefill=_make_mock_embed(config, 0),
                    layer_fn_prefill=_make_mock_prefill_layer(config),
                    logits_fn_prefill=_make_mock_logits(config, [42]),
                    embed_fn_decode=_make_mock_embed(config, 1),
                    layer_fn_decode=_make_mock_decode_layer(config),
                    logits_fn_decode=_make_mock_logits(config, [43]),
                    token_ids=[],
                    mlx_weights=mlx_weights,
                    cos_table=cos_table,
                    sin_table=sin_table,
                    config=config,
                    cache_capacity=32,
                    max_tokens=5,
                    eos_id=2,
                )

    def test_token_ids_plus_max_tokens_exceeds_cache_raises(self):
        """T + max_tokens > cache_capacity + 1 raises ValueError."""
        config = _make_small_config()
        mock_mlx = _make_mock_mlx()
        mlx_weights = _make_mock_weights(config)

        cos_table = np.ones((20, config.head_dim // 2), dtype=np.float32)
        sin_table = np.zeros((20, config.head_dim // 2), dtype=np.float32)

        # T=3, max_tokens=10, C=5 → 3+10=13 > 5+1=6
        modules = {"mlx": MagicMock(), "mlx.core": mock_mlx}
        with patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="must be <= cache_capacity"):
                run_generation(
                    embed_fn_prefill=_make_mock_embed(config, 3),
                    layer_fn_prefill=_make_mock_prefill_layer(config),
                    logits_fn_prefill=_make_mock_logits(config, [42]),
                    embed_fn_decode=_make_mock_embed(config, 1),
                    layer_fn_decode=_make_mock_decode_layer(config),
                    logits_fn_decode=_make_mock_logits(config, [43]),
                    token_ids=[1, 2, 3],
                    mlx_weights=mlx_weights,
                    cos_table=cos_table,
                    sin_table=sin_table,
                    config=config,
                    cache_capacity=5,
                    max_tokens=10,
                    eos_id=2,
                )


# ---------------------------------------------------------------------------
# Tests: atomic_write_json
# ---------------------------------------------------------------------------


class TestAtomicWriteJson:
    """Tests for atomic JSON writing."""

    def test_writes_valid_json(self, tmp_path):
        dest = tmp_path / "out.json"
        data = {"key": "value", "num": 42}
        atomic_write_json(data, dest)
        assert dest.exists()
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        assert loaded == data

    def test_rejects_nan(self, tmp_path):
        dest = tmp_path / "out.json"
        with pytest.raises(ValueError):
            atomic_write_json({"bad": float("nan")}, dest)
        assert not dest.exists()

    def test_rejects_inf(self, tmp_path):
        dest = tmp_path / "out.json"
        with pytest.raises(ValueError):
            atomic_write_json({"bad": float("inf")}, dest)
        assert not dest.exists()


# ---------------------------------------------------------------------------
# Integration test (gated by SMOLLM2_GGUF)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("SMOLLM2_GGUF"),
    reason="Set SMOLLM2_GGUF=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf",
)
class TestFullIntegration:
    """End-to-end integration test against the real GGUF artifact.

    Runs max 8 tokens with cache capacity 128.  Asserts finite logits,
    exact generated_ids for the pinned artifact, and strict reference
    comparison metrics.
    """

    # Pinned expected output for SmolLM2-135M-Instruct-Q4_K_M.gguf
    # with prompt "What is 2 + 2? Answer with only the number."
    EXPECTED_GENERATED_IDS = [34, 1232, 216, 34, 446, 216, 36]

    @classmethod
    def setup_class(cls):
        """Load model and run generation once for the class."""
        from llama_cpp import Llama
        from cetagostini.utils.pytensor.gguf_weights import SmolLM2Config, load_smollm2_weights
        from cetagostini.utils.pytensor.smollm2_pytensor import (
            build_rope_table,
            compile_decode_layer,
            compile_embedding,
            compile_logits,
            compile_prefill_layer,
        )

        model_path = Path(os.environ["SMOLLM2_GGUF"])
        cls.model_path = model_path

        # Load llama for tokenization
        cls.llm = Llama(
            model_path=str(model_path),
            n_ctx=512,
            n_gpu_layers=0,
            verbose=False,
            logits_all=True,
        )

        # Format prompt
        from cetagostini.utils.pytensor.run_smollm2_pytensor import format_chat_prompt

        prompt = "What is 2 + 2? Answer with only the number."
        formatted, token_ids, bos_id, eos_id = format_chat_prompt(cls.llm, prompt)
        cls.formatted = formatted
        cls.token_ids = token_ids
        cls.bos_id = bos_id
        cls.eos_id = eos_id
        T = len(token_ids)
        cls.T = T

        config = SmolLM2Config()
        cls.config = config
        C = 128
        cls.C = C
        max_tokens = 8
        cls.max_tokens = max_tokens

        assert bos_id == config.bos, f"BOS mismatch: {bos_id} != {config.bos}"
        assert eos_id == config.eos, f"EOS mismatch: {eos_id} != {config.eos}"

        # Load weights
        weights = load_smollm2_weights(model_path, verify_hash=True)

        # Convert to MLX
        from cetagostini.utils.pytensor.run_smollm2_pytensor import convert_weights_to_mlx

        cls.mlx_weights, _ = convert_weights_to_mlx(weights)

        # RoPE table
        max_seq_len = T + max_tokens + 1
        cls.cos_table, cls.sin_table = build_rope_table(config, max_seq_len)

        # Compile
        embed_fn_prefill = compile_embedding(config, 1, T)
        layer_fn_prefill = compile_prefill_layer(config, 1, T)
        logits_fn_prefill = compile_logits(config, 1, T)
        embed_fn_decode = compile_embedding(config, 1, 1)
        layer_fn_decode = compile_decode_layer(config, 1, C)
        logits_fn_decode = compile_logits(config, 1, 1)

        # Run generation
        cls.gen_result = run_generation(
            embed_fn_prefill,
            layer_fn_prefill,
            logits_fn_prefill,
            embed_fn_decode,
            layer_fn_decode,
            logits_fn_decode,
            token_ids,
            cls.mlx_weights,
            cls.cos_table,
            cls.sin_table,
            config,
            C,
            max_tokens,
            eos_id,
        )

        # Detokenize
        from cetagostini.utils.pytensor.run_smollm2_pytensor import _detokenize_generated

        cls.generated_text = _detokenize_generated(
            cls.llm, cls.gen_result["generated_ids"]
        )

        # Run reference for comparison
        from cetagostini.utils.pytensor.run_smollm2_pytensor import run_reference

        cls.reference = run_reference(
            cls.llm,
            cls.token_ids,
            cls.max_tokens,
            cls.eos_id,
            cls.gen_result["first_logits"],
        )

    def test_bos_eos_confirmed(self):
        assert self.bos_id == 1
        assert self.eos_id == 2

    def test_finite_logits(self):
        first_logits = self.gen_result["first_logits"]
        assert np.all(np.isfinite(first_logits))

    def test_exact_generated_ids(self):
        """PyTensor generated_ids must exactly match the pinned artifact."""
        assert self.gen_result["generated_ids"] == self.EXPECTED_GENERATED_IDS, (
            f"Expected {self.EXPECTED_GENERATED_IDS}, "
            f"got {self.gen_result['generated_ids']}"
        )

    def test_output_contains_expected_text(self):
        text = self.generated_text.strip()
        assert "4" in text or "2 + 2" in text or "2+2" in text, (
            f"Expected '4' or '2 + 2' in output, got: {text!r}"
        )

    def test_no_eos_in_generated_ids(self):
        """EOS must not appear in generated_ids."""
        for tid in self.gen_result["generated_ids"]:
            assert tid != self.eos_id

    def test_cache_status_ok(self):
        assert self.gen_result["cache_status"] == "ok"

    def test_prefill_timing_positive(self):
        assert self.gen_result["prefill_s"] > 0

    def test_reference_argmax_match(self):
        """PyTensor and llama.cpp must agree on argmax."""
        assert self.reference["argmax_match"] is True

    def test_reference_top10_overlap(self):
        """PyTensor and llama.cpp must have perfect top10 overlap."""
        assert self.reference["top10_overlap"] == 10, (
            f"Expected top10_overlap=10, got {self.reference['top10_overlap']}"
        )

    def test_reference_pearson(self):
        """Pearson correlation must be >= 0.999."""
        assert self.reference["pearson"] >= 0.999, (
            f"Expected Pearson >= 0.999, got {self.reference['pearson']}"
        )

    def test_reference_greedy_ids_match(self):
        """PyTensor generated_ids must exactly equal llama greedy_ids."""
        assert self.gen_result["generated_ids"] == self.reference["greedy_ids"], (
            f"PyTensor: {self.gen_result['generated_ids']} != "
            f"llama: {self.reference['greedy_ids']}"
        )
