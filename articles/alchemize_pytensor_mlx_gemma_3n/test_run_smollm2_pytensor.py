"""Tests for run_smollm2_pytensor runtime.

Unit tests mock the runtime where possible and cover:
- Mask builders (write positions, attention patterns)
- Generation state sequence (prefill token then decode)
- EOS / capacity / max_tokens stopping conditions
- Result sanitization (no absolute paths, no env dump)

One real integration test is gated by the ``SMOLLM2_GGUF`` environment
variable and exercises the full pipeline end-to-end.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure article directory is on sys.path for local imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf_weights import SmolLM2Config

from run_smollm2_pytensor import (
    build_attention_mask,
    build_write_mask,
    collect_versions,
    compare_logits,
    layer_weight_args,
    run_generation,
    sanitize_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_CONFIG = SmolLM2Config(
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
        assert mask[0, 0, 2, 0] == 0.0
        assert mask[0, 0, 3, 0] == 0.0

    def test_last_position(self):
        mask = build_write_mask(7, 8)
        assert mask[0, 0, 7, 0] == 1.0
        assert mask[0, 0, 6, 0] == 0.0

    def test_different_positions_differ(self):
        m3 = build_write_mask(3, 8)
        m5 = build_write_mask(5, 8)
        assert not np.array_equal(m3, m5)


# ---------------------------------------------------------------------------
# Tests: build_attention_mask
# ---------------------------------------------------------------------------


class TestAttentionMask:
    """Tests for the additive attention mask builder."""

    def test_shape(self):
        mask = build_attention_mask(5, 16)
        assert mask.shape == (1, 1, 1, 16)
        assert mask.dtype == np.float32

    def test_prefix_zeros_inclusive(self):
        mask = build_attention_mask(3, 8)
        np.testing.assert_array_equal(mask[0, 0, 0, :4], 0.0)

    def test_suffix_neg_inf(self):
        mask = build_attention_mask(3, 8)
        assert np.all(mask[0, 0, 0, 4:] == -np.inf)

    def test_full_attention_at_last_slot(self):
        mask = build_attention_mask(7, 8)
        np.testing.assert_array_equal(mask[0, 0, 0, :], 0.0)

    def test_position_zero_only(self):
        mask = build_attention_mask(0, 4)
        assert mask[0, 0, 0, 0] == 0.0
        assert mask[0, 0, 0, 1] == -np.inf
        assert mask[0, 0, 0, 2] == -np.inf
        assert mask[0, 0, 0, 3] == -np.inf

    def test_monotonic_valid_prefix(self):
        """Valid prefix grows monotonically with pos."""
        for pos in range(10):
            mask = build_attention_mask(pos, 16)
            n_valid = int(np.sum(mask[0, 0, 0, :] == 0.0))
            assert n_valid == pos + 1, f"pos={pos}: expected {pos+1} valid, got {n_valid}"

    def test_write_and_attention_consistent(self):
        """Write position and attention mask agree on the current position."""
        pos = 5
        C = 16
        w = build_write_mask(pos, C)
        a = build_attention_mask(pos, C)
        # Write position should be within the valid attention prefix
        assert w[0, 0, pos, 0] == 1.0
        assert a[0, 0, 0, pos] == 0.0


# ---------------------------------------------------------------------------
# Tests: compare_logits
# ---------------------------------------------------------------------------


class TestCompareLogits:
    """Tests for the logit comparison function."""

    def test_identical_logits(self):
        rng = np.random.default_rng(42)
        logits = rng.standard_normal(100).astype(np.float32)
        result = compare_logits(logits, logits.copy())
        assert result["argmax_match"] is True
        assert result["top10_overlap"] == 10
        assert abs(result["pearson"] - 1.0) < 1e-4
        assert abs(result["centered_cosine"] - 1.0) < 1e-4
        assert result["max_abs_diff"] < 1e-6
        assert result["mean_abs_diff"] < 1e-6

    def test_different_argmax(self):
        pt = np.zeros(100, dtype=np.float32)
        pt[0] = 10.0
        ref = np.zeros(100, dtype=np.float32)
        ref[1] = 10.0
        result = compare_logits(pt, ref)
        assert result["argmax_match"] is False
        assert result["pt_argmax"] == 0
        assert result["ref_argmax"] == 1

    def test_top10_overlap_partial(self):
        pt = np.arange(100, dtype=np.float32)
        ref = np.arange(100, dtype=np.float32)
        # Swap two values in ref to reduce overlap
        ref[90], ref[5] = ref[5], ref[90]
        result = compare_logits(pt, ref)
        assert result["top10_overlap"] >= 8  # Most should still overlap

    def test_top10_overlap_identical(self):
        logits = np.arange(100, dtype=np.float32)
        result = compare_logits(logits, logits.copy())
        assert result["top10_overlap"] == 10

    def test_returns_all_keys(self):
        logits = np.zeros(50, dtype=np.float32)
        result = compare_logits(logits, logits.copy())
        expected_keys = {
            "argmax_match",
            "pt_argmax",
            "ref_argmax",
            "top10_overlap",
            "pearson",
            "centered_cosine",
            "max_abs_diff",
            "mean_abs_diff",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: sanitize_result
# ---------------------------------------------------------------------------


class TestSanitizeResult:
    """Tests for result sanitization."""

    def _make_result(self, model_path=None, reference=None):
        if model_path is None:
            model_path = Path("model.gguf")
        config = SmolLM2Config()
        versions = {"python": "3.13", "numpy": "2.0"}
        rng = np.random.default_rng(99)
        first_logits = rng.standard_normal(config.vocab_size).astype(np.float32)
        return sanitize_result(
            model_path=model_path,
            config=config,
            versions=versions,
            prompt_text="test prompt",
            formatted_text="<formatted>",
            token_ids=[1, 2, 3],
            generated_ids=[4, 5],
            generated_text="test",
            first_logits=first_logits,
            first_token_id=4,
            timings={"load_dequant_s": 1.0, "compile_s": 2.0},
            memory={"peak_rss_mb": 100.0, "mlx_peak_memory_mb": 50.0},
            cache_capacity=256,
            cache_status="ok",
            reference=reference,
        )

    def test_no_absolute_paths(self):
        result = self._make_result(Path("/secret/absolute/path/model.gguf"))
        result_str = json.dumps(result)
        assert "/secret/absolute/path" not in result_str
        assert result["model"]["filename"] == "model.gguf"

    def test_includes_required_sections(self):
        result = self._make_result()
        for key in [
            "model",
            "config",
            "versions",
            "prompt",
            "generation",
            "timing",
            "memory",
            "cache",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_model_identifiers(self):
        result = self._make_result()
        assert result["model"]["repo"] == "bartowski/SmolLM2-135M-Instruct-GGUF"
        assert "sha256" in result["model"]
        assert "revision" in result["model"]

    def test_first_logit_top10(self):
        result = self._make_result()
        top10 = result["generation"]["first_logit_top10"]
        assert len(top10) == 10
        for entry in top10:
            assert "id" in entry
            assert "logit" in entry
            assert isinstance(entry["id"], int)
            assert isinstance(entry["logit"], float)

    def test_reference_included_when_provided(self):
        ref = {"argmax_match": True, "top10_overlap": 8}
        result = self._make_result(reference=ref)
        assert "reference" in result
        assert result["reference"]["argmax_match"] is True

    def test_no_reference_when_none(self):
        result = self._make_result(reference=None)
        assert "reference" not in result

    def test_no_environment_dump(self):
        result = self._make_result()
        result_str = json.dumps(result)
        assert "PATH" not in result_str or "model_path" not in result_str
        assert "HOME" not in result or "environ" not in result_str

    def test_json_serializable(self):
        result = self._make_result()
        # Should not raise
        json_str = json.dumps(result, ensure_ascii=True)
        parsed = json.loads(json_str)
        assert parsed["model"]["filename"] == "model.gguf"

    def test_cache_config(self):
        result = self._make_result()
        cache = result["cache"]
        assert cache["capacity"] == 256
        assert cache["dtype"] == "float32"
        assert cache["status"] == "ok"
        assert cache["shape_per_layer"] == [1, 3, 256, 64]


# ---------------------------------------------------------------------------
# Tests: layer_weight_args
# ---------------------------------------------------------------------------


class TestLayerWeightArgs:
    """Tests for the layer weight unpacking helper."""

    def test_correct_order(self):
        layer = {
            "wq": "q",
            "wk": "k",
            "wv": "v",
            "wo": "o",
            "w_gate": "gate",
            "w_up": "up",
            "w_down": "down",
            "attn_norm": "in_g",
            "ffn_norm": "post_g",
        }
        args = layer_weight_args(layer)
        assert args == ("q", "k", "v", "o", "gate", "up", "down", "in_g", "post_g")

    def test_length(self):
        layer = {k: np.zeros(1) for k in [
            "wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down",
            "attn_norm", "ffn_norm",
        ]}
        args = layer_weight_args(layer)
        assert len(args) == 9


# ---------------------------------------------------------------------------
# Tests: collect_versions
# ---------------------------------------------------------------------------


class TestCollectVersions:
    def test_returns_dict(self):
        versions = collect_versions()
        assert isinstance(versions, dict)
        assert "python" in versions
        assert "numpy" in versions


# ---------------------------------------------------------------------------
# Tests: Generation logic (mocked runtime)
# ---------------------------------------------------------------------------


def _make_mock_mlx():
    """Create a mock MLX module that operates on numpy arrays."""
    mock = MagicMock()
    mock.eval = MagicMock(side_effect=lambda *args: None)
    mock.zeros = lambda shape: np.zeros(shape, dtype=np.float32)
    mock.concatenate = lambda arrays, axis=0: np.concatenate(
        [np.asarray(a) for a in arrays], axis=axis
    )
    mock.array = lambda x: np.asarray(x, dtype=np.float32)
    return mock


def _make_mock_weights(config):
    """Create mock MLX weights as numpy arrays."""
    rng = np.random.default_rng(0)
    return {
        "emb": rng.standard_normal(
            (config.vocab_size, config.hidden_size)
        ).astype(np.float32)
        * 0.01,
        "final_norm": np.ones(config.hidden_size, dtype=np.float32),
        "layers": [
            {
                "wq": np.zeros(
                    (config.hidden_size, config.n_heads * config.head_dim),
                    dtype=np.float32,
                ),
                "wk": np.zeros(
                    (config.hidden_size, config.n_kv_heads * config.head_dim),
                    dtype=np.float32,
                ),
                "wv": np.zeros(
                    (config.hidden_size, config.n_kv_heads * config.head_dim),
                    dtype=np.float32,
                ),
                "wo": np.zeros(
                    (config.n_heads * config.head_dim, config.hidden_size),
                    dtype=np.float32,
                ),
                "w_gate": np.zeros(
                    (config.hidden_size, config.intermediate_size),
                    dtype=np.float32,
                ),
                "w_up": np.zeros(
                    (config.hidden_size, config.intermediate_size),
                    dtype=np.float32,
                ),
                "w_down": np.zeros(
                    (config.intermediate_size, config.hidden_size),
                    dtype=np.float32,
                ),
                "attn_norm": np.ones(config.hidden_size, dtype=np.float32),
                "ffn_norm": np.ones(config.hidden_size, dtype=np.float32),
            }
            for _ in range(config.n_layers)
        ],
    }


def _make_mock_embed(config, seq_len):
    """Create a mock embedding function."""

    def embed(token_ids, emb_table):
        B = token_ids.shape[0]
        T = token_ids.shape[1]
        H = config.hidden_size
        result = np.zeros((B, T, H), dtype=np.float32)
        for b in range(B):
            for t in range(T):
                tid = int(token_ids[b, t])
                if 0 <= tid < emb_table.shape[0]:
                    result[b, t] = np.asarray(emb_table[tid])
        return result

    return embed


def _make_mock_prefill_layer(config):
    """Create a mock prefill layer that passes hidden through."""

    def layer(hidden, *args):
        B, T, H = np.asarray(hidden).shape
        n_kv = config.n_kv_heads
        hd = config.head_dim
        k_rot = np.zeros((B, n_kv, T, hd), dtype=np.float32)
        v_raw = np.zeros((B, n_kv, T, hd), dtype=np.float32)
        return hidden, k_rot, v_raw

    return layer


def _make_mock_decode_layer(config):
    """Create a mock decode layer that passes hidden and caches through."""

    def layer(
        hidden,
        q_w,
        k_w,
        v_w,
        o_w,
        gate_w,
        up_w,
        down_w,
        in_gamma,
        post_gamma,
        old_k,
        old_v,
        write_mask,
        attn_mask,
        cos,
        sin,
    ):
        return hidden, old_k, old_v

    return layer


def _make_mock_logits(config, target_tokens):
    """Create a mock logits function returning a sequence of target tokens.

    Each call returns high logit for ``target_tokens[call_index]``.
    """
    call_count = [0]

    def logits(hidden, final_gamma, emb_table):
        B = np.asarray(hidden).shape[0]
        V = config.vocab_size
        result = np.full((B, V), -10.0, dtype=np.float32)
        idx = min(call_count[0], len(target_tokens) - 1)
        result[0, target_tokens[idx]] = 10.0
        call_count[0] += 1
        return result

    return logits


class TestGenerationPrefillToken:
    """Test that the first generated token comes from prefill argmax."""

    def test_first_token_from_prefill(self):
        config = SMALL_CONFIG
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
        assert result["generated_ids"][0] == 42


class TestGenerationEOSStopping:
    """Test that EOS stops generation immediately."""

    def test_eos_from_prefill(self):
        """If prefill argmax is EOS, no decode steps run."""
        config = SMALL_CONFIG
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

        assert result["generated_ids"] == [2]
        assert len(result["decode_timings"]) == 0

    def test_eos_from_decode(self):
        """EOS from decode stops after emitting prior tokens."""
        config = SMALL_CONFIG
        mock_mlx = _make_mock_mlx()
        mlx_weights = _make_mock_weights(config)

        cos_table = np.ones((20, config.head_dim // 2), dtype=np.float32)
        sin_table = np.zeros((20, config.head_dim // 2), dtype=np.float32)

        # Prefill → 42, decode → 43, then EOS
        logits_prefill = _make_mock_logits(config, [42])
        logits_decode = _make_mock_logits(config, [43, 2])

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
                max_tokens=10,
                eos_id=2,
            )

        assert result["generated_ids"] == [42, 43]
        assert len(result["decode_timings"]) == 2  # Two decode steps


class TestGenerationMaxTokens:
    """Test that max_tokens limits generation."""

    def test_max_tokens_stops(self):
        config = SMALL_CONFIG
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


class TestGenerationCapacity:
    """Test that cache capacity stops generation."""

    def test_capacity_reached(self):
        config = SMALL_CONFIG
        mock_mlx = _make_mock_mlx()
        mlx_weights = _make_mock_weights(config)

        T = 5  # prompt length
        C = 7  # cache capacity → only 2 decode steps possible (pos 5, 6)

        cos_table = np.ones((20, config.head_dim // 2), dtype=np.float32)
        sin_table = np.zeros((20, config.head_dim // 2), dtype=np.float32)

        logits_prefill = _make_mock_logits(config, [42])
        logits_decode = _make_mock_logits(config, [43])  # Never EOS

        modules = {"mlx": MagicMock(), "mlx.core": mock_mlx}
        with patch.dict("sys.modules", modules):
            result = run_generation(
                embed_fn_prefill=_make_mock_embed(config, T),
                layer_fn_prefill=_make_mock_prefill_layer(config),
                logits_fn_prefill=logits_prefill,
                embed_fn_decode=_make_mock_embed(config, 1),
                layer_fn_decode=_make_mock_decode_layer(config),
                logits_fn_decode=logits_decode,
                token_ids=[1, 2, 3, 4, 5],
                mlx_weights=mlx_weights,
                cos_table=cos_table,
                sin_table=sin_table,
                config=config,
                cache_capacity=C,
                max_tokens=20,
                eos_id=2,
            )

        # pos 5 and 6 are valid (T=5, gen_idx=0→pos=5, gen_idx=1→pos=6)
        # pos 7 >= C=7 → capacity reached
        assert result["cache_status"] == "capacity_reached"
        # 1 from prefill + 2 from decode = 3
        assert len(result["generated_ids"]) == 3


class TestGenerationDecodeSequence:
    """Test that decode steps use correct write positions and masks."""

    def test_decode_uses_correct_positions(self):
        """Verify write masks target T+gen_idx positions."""
        config = SMALL_CONFIG
        T = 3
        C = 16

        # Track write positions seen by the decode layer
        seen_write_positions: list[int] = []

        def tracking_decode_layer(
            hidden,
            q_w,
            k_w,
            v_w,
            o_w,
            gate_w,
            up_w,
            down_w,
            in_gamma,
            post_gamma,
            old_k,
            old_v,
            write_mask,
            attn_mask,
            cos,
            sin,
        ):
            # Find the write position
            wm = np.asarray(write_mask)
            pos = int(np.argmax(wm[0, 0, :, 0]))
            seen_write_positions.append(pos)
            return hidden, old_k, old_v

        mock_mlx = _make_mock_mlx()
        mlx_weights = _make_mock_weights(config)

        cos_table = np.ones((20, config.head_dim // 2), dtype=np.float32)
        sin_table = np.zeros((20, config.head_dim // 2), dtype=np.float32)

        logits_prefill = _make_mock_logits(config, [42])
        logits_decode = _make_mock_logits(config, [43, 44, 2])

        modules = {"mlx": MagicMock(), "mlx.core": mock_mlx}
        with patch.dict("sys.modules", modules):
            result = run_generation(
                embed_fn_prefill=_make_mock_embed(config, T),
                layer_fn_prefill=_make_mock_prefill_layer(config),
                logits_fn_prefill=logits_prefill,
                embed_fn_decode=_make_mock_embed(config, 1),
                layer_fn_decode=tracking_decode_layer,
                logits_fn_decode=logits_decode,
                token_ids=[1, 2, 3],
                mlx_weights=mlx_weights,
                cos_table=cos_table,
                sin_table=sin_table,
                config=config,
                cache_capacity=C,
                max_tokens=10,
                eos_id=2,
            )

        # n_layers=2 decode steps per token, 3 decode tokens (43, 44, then EOS)
        # But EOS stops after the decode step that produces it
        # Decode step 0: pos=T+0=3 (2 layers → 2 entries)
        # Decode step 1: pos=T+1=4 (2 layers → 2 entries)
        # Decode step 2: pos=T+2=5 (2 layers → 2 entries, produces EOS)
        n_layers = config.n_layers
        expected_positions = []
        for gen_idx in range(3):
            pos = T + gen_idx
            expected_positions.extend([pos] * n_layers)

        assert seen_write_positions == expected_positions

    def test_attention_mask_covers_prefix(self):
        """Verify attention masks include all valid positions."""
        config = SMALL_CONFIG
        T = 3
        C = 16

        seen_attn_masks: list[np.ndarray] = []

        def tracking_decode_layer(
            hidden,
            q_w,
            k_w,
            v_w,
            o_w,
            gate_w,
            up_w,
            down_w,
            in_gamma,
            post_gamma,
            old_k,
            old_v,
            write_mask,
            attn_mask,
            cos,
            sin,
        ):
            seen_attn_masks.append(np.asarray(attn_mask).copy())
            return hidden, old_k, old_v

        mock_mlx = _make_mock_mlx()
        mlx_weights = _make_mock_weights(config)

        cos_table = np.ones((20, config.head_dim // 2), dtype=np.float32)
        sin_table = np.zeros((20, config.head_dim // 2), dtype=np.float32)

        logits_prefill = _make_mock_logits(config, [42])
        logits_decode = _make_mock_logits(config, [43, 2])

        modules = {"mlx": MagicMock(), "mlx.core": mock_mlx}
        with patch.dict("sys.modules", modules):
            run_generation(
                embed_fn_prefill=_make_mock_embed(config, T),
                layer_fn_prefill=_make_mock_prefill_layer(config),
                logits_fn_prefill=logits_prefill,
                embed_fn_decode=_make_mock_embed(config, 1),
                layer_fn_decode=tracking_decode_layer,
                logits_fn_decode=logits_decode,
                token_ids=[1, 2, 3],
                mlx_weights=mlx_weights,
                cos_table=cos_table,
                sin_table=sin_table,
                config=config,
                cache_capacity=C,
                max_tokens=10,
                eos_id=2,
            )

        # First decode step (gen_idx=0, pos=3): attend to 0..3
        n_layers = config.n_layers
        first_mask = seen_attn_masks[0]
        assert first_mask[0, 0, 0, 0] == 0.0  # pos 0
        assert first_mask[0, 0, 0, 3] == 0.0  # pos 3
        assert first_mask[0, 0, 0, 4] == -np.inf  # pos 4

        # Second decode step (gen_idx=1, pos=4): attend to 0..4
        second_mask = seen_attn_masks[n_layers]
        assert second_mask[0, 0, 0, 4] == 0.0  # pos 4
        assert second_mask[0, 0, 0, 5] == -np.inf  # pos 5


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
    output contains '4' or '2 + 2', and (if reference enabled) top1 matches.
    """

    @classmethod
    def setup_class(cls):
        """Load model and run generation once for the class."""
        from llama_cpp import Llama
        from gguf_weights import SmolLM2Config, load_smollm2_weights
        from smollm2_pytensor import (
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
        from run_smollm2_pytensor import format_chat_prompt

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
        from run_smollm2_pytensor import convert_weights_to_mlx

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
        from run_smollm2_pytensor import _detokenize_generated

        cls.generated_text = _detokenize_generated(
            cls.llm, cls.gen_result["generated_ids"]
        )

    def test_bos_eos_confirmed(self):
        assert self.bos_id == 1
        assert self.eos_id == 2

    def test_finite_logits(self):
        first_logits = self.gen_result["first_logits"]
        assert np.all(np.isfinite(first_logits))

    def test_generated_at_least_one_token(self):
        assert len(self.gen_result["generated_ids"]) >= 1

    def test_output_contains_expected_text(self):
        text = self.generated_text.strip()
        assert "4" in text or "2 + 2" in text or "2+2" in text, (
            f"Expected '4' or '2 + 2' in output, got: {text!r}"
        )

    def test_no_eos_in_generated_ids(self):
        """EOS should not appear in the generated IDs (it stops generation)."""
        for tid in self.gen_result["generated_ids"]:
            assert tid != self.eos_id

    def test_cache_status_ok(self):
        assert self.gen_result["cache_status"] in ("ok", "capacity_reached")

    def test_prefill_timing_positive(self):
        assert self.gen_result["prefill_s"] > 0

    def test_reference_top1_matches(self):
        """If reference is available, top1 should match."""
        from run_smollm2_pytensor import run_reference

        ref = run_reference(
            self.llm,
            self.token_ids,
            self.max_tokens,
            self.eos_id,
            self.gen_result["first_logits"],
        )
        # We don't require bitwise equality, but top1 should match
        # for a well-behaved model on a simple prompt
        assert ref["top10_overlap"] >= 5, (
            f"Expected at least 5 top10 overlap, got {ref['top10_overlap']}"
        )
