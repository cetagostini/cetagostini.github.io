#!/usr/bin/env python3
"""Run SmolLM2-135M-Instruct via PyTensor MLX runtime.

Loads dequantized weights from GGUF, compiles PyTensor graphs for embedding,
prefill layer, decode layer, and logits, then runs greedy autoregressive
generation.  Optionally compares first-logit statistics against a llama.cpp
reference evaluated on identical prompt IDs.

Usage::

    python -m cetagostini.utils.pytensor.run_smollm2_pytensor --model /path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf
    python -m cetagostini.utils.pytensor.run_smollm2_pytensor --model /path/to/model.gguf --reference
    python -m cetagostini.utils.pytensor.run_smollm2_pytensor --model /path/to/model.gguf --output results/run.json
    python -m cetagostini.utils.pytensor.run_smollm2_pytensor --model /path/to/model.gguf --max-tokens 32 --cache-capacity 512
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = "What is 2 + 2? Answer with only the number."
DEFAULT_MAX_TOKENS = 16
DEFAULT_CACHE_CAPACITY = 256


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
        description="Run SmolLM2-135M-Instruct via PyTensor MLX runtime.",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Exact path to the GGUF model file.",
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
        "--cache-capacity",
        type=_positive_int,
        default=DEFAULT_CACHE_CAPACITY,
        help=f"KV-cache slot capacity C (default: {DEFAULT_CACHE_CAPACITY}, must be >= 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON result report.",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Compare against llama.cpp reference on identical prompt IDs.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Chat template formatting & tokenization
# ---------------------------------------------------------------------------


def format_chat_prompt(
    llm: Any,
    prompt_text: str,
) -> tuple[str, list[int], int, int]:
    """Render the GGUF-embedded chat template and tokenize.

    Uses ``llama_cpp.llama_chat_format.Jinja2ChatFormatter`` when available,
    falling back to direct ``jinja2`` rendering.

    Parameters
    ----------
    llm : llama_cpp.Llama
        Loaded Llama model (used only as tokenizer).
    prompt_text : str
        User message text.

    Returns
    -------
    tuple
        ``(formatted_text, token_ids, bos_id, eos_id)``
    """
    chat_template = llm.metadata["tokenizer.chat_template"]
    bos_id = llm.token_bos()
    eos_id = llm.token_eos()

    bos_token = llm._model.token_get_text(bos_id)
    eos_token = llm._model.token_get_text(eos_id)

    formatted = _render_chat_template(
        chat_template,
        messages=[{"role": "user", "content": prompt_text}],
        bos_token=bos_token,
        eos_token=eos_token,
        stop_token_ids=[eos_id],
    )

    raw_tokens = llm.tokenize(formatted.encode("utf-8"), add_bos=False, special=True)
    token_ids = [int(t) for t in raw_tokens]

    return formatted, token_ids, bos_id, eos_id


def _render_chat_template(
    template_str: str,
    messages: list[dict[str, str]],
    bos_token: str,
    eos_token: str,
    stop_token_ids: list[int] | None = None,
) -> str:
    """Render a Jinja2 chat template string.

    Tries ``Jinja2ChatFormatter`` first, falls back to raw ``jinja2``.
    """
    try:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        formatter = Jinja2ChatFormatter(
            template=template_str,
            eos_token=eos_token,
            bos_token=bos_token,
            stop_token_ids=stop_token_ids or [],
        )
        response = formatter(
            messages=messages,
            add_generation_prompt=True,
            bos_token=bos_token,
            eos_token=eos_token,
        )
        return response.prompt
    except Exception:
        pass

    import jinja2

    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    template = env.from_string(template_str)
    return template.render(
        messages=messages,
        add_generation_prompt=True,
        bos_token=bos_token,
        eos_token=eos_token,
    )


def _decode_tokens(llm: Any, token_ids: list[int]) -> str:
    """Detokenize a short list of IDs to a UTF-8 string."""
    raw = llm.detokenize(token_ids)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


# ---------------------------------------------------------------------------
# Weight loading & MLX conversion
# ---------------------------------------------------------------------------


def convert_weights_to_mlx(
    weights: Any,
) -> tuple[dict[str, Any], float]:
    """Convert all numpy weight arrays to ``mlx.core`` arrays and materialize.

    Parameters
    ----------
    weights : SmolLM2Weights
        Loaded weights with numpy arrays.

    Returns
    -------
    tuple
        ``(mlx_weights_dict, elapsed_seconds)``
    """
    import mlx.core as mx

    t0 = time.time()

    emb_mx = mx.array(weights.token_embedding)
    final_norm_mx = mx.array(weights.final_norm)

    layers_mx: list[dict[str, Any]] = []
    for layer in weights.layers:
        layers_mx.append({k: mx.array(v) for k, v in layer.items()})

    all_arrays = [emb_mx, final_norm_mx]
    for layer_dict in layers_mx:
        all_arrays.extend(layer_dict.values())
    mx.eval(*all_arrays)

    elapsed = time.time() - t0
    return {
        "emb": emb_mx,
        "final_norm": final_norm_mx,
        "layers": layers_mx,
    }, elapsed


# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------


def build_write_mask(pos: int, capacity: int) -> np.ndarray:
    """Build a one-hot write mask ``[1, 1, C, 1]``.

    Parameters
    ----------
    pos : int
        Absolute cache position to write.
    capacity : int
        Total cache capacity C.

    Returns
    -------
    np.ndarray
        Float32 array with 1.0 at ``pos`` and 0.0 elsewhere.

    Raises
    ------
    ValueError
        If ``capacity <= 0`` or ``pos`` is out of range ``[0, capacity)``.
    """
    if capacity <= 0:
        raise ValueError(f"capacity must be > 0, got {capacity}")
    if not (0 <= pos < capacity):
        raise ValueError(
            f"pos must be in [0, {capacity}), got {pos}"
        )
    mask = np.zeros((1, 1, capacity, 1), dtype=np.float32)
    mask[0, 0, pos, 0] = 1.0
    return mask


def build_attention_mask(pos: int, capacity: int) -> np.ndarray:
    """Build an additive attention mask ``[1, 1, 1, C]``.

    Positions ``0`` through ``pos`` (inclusive) are ``0.0``; the rest are
    ``-inf``.

    Parameters
    ----------
    pos : int
        Last valid position (inclusive).
    capacity : int
        Total cache capacity C.

    Returns
    -------
    np.ndarray
        Float32 additive mask.

    Raises
    ------
    ValueError
        If ``capacity <= 0`` or ``pos`` is out of range ``[0, capacity)``.
    """
    if capacity <= 0:
        raise ValueError(f"capacity must be > 0, got {capacity}")
    if not (0 <= pos < capacity):
        raise ValueError(
            f"pos must be in [0, {capacity}), got {pos}"
        )
    mask = np.full((1, 1, 1, capacity), -np.inf, dtype=np.float32)
    mask[0, 0, 0, : pos + 1] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Layer weight unpacking
# ---------------------------------------------------------------------------


def layer_weight_args(layer_mx: dict[str, Any]) -> tuple:
    """Unpack a layer weight dict into positional args for compiled functions.

    Parameters
    ----------
    layer_mx : dict
        Keys: ``wq``, ``wk``, ``wv``, ``wo``, ``w_gate``, ``w_up``,
        ``w_down``, ``attn_norm``, ``ffn_norm``.

    Returns
    -------
    tuple
        ``(q_w, k_w, v_w, o_w, gate_w, up_w, down_w, in_gamma, post_gamma)``
    """
    return (
        layer_mx["wq"],
        layer_mx["wk"],
        layer_mx["wv"],
        layer_mx["wo"],
        layer_mx["w_gate"],
        layer_mx["w_up"],
        layer_mx["w_down"],
        layer_mx["attn_norm"],
        layer_mx["ffn_norm"],
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def run_generation(
    embed_fn_prefill,
    layer_fn_prefill,
    logits_fn_prefill,
    embed_fn_decode,
    layer_fn_decode,
    logits_fn_decode,
    token_ids: list[int],
    mlx_weights: dict[str, Any],
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    config: Any,
    cache_capacity: int,
    max_tokens: int,
    eos_id: int,
) -> dict[str, Any]:
    """Run prefill + autoregressive decode generation.

    Parameters
    ----------
    embed_fn_prefill, layer_fn_prefill, logits_fn_prefill : callable
        Compiled prefill functions (fixed T).
    embed_fn_decode, layer_fn_decode, logits_fn_decode : callable
        Compiled decode functions (seq_len=1).
    token_ids : list[int]
        Prompt token IDs (length T).
    mlx_weights : dict
        MLX weight container with keys ``emb``, ``final_norm``, ``layers``.
    cos_table, sin_table : np.ndarray
        RoPE tables of shape ``(max_seq_len, head_dim // 2)``.
    config : SmolLM2Config
        Model configuration.
    cache_capacity : int
        KV-cache slot capacity C.
    max_tokens : int
        Maximum total tokens to generate (including first from prefill).
    eos_id : int
        End-of-sequence token ID.

    Returns
    -------
    dict
        Keys: ``generated_ids``, ``first_token``, ``first_logits``,
        ``prefill_s``, ``decode_timings``, ``cache_status``,
        ``stop_reason``, ``eos_emitted``, ``model_eval_steps``.

    Raises
    ------
    ValueError
        If ``max_tokens < 1``.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    if not token_ids:
        raise ValueError("token_ids must be nonempty")

    T = len(token_ids)
    C = cache_capacity

    # The cache must hold T prompt positions plus up to (max_tokens - 1)
    # decode positions.  The last decode position is T + max_tokens - 2,
    # so we need T + max_tokens - 1 <= C, i.e. T + max_tokens <= C + 1.
    if T + max_tokens > C + 1:
        raise ValueError(
            f"len(token_ids) + max_tokens ({T} + {max_tokens} = {T + max_tokens}) "
            f"must be <= cache_capacity + 1 ({C} + 1 = {C + 1}); "
            f"increase --cache-capacity or reduce --max-tokens"
        )
    if T + max_tokens > config.context_length:
        raise ValueError(
            f"len(token_ids) + max_tokens ({T} + {max_tokens} = {T + max_tokens}) "
            f"exceeds model context_length ({config.context_length})"
        )

    import mlx.core as mx

    n_kv = config.n_kv_heads
    hd = config.head_dim
    n_layers = config.n_layers

    # ── Prefill ──────────────────────────────────────────────────────
    t0 = time.time()
    token_ids_np = np.array([token_ids], dtype=np.int32)
    hidden = embed_fn_prefill(token_ids_np, mlx_weights["emb"])

    prefill_ks: list = []
    prefill_vs: list = []
    cos_prefill = cos_table[:T]
    sin_prefill = sin_table[:T]

    for layer_idx in range(n_layers):
        hidden, k_rot, v_raw = layer_fn_prefill(
            hidden,
            *layer_weight_args(mlx_weights["layers"][layer_idx]),
            cos_prefill,
            sin_prefill,
        )
        prefill_ks.append(k_rot)
        prefill_vs.append(v_raw)

    mx.eval(hidden, *prefill_ks, *prefill_vs)

    logits = logits_fn_prefill(
        hidden, mlx_weights["final_norm"], mlx_weights["emb"]
    )
    mx.eval(logits)
    logits_np = np.asarray(logits, dtype=np.float32)
    first_logits = logits_np[0].copy()
    if not np.all(np.isfinite(first_logits)):
        n_bad = int(np.sum(~np.isfinite(first_logits)))
        raise ValueError(
            f"Prefill logits contain {n_bad} non-finite values "
            f"(NaN or Inf); aborting generation"
        )
    first_token = int(np.argmax(first_logits))
    t_prefill = time.time() - t0

    # ── Build fixed float32 caches [1, n_kv, C, hd] ─────────────────
    caches_k: list = []
    caches_v: list = []
    for k, v in zip(prefill_ks, prefill_vs):
        if T < C:
            pad_k = mx.zeros((1, n_kv, C - T, hd))
            pad_v = mx.zeros((1, n_kv, C - T, hd))
            cache_k = mx.concatenate([k, pad_k], axis=2)
            cache_v = mx.concatenate([v, pad_v], axis=2)
        else:
            cache_k = k[:, :, :C, :]
            cache_v = v[:, :, :C, :]
        caches_k.append(cache_k)
        caches_v.append(cache_v)
    mx.eval(*caches_k, *caches_v)

    # ── Decode loop ──────────────────────────────────────────────────
    # EOS policy: exclude EOS from generated_ids whether from prefill or decode
    generated_ids: list[int] = []
    decode_timings: list[float] = []
    stop_reason = "max_tokens"
    eos_emitted = False
    model_eval_steps = 0  # total model forward passes (prefill + decode)

    # Prefill counts as one model eval step
    model_eval_steps += 1

    # If prefill argmax is EOS, stop immediately (do not add to generated_ids)
    if first_token == eos_id:
        eos_emitted = True
        stop_reason = "eos"
    elif len(generated_ids) < max_tokens:
        # Add first token from prefill (not EOS)
        generated_ids.append(first_token)
        current_token = first_token

        for gen_idx in range(max_tokens - 1):
            pos = T + gen_idx
            if pos >= C:
                stop_reason = "capacity_reached"
                break

            t_step = time.time()

            write_mask = build_write_mask(pos, C)
            attn_mask = build_attention_mask(pos, C)
            cos_row = cos_table[pos : pos + 1]
            sin_row = sin_table[pos : pos + 1]

            token_np = np.array([[current_token]], dtype=np.int32)
            hidden = embed_fn_decode(token_np, mlx_weights["emb"])

            # NOTE: The decode layer uses pt.where for cache updates, which
            # is O(capacity) per layer per step. This is demonstrator-only;
            # a production runtime would use in-place scatter/gather or
            # ring-buffer indexing for O(1) updates.
            for layer_idx in range(n_layers):
                hidden, new_k, new_v = layer_fn_decode(
                    hidden,
                    *layer_weight_args(mlx_weights["layers"][layer_idx]),
                    caches_k[layer_idx],
                    caches_v[layer_idx],
                    write_mask,
                    attn_mask,
                    cos_row,
                    sin_row,
                )
                caches_k[layer_idx] = new_k
                caches_v[layer_idx] = new_v

            mx.eval(hidden, *caches_k, *caches_v)
            model_eval_steps += 1

            logits = logits_fn_decode(
                hidden, mlx_weights["final_norm"], mlx_weights["emb"]
            )
            mx.eval(logits)
            logits_np = np.asarray(logits, dtype=np.float32)
            if not np.all(np.isfinite(logits_np)):
                n_bad = int(np.sum(~np.isfinite(logits_np)))
                raise ValueError(
                    f"Decode logits at step {gen_idx} contain {n_bad} "
                    f"non-finite values (NaN or Inf); aborting generation"
                )

            next_token = int(np.argmax(logits_np[0]))
            decode_timings.append(time.time() - t_step)

            if next_token == eos_id:
                eos_emitted = True
                stop_reason = "eos"
                break

            generated_ids.append(next_token)
            current_token = next_token

    return {
        "generated_ids": generated_ids,
        "first_token": first_token,
        "first_logits": first_logits,
        "prefill_s": t_prefill,
        "decode_timings": decode_timings,
        "cache_status": "ok" if stop_reason != "capacity_reached" else "capacity_reached",
        "stop_reason": stop_reason,
        "eos_emitted": eos_emitted,
        "model_eval_steps": model_eval_steps,
    }


# ---------------------------------------------------------------------------
# llama.cpp helpers
# ---------------------------------------------------------------------------


def get_llama_logits(llm: Any) -> np.ndarray:
    """Extract next-token logits from a llama model after ``eval``.

    Uses ``llm._scores`` which has shape ``(n_evaluated, vocab_size)`` and
    contains the actual logits for all evaluated tokens.  Falls back to
    ``llm.scores`` indexed by ``llm.n_tokens - 1`` for compatibility.

    Parameters
    ----------
    llm : llama_cpp.Llama
        Model after ``eval`` (must be created with ``logits_all=True``).

    Returns
    -------
    np.ndarray
        Logits array of shape ``(vocab_size,)``.
    """
    # Prefer _scores which has the correct (n_evaluated, vocab) shape
    try:
        s = llm._scores
        if s is not None:
            arr = np.asarray(s, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] > 0:
                return arr[-1].copy()
    except (AttributeError, TypeError, ValueError):
        pass

    # Fallback: use scores buffer indexed by n_tokens
    try:
        n_tok = llm.n_tokens
        s = llm.scores
        if s is not None and n_tok > 0:
            arr = np.asarray(s, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= n_tok:
                return arr[n_tok - 1].copy()
    except (AttributeError, TypeError, ValueError):
        pass

    raise RuntimeError("Cannot extract logits from llama model")


# ---------------------------------------------------------------------------
# Logit comparison
# ---------------------------------------------------------------------------


def compare_logits(
    pt_logits: np.ndarray,
    ref_logits: np.ndarray,
) -> dict[str, Any]:
    """Compare PyTensor and reference first-token logits.

    Parameters
    ----------
    pt_logits : np.ndarray
        PyTensor logits, shape ``(vocab_size,)``.
    ref_logits : np.ndarray
        Reference logits, shape ``(vocab_size,)``.

    Returns
    -------
    dict
        Keys: ``argmax_match``, ``pt_argmax``, ``ref_argmax``,
        ``top10_overlap``, ``pearson``, ``centered_cosine``,
        ``max_abs_diff``, ``mean_abs_diff``.
    """
    pt_argmax = int(np.argmax(pt_logits))
    ref_argmax = int(np.argmax(ref_logits))

    pt_top10 = set(np.argsort(pt_logits)[-10:].tolist())
    ref_top10 = set(np.argsort(ref_logits)[-10:].tolist())
    top10_overlap = len(pt_top10 & ref_top10)

    pt_std = float(np.std(pt_logits))
    ref_std = float(np.std(ref_logits))
    if pt_std > 0 and ref_std > 0:
        pearson = float(np.corrcoef(pt_logits, ref_logits)[0, 1])
    else:
        pearson = 0.0

    pt_c = pt_logits - np.mean(pt_logits)
    ref_c = ref_logits - np.mean(ref_logits)
    pt_norm = float(np.linalg.norm(pt_c))
    ref_norm = float(np.linalg.norm(ref_c))
    if pt_norm > 0 and ref_norm > 0:
        centered_cosine = float(np.dot(pt_c, ref_c) / (pt_norm * ref_norm))
    else:
        centered_cosine = 0.0

    abs_diff = np.abs(pt_logits.astype(np.float64) - ref_logits.astype(np.float64))
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    return {
        "argmax_match": pt_argmax == ref_argmax,
        "pt_argmax": pt_argmax,
        "ref_argmax": ref_argmax,
        "top10_overlap": top10_overlap,
        "pearson": round(pearson, 6),
        "centered_cosine": round(centered_cosine, 6),
        "max_abs_diff": round(max_abs_diff, 6),
        "mean_abs_diff": round(mean_abs_diff, 6),
    }


# ---------------------------------------------------------------------------
# Reference generation
# ---------------------------------------------------------------------------


def run_reference(
    llm: Any,
    prompt_ids: list[int],
    max_tokens: int,
    eos_id: int,
    pt_first_logits: np.ndarray,
) -> dict[str, Any]:
    """Run llama.cpp reference and compare first logits.

    Parameters
    ----------
    llm : llama_cpp.Llama
        Llama model instance.
    prompt_ids : list[int]
        Prompt token IDs (identical to PyTensor run).
    max_tokens : int
        Maximum tokens for greedy continuation.
    eos_id : int
        EOS token ID.
    pt_first_logits : np.ndarray
        PyTensor first-token logits for comparison.

    Returns
    -------
    dict
        Comparison metrics plus greedy continuation.
    """
    llm.reset()
    llm.eval(list(prompt_ids))

    ref_first_logits = get_llama_logits(llm)
    comparison = compare_logits(pt_first_logits, ref_first_logits)

    ref_generated: list[int] = []
    ref_argmax = int(np.argmax(ref_first_logits))
    if ref_argmax != eos_id:
        ref_generated.append(ref_argmax)
        llm.eval([ref_argmax])
        for _ in range(max_tokens - 1):
            logits = get_llama_logits(llm)
            next_token = int(np.argmax(logits))
            if next_token == eos_id:
                break
            ref_generated.append(next_token)
            llm.eval([next_token])

    if ref_generated:
        ref_text = _detokenize_generated(llm, ref_generated)
    else:
        ref_text = ""

    comparison["greedy_text"] = ref_text
    comparison["greedy_ids"] = ref_generated
    return comparison


def _detokenize_generated(llm: Any, token_ids: list[int]) -> str:
    """Detokenize generated IDs, preserving special-token behavior."""
    try:
        raw = llm.detokenize(token_ids, special=True)
    except TypeError:
        raw = llm.detokenize(token_ids)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


# ---------------------------------------------------------------------------
# Version & memory helpers
# ---------------------------------------------------------------------------


def collect_versions() -> dict[str, str]:
    """Collect package versions for the result report.

    Uses ``importlib.metadata`` for packages that don't expose
    ``__version__`` at the module level (e.g., MLX).
    """
    import importlib.metadata

    versions: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for mod_name, attr in [
        ("pytensor", "__version__"),
        ("llama_cpp", "__version__"),
        ("pytensor_ml", "__version__"),
    ]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, attr, "unknown")
        except Exception:
            versions[mod_name] = "unavailable"

    # MLX does not expose __version__ at module level; use importlib.metadata
    try:
        versions["mlx"] = importlib.metadata.version("mlx")
    except Exception:
        versions["mlx"] = "unavailable"

    return versions


def get_peak_rss_mb() -> float:
    """Get peak RSS in megabytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return round(rss / (1024 * 1024), 2)
    return round(rss / 1024, 2)


def get_mlx_peak_memory_mb() -> float | None:
    """Get MLX peak memory in megabytes, or None if unavailable."""
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
# Result sanitization
# ---------------------------------------------------------------------------


def sanitize_result(
    model_path: Path,
    config: Any,
    versions: dict[str, str],
    prompt_text: str,
    formatted_text: str,
    token_ids: list[int],
    generated_ids: list[int],
    generated_text: str,
    first_logits: np.ndarray,
    first_token_id: int,
    timings: dict[str, Any],
    memory: dict[str, Any],
    cache_capacity: int,
    cache_status: str,
    reference: dict[str, Any] | None = None,
    stop_reason: str = "max_tokens",
    eos_emitted: bool = False,
    model_eval_steps: int = 0,
) -> dict[str, Any]:
    """Build a sanitized JSON result without absolute paths or env dumps.

    Parameters
    ----------
    model_path : Path
        GGUF file path (only basename is used).
    config : SmolLM2Config
        Model configuration.
    versions : dict
        Package versions.
    prompt_text : str
        Original user prompt.
    formatted_text : str
        Chat-template-rendered prompt.
    token_ids : list[int]
        Prompt token IDs.
    generated_ids : list[int]
        Generated token IDs (EOS excluded).
    generated_text : str
        Detokenized generated text.
    first_logits : np.ndarray
        First-token logits vector.
    first_token_id : int
        Argmax of first logits (may be EOS).
    timings : dict
        Timing phases.
    memory : dict
        Memory metrics.
    cache_capacity : int
        KV-cache capacity.
    cache_status : str
        Cache status string.
    reference : dict or None
        Reference comparison metrics.
    stop_reason : str
        Why generation stopped: ``'eos'``, ``'max_tokens'``, or
        ``'capacity_reached'``.
    eos_emitted : bool
        Whether the model selected EOS at any point.
    model_eval_steps : int
        Total model forward passes (1 prefill + N decode).

    Returns
    -------
    dict
        Sanitized result dictionary.
    """
    from cetagostini.utils.pytensor.gguf_weights import (
        EXPECTED_ARCHITECTURE,
        EXPECTED_GGUF_VERSION,
        EXPECTED_REPO,
        EXPECTED_REVISION,
        EXPECTED_SHA256,
    )

    if not np.all(np.isfinite(first_logits)):
        n_bad = int(np.sum(~np.isfinite(first_logits)))
        raise ValueError(
            f"sanitize_result: first_logits contains {n_bad} non-finite "
            f"values; refusing to serialize"
        )

    top10_indices = np.argsort(first_logits)[-10:][::-1]
    top10 = [
        {"id": int(idx), "logit": round(float(first_logits[idx]), 4)}
        for idx in top10_indices
    ]

    result: dict[str, Any] = {
        "model": {
            "repo": EXPECTED_REPO,
            "revision": EXPECTED_REVISION,
            "filename": model_path.name,
            "sha256": EXPECTED_SHA256,
            "architecture": EXPECTED_ARCHITECTURE,
            "gguf_version": EXPECTED_GGUF_VERSION,
        },
        "config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "head_dim": config.head_dim,
            "intermediate_size": config.intermediate_size,
            "context_length": config.context_length,
            "rms_eps": config.rms_eps,
            "rope_theta": config.rope_theta,
            "bos": config.bos,
            "eos": config.eos,
        },
        "versions": versions,
        "prompt": {
            "text": prompt_text,
            "formatted": formatted_text,
            "token_ids": token_ids,
            "n_tokens": len(token_ids),
        },
        "generation": {
            "generated_ids": generated_ids,
            "text": generated_text,
            "n_tokens": len(generated_ids),
            "first_token_id": first_token_id,
            "first_logit_top10": top10,
            "stop_reason": stop_reason,
            "eos_emitted": eos_emitted,
            "model_eval_steps": model_eval_steps,
        },
        "timing": timings,
        "memory": memory,
        "cache": {
            "capacity": cache_capacity,
            "n_layers": config.n_layers,
            "dtype": "float32",
            "shape_per_layer": [
                1,
                config.n_kv_heads,
                cache_capacity,
                config.head_dim,
            ],
            "status": cache_status,
        },
    }

    if reference is not None:
        result["reference"] = reference

    return result


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
        dir=str(dest.parent), suffix=".tmp", prefix=".run_"
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


def compute_n_ctx(config: Any, cache_capacity: int) -> int:
    """Compute the llama.cpp context size.

    Returns ``min(context_length, max(512, cache_capacity))`` so that the
    reference model can handle prompts up to the configured cache capacity
    while never exceeding the model's native context length.

    Parameters
    ----------
    config : SmolLM2Config
        Model configuration (provides ``context_length``).
    cache_capacity : int
        KV-cache slot capacity.

    Returns
    -------
    int
        Context size for llama.cpp.
    """
    return min(config.context_length, max(512, cache_capacity))


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
    model_name = args.model.name  # preserve original filename (HF cache symlinks)

    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        return 1

    # ── Phase 0: Instantiate config & validate capacity ──────────────
    # Config is needed before llama load to compute n_ctx and validate
    # cache_capacity against context_length.
    from cetagostini.utils.pytensor.gguf_weights import SmolLM2Config

    config = SmolLM2Config()

    if args.cache_capacity > config.context_length:
        print(
            f"ERROR: cache_capacity ({args.cache_capacity}) exceeds model "
            f"context_length ({config.context_length}). "
            f"Reduce --cache-capacity.",
            file=sys.stderr,
        )
        return 1

    # ── Phase 0b: Set PyTensor floatX before any graph compilation ───
    # This is executable-owned state. The library (smollm2_pytensor) also
    # sets this at import time, but the executable asserts it explicitly
    # to make the contract visible and independent of import order.
    import pytensor

    pytensor.config.floatX = "float32"

    # ── Phase 1: Load llama model for tokenization ───────────────────
    n_ctx = compute_n_ctx(config, args.cache_capacity)
    print(f"Loading llama model for tokenization (n_ctx={n_ctx})...")
    from llama_cpp import Llama

    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=0,
        verbose=False,
        logits_all=True,
    )

    # ── Phase 2: Format chat template & tokenize ─────────────────────
    print("Formatting chat template...")
    formatted, token_ids, bos_id, eos_id = format_chat_prompt(llm, args.prompt)
    T = len(token_ids)
    print(f"  {T} tokens, BOS={bos_id}, EOS={eos_id}")

    if bos_id != config.bos:
        print(
            f"WARNING: BOS mismatch: llama={bos_id}, config={config.bos}",
            file=sys.stderr,
        )
    if eos_id != config.eos:
        print(
            f"WARNING: EOS mismatch: llama={eos_id}, config={config.eos}",
            file=sys.stderr,
        )

    if T + args.max_tokens > args.cache_capacity + 1:
        print(
            f"ERROR: len(token_ids) + max_tokens ({T} + {args.max_tokens} = "
            f"{T + args.max_tokens}) must be <= cache_capacity + 1 "
            f"({args.cache_capacity} + 1 = {args.cache_capacity + 1}). "
            f"Increase --cache-capacity or reduce --max-tokens.",
            file=sys.stderr,
        )
        return 1

    if T + args.max_tokens > config.context_length:
        print(
            f"ERROR: len(token_ids) + max_tokens ({T} + {args.max_tokens} = "
            f"{T + args.max_tokens}) exceeds model context_length "
            f"({config.context_length}).",
            file=sys.stderr,
        )
        return 1

    # ── Phase 3: Load & dequantize weights ───────────────────────────
    # The GGUF reader uses mmap for input-only access to the file; all
    # tensors are dequantized into fresh float32 numpy arrays below.
    print("Loading GGUF weights (mmap input + dequant to float32)...")
    from cetagostini.utils.pytensor.gguf_weights import load_smollm2_weights

    t0 = time.time()
    weights = load_smollm2_weights(model_path, verify_hash=True)
    t_load = time.time() - t0
    print(f"  loaded in {t_load:.2f}s ({len(weights.layers)} layers)")

    # ── Phase 4: Convert to MLX arrays ───────────────────────────────
    print("Converting weights to MLX arrays...")
    mlx_weights, t_mlx = convert_weights_to_mlx(weights)
    print(f"  converted in {t_mlx:.2f}s")

    # Release the numpy weight container now that all arrays are in MLX.
    # The GGUFReader mmap stays alive via weights.reader, but the dequantized
    # numpy arrays are no longer needed.
    del weights
    gc.collect()

    # ── Phase 5: Build RoPE table ────────────────────────────────────
    from cetagostini.utils.pytensor.smollm2_pytensor import build_rope_table

    max_seq_len = T + args.max_tokens + 1
    cos_table, sin_table = build_rope_table(config, max_seq_len)

    # ── Phase 6: Compile PyTensor functions ──────────────────────────
    print("Compiling PyTensor functions...")
    from cetagostini.utils.pytensor.smollm2_pytensor import (
        compile_decode_layer,
        compile_embedding,
        compile_logits,
        compile_prefill_layer,
    )

    t0 = time.time()
    embed_fn_prefill = compile_embedding(config, 1, T)
    layer_fn_prefill = compile_prefill_layer(config, 1, T)
    logits_fn_prefill = compile_logits(config, 1, T)
    embed_fn_decode = compile_embedding(config, 1, 1)
    layer_fn_decode = compile_decode_layer(config, 1, args.cache_capacity)
    logits_fn_decode = compile_logits(config, 1, 1)
    t_compile = time.time() - t0
    print(f"  compiled in {t_compile:.2f}s")

    # ── Phase 7: Run generation ──────────────────────────────────────
    print("Running generation...")
    gen_result = run_generation(
        embed_fn_prefill,
        layer_fn_prefill,
        logits_fn_prefill,
        embed_fn_decode,
        layer_fn_decode,
        logits_fn_decode,
        token_ids,
        mlx_weights,
        cos_table,
        sin_table,
        config,
        args.cache_capacity,
        args.max_tokens,
        eos_id,
    )

    generated_ids = gen_result["generated_ids"]
    first_token = gen_result["first_token"]
    first_logits = gen_result["first_logits"]
    t_prefill = gen_result["prefill_s"]
    decode_timings = gen_result["decode_timings"]
    cache_status = gen_result["cache_status"]
    stop_reason = gen_result["stop_reason"]
    eos_emitted = gen_result["eos_emitted"]
    model_eval_steps = gen_result["model_eval_steps"]

    print(f"  prefill: {t_prefill:.2f}s, first token: {first_token}")
    print(f"  generated {len(generated_ids)} tokens (stop: {stop_reason})")

    # ── Phase 8: Detokenize ──────────────────────────────────────────
    generated_text = _detokenize_generated(llm, generated_ids)
    print(f"  text: {generated_text!r}")

    # ── Phase 9: Reference (optional) ────────────────────────────────
    reference = None
    if args.reference:
        print("Running llama.cpp reference...")
        reference = run_reference(
            llm, token_ids, args.max_tokens, eos_id, first_logits
        )
        print(f"  argmax match: {reference['argmax_match']}")
        print(f"  top10 overlap: {reference['top10_overlap']}")
        print(f"  reference text: {reference.get('greedy_text', '')!r}")

    # ── Phase 10: Build result ───────────────────────────────────────
    versions = collect_versions()
    t_decode_total = sum(decode_timings) if decode_timings else 0.0
    # All timings are from the first (cold) run — no warmup or caching
    # of compiled graphs is performed.
    timings = {
        "cold_first_run": True,
        "load_dequant_s": round(t_load, 3),
        "mlx_convert_s": round(t_mlx, 3),
        "compile_s": round(t_compile, 3),
        "prefill_s": round(t_prefill, 3),
        "decode_total_s": round(t_decode_total, 3),
        "decode_per_token_s": [round(t, 4) for t in decode_timings],
    }
    memory = {
        "peak_rss_mb": get_peak_rss_mb(),
        "mlx_peak_memory_mb": get_mlx_peak_memory_mb(),
    }

    result = sanitize_result(
        Path(model_name),
        config,
        versions,
        args.prompt,
        formatted,
        token_ids,
        generated_ids,
        generated_text,
        first_logits,
        first_token,
        timings,
        memory,
        args.cache_capacity,
        cache_status,
        reference,
        stop_reason=stop_reason,
        eos_emitted=eos_emitted,
        model_eval_steps=model_eval_steps,
    )

    # ── Phase 11: Write output ───────────────────────────────────────
    if args.output:
        atomic_write_json(result, args.output)
        print(f"Report written to {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=True))

    return 0


# ---------------------------------------------------------------------------
# Reusable inference entry point
# ---------------------------------------------------------------------------


def run_inference(
    model_path: str | Path,
    *,
    prompt: str,
    max_tokens: int,
    cache_capacity: int,
    reference: bool,
    verify_hash: bool,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run SmolLM2-135M-Instruct inference from Python.

    Parameters
    ----------
    model_path : str or Path
        Exact path to the GGUF model file.
    prompt : str
        User prompt text.
    max_tokens : int
        Maximum tokens to generate (must be >= 1).
    cache_capacity : int
        KV-cache slot capacity C (must be >= 1).
    reference : bool
        Compare against llama.cpp reference on identical prompt IDs.
    verify_hash : bool
        Verify the GGUF file SHA-256 hash against the pinned value.
    progress : callable or None
        Optional callback receiving progress messages.

    Returns
    -------
    dict
        Sanitized result dictionary.
    """
    from typing import Callable

    model_path = Path(model_path).resolve()
    model_name = model_path.name

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # ── Phase 0: Instantiate config & validate capacity ──────────────
    from cetagostini.utils.pytensor.gguf_weights import SmolLM2Config

    config = SmolLM2Config()

    if cache_capacity > config.context_length:
        raise ValueError(
            f"cache_capacity ({cache_capacity}) exceeds model "
            f"context_length ({config.context_length})"
        )

    # ── Phase 0b: Set PyTensor floatX before any graph compilation ───
    import pytensor

    pytensor.config.floatX = "float32"

    # ── Phase 1: Load llama model for tokenization ───────────────────
    if progress:
        progress("Loading llama model for tokenization")
    n_ctx = compute_n_ctx(config, cache_capacity)
    from llama_cpp import Llama

    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=0,
        verbose=False,
        logits_all=True,
    )

    # ── Phase 2: Format chat template & tokenize ─────────────────────
    if progress:
        progress("Formatting chat template")
    formatted, token_ids, bos_id, eos_id = format_chat_prompt(llm, prompt)
    T = len(token_ids)

    if T + max_tokens > cache_capacity + 1:
        raise ValueError(
            f"len(token_ids) + max_tokens ({T} + {max_tokens} = {T + max_tokens}) "
            f"must be <= cache_capacity + 1 ({cache_capacity} + 1)"
        )

    if T + max_tokens > config.context_length:
        raise ValueError(
            f"len(token_ids) + max_tokens ({T} + {max_tokens} = {T + max_tokens}) "
            f"exceeds model context_length ({config.context_length})"
        )

    # ── Phase 3: Load & dequantize weights ───────────────────────────
    if progress:
        progress("Loading GGUF weights")
    from cetagostini.utils.pytensor.gguf_weights import load_smollm2_weights

    t0 = time.time()
    weights = load_smollm2_weights(model_path, verify_hash=verify_hash)
    t_load = time.time() - t0

    # ── Phase 4: Convert to MLX arrays ───────────────────────────────
    if progress:
        progress("Converting weights to MLX arrays")
    mlx_weights, t_mlx = convert_weights_to_mlx(weights)

    del weights
    gc.collect()

    # ── Phase 5: Build RoPE table ────────────────────────────────────
    from cetagostini.utils.pytensor.smollm2_pytensor import build_rope_table

    max_seq_len = T + max_tokens + 1
    cos_table, sin_table = build_rope_table(config, max_seq_len)

    # ── Phase 6: Compile PyTensor functions ──────────────────────────
    if progress:
        progress("Compiling PyTensor functions")
    from cetagostini.utils.pytensor.smollm2_pytensor import (
        compile_decode_layer,
        compile_embedding,
        compile_logits,
        compile_prefill_layer,
    )

    t0 = time.time()
    embed_fn_prefill = compile_embedding(config, 1, T)
    layer_fn_prefill = compile_prefill_layer(config, 1, T)
    logits_fn_prefill = compile_logits(config, 1, T)
    embed_fn_decode = compile_embedding(config, 1, 1)
    layer_fn_decode = compile_decode_layer(config, 1, cache_capacity)
    logits_fn_decode = compile_logits(config, 1, 1)
    t_compile = time.time() - t0

    # ── Phase 7: Run generation ──────────────────────────────────────
    if progress:
        progress("Running generation")
    gen_result = run_generation(
        embed_fn_prefill,
        layer_fn_prefill,
        logits_fn_prefill,
        embed_fn_decode,
        layer_fn_decode,
        logits_fn_decode,
        token_ids,
        mlx_weights,
        cos_table,
        sin_table,
        config,
        cache_capacity,
        max_tokens,
        eos_id,
    )

    generated_ids = gen_result["generated_ids"]
    first_token = gen_result["first_token"]
    first_logits = gen_result["first_logits"]
    t_prefill = gen_result["prefill_s"]
    decode_timings = gen_result["decode_timings"]
    cache_status = gen_result["cache_status"]
    stop_reason = gen_result["stop_reason"]
    eos_emitted = gen_result["eos_emitted"]
    model_eval_steps = gen_result["model_eval_steps"]

    # ── Phase 8: Detokenize ──────────────────────────────────────────
    generated_text = _detokenize_generated(llm, generated_ids)

    # ── Phase 9: Reference (optional) ────────────────────────────────
    reference_result = None
    if reference:
        if progress:
            progress("Running llama.cpp reference")
        reference_result = run_reference(
            llm, token_ids, max_tokens, eos_id, first_logits
        )

    # ── Phase 10: Build result ───────────────────────────────────────
    versions = collect_versions()
    t_decode_total = sum(decode_timings) if decode_timings else 0.0
    timings = {
        "cold_first_run": True,
        "load_dequant_s": round(t_load, 3),
        "mlx_convert_s": round(t_mlx, 3),
        "compile_s": round(t_compile, 3),
        "prefill_s": round(t_prefill, 3),
        "decode_total_s": round(t_decode_total, 3),
        "decode_per_token_s": [round(t, 4) for t in decode_timings],
    }
    memory = {
        "peak_rss_mb": get_peak_rss_mb(),
        "mlx_peak_memory_mb": get_mlx_peak_memory_mb(),
    }

    result = sanitize_result(
        Path(model_name),
        config,
        versions,
        prompt,
        formatted,
        token_ids,
        generated_ids,
        generated_text,
        first_logits,
        first_token,
        timings,
        memory,
        cache_capacity,
        cache_status,
        reference_result,
        stop_reason=stop_reason,
        eos_emitted=eos_emitted,
        model_eval_steps=model_eval_steps,
    )

    return result


if __name__ == "__main__":
    sys.exit(main())
