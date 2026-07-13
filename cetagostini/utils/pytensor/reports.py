"""Structured reports returned by the PyTensor inference API."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class InferenceReport:
    """Compact, model-independent inference report.

    Parameters
    ----------
    model : str
        Stable model identifier.
    backend : str
        PyTensor execution backend.
    mode : str
        ``"generation"``, ``"cache_free_generation"``, or ``"next_token"``.
    prompt_tokens : int
        Number of tokens in the formatted prompt.
    output_tokens : int
        Number of visible output tokens.
    stop_reason : str
        Why execution stopped.
    timing_s : Mapping[str, Any]
        Recorded timing measurements in seconds.
    memory : Mapping[str, Any]
        Recorded memory measurements.
    validation : Mapping[str, Any]
        Differential validation summary.
    raw : Mapping[str, Any]
        Full sanitized implementation report.
    """

    model: str
    backend: str
    mode: str
    prompt_tokens: int
    output_tokens: int
    stop_reason: str
    timing_s: Mapping[str, Any] = field(default_factory=dict, repr=False)
    memory: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def total_s(self) -> float | None:
        """Return the recorded end-to-end time when it can be reconstructed."""
        if self.mode == "generation":
            keys = (
                "load_dequant_s",
                "mlx_convert_s",
                "compile_s",
                "prefill_s",
                "decode_total_s",
            )
        elif self.mode == "cache_free_generation":
            keys = (
                "tokenize_s",
                "ref_load_s",
                "ref_forward_s",
                "ref_sync_s",
                "ref_generation_s",
                "pt_compile_s",
                "pt_total_s",
                "generation_additional_wall_s",
            )
        else:
            keys = (
                "tokenize_s",
                "ref_load_s",
                "ref_forward_s",
                "ref_sync_s",
                "pt_compile_s",
                "pt_total_s",
            )

        values = [self.timing_s.get(key) for key in keys]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]

        return round(sum(numeric), 3) if numeric else None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-safe copy of the full implementation report."""
        return deepcopy(dict(self.raw))

    def __repr__(self) -> str:
        validation = dict(self.validation)

        return (
            "InferenceReport(\n"
            f"  model={self.model!r}, backend={self.backend!r}, mode={self.mode!r},\n"
            f"  prompt_tokens={self.prompt_tokens}, output_tokens={self.output_tokens}, stop_reason={self.stop_reason!r},\n"
            f"  total_s={self.total_s!r}, validation={validation!r}\n"
            ")"
        )


@dataclass(frozen=True)
class InferenceResult:
    """Text, token IDs, and evidence produced by an inference call."""

    input: str
    output: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    report: InferenceReport

    @classmethod
    def from_smollm2_report(cls, report: Mapping[str, Any]) -> InferenceResult:
        """Build a result from a sanitized SmolLM2 generation report."""
        prompt = report["prompt"]
        generation = report["generation"]
        reference = report.get("reference", {})

        validation = {
            key: reference[key]
            for key in ("argmax_match", "top10_overlap", "pearson", "mean_abs_diff")
            if key in reference
        }

        inference_report = InferenceReport(
            model=str(report["model"]["repo"]),
            backend="mlx",
            mode="generation",
            prompt_tokens=int(prompt["n_tokens"]),
            output_tokens=int(generation["n_tokens"]),
            stop_reason=str(generation["stop_reason"]),
            timing_s=deepcopy(report.get("timing", {})),
            memory=deepcopy(report.get("memory", {})),
            validation=validation,
            raw=deepcopy(dict(report)),
        )

        return cls(
            input=str(prompt["text"]),
            output=str(generation["text"]),
            input_token_ids=tuple(int(token_id) for token_id in prompt["token_ids"]),
            output_token_ids=tuple(
                int(token_id) for token_id in generation["generated_ids"]
            ),
            report=inference_report,
        )

    @classmethod
    def from_gemma3n_report(cls, report: Mapping[str, Any]) -> InferenceResult:
        """Build a result from a sanitized Gemma 3n inference report."""
        prompt = report["prompt"]
        metrics = report.get("metrics", {})
        aggregate = metrics.get("aggregate", {})
        generation = report.get("generation")

        if generation is None:
            token_id = metrics.get("final_top1_pt")
            output_ids = (int(token_id),) if token_id is not None else ()
            output = metrics.get("final_top1_pt_text") or ""
            mode = "next_token"
            stop_reason = "next_token"
        else:
            output_ids = tuple(
                int(token_id)
                for token_id in generation.get("generated_ids", ())
            )
            output = generation.get("text") or ""
            mode = "cache_free_generation"
            stop_reason = str(generation.get("stop_reason", "max_tokens"))

        thresholds = report.get("publication_thresholds", {})

        validation = {
            "final_top1_match": metrics.get("final_top1_match"),
            "all_top1_match": metrics.get("all_top1_match"),
            "pearson_mean": aggregate.get("pearson_mean"),
            "top10_overlap_mean": aggregate.get("top10_overlap_mean"),
            "thresholds_passed": thresholds.get("passed"),
            "scope": (
                generation.get("validation_scope")
                if generation is not None
                else "first_step_all_prompt_positions"
            ),
            "all_generated_tokens_match": (
                generation.get("all_reference_tokens_match")
                if generation is not None
                else None
            ),
        }

        validation = {key: value for key, value in validation.items() if value is not None}

        backend = report.get("backend", {})

        inference_report = InferenceReport(
            model=str(report["model"]["repo"]),
            backend=str(backend.get("name", "unknown")),
            mode=mode,
            prompt_tokens=int(prompt["n_tokens"]),
            output_tokens=len(output_ids),
            stop_reason=stop_reason,
            timing_s=deepcopy(report.get("timing", {})),
            memory=deepcopy(report.get("memory", {})),
            validation=validation,
            raw=deepcopy(dict(report)),
        )

        return cls(
            input=str(prompt["text"]),
            output=str(output),
            input_token_ids=tuple(int(token_id) for token_id in prompt["token_ids"]),
            output_token_ids=output_ids,
            report=inference_report,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the full sanitized implementation report."""
        return self.report.to_dict()

    def __repr__(self) -> str:
        return (
            "InferenceResult(\n"
            f"  output={self.output!r},\n"
            f"  output_token_ids={self.output_token_ids!r},\n"
            f"  report={self.report!r}\n"
            ")"
        )
