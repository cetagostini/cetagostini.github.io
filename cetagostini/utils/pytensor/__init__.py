"""Reusable PyTensor building blocks and local-LLM inference APIs.

The package keeps the notebook-facing surface deliberately small:

``SmolLM2`` and ``Gemma3n``
    Pinned model descriptors.
``inference``
    Uniform Python entry point.
``InferenceResult`` and ``InferenceReport``
    Structured output and validation evidence.

Lower-level graph, model, and weight utilities live in the ``ops``, ``models``,
``weights``, and ``backends`` submodules.
"""

from .api import Gemma3n, InferenceModel, SmolLM2, inference
from .reports import InferenceReport, InferenceResult

__all__ = [
    "Gemma3n",
    "InferenceModel",
    "InferenceReport",
    "InferenceResult",
    "SmolLM2",
    "inference",
]
