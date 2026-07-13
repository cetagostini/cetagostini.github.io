<a href="#quarto-document-content" class="skip-link">Skip to content</a>

<div class="quarto-title">

<div class="quarto-title-block">

<div>

Code

-   <a href="javascript:void(0)" id="quarto-show-all-code" class="dropdown-item">Show All Code</a>

-   <a href="javascript:void(0)" id="quarto-hide-all-code" class="dropdown-item">Hide All Code</a>

-   

    ------------------------------------------------------------------------

-   <a href="javascript:void(0)" id="quarto-view-source" class="dropdown-item">View Source</a>

</div>

</div>

<div class="quarto-categories">

<div class="quarto-category">

python

</div>

<div class="quarto-category">

pytensor

</div>

<div class="quarto-category">

mlx

</div>

<div class="quarto-category">

numba

</div>

<div class="quarto-category">

llm

</div>

<div class="quarto-category">

gguf

</div>

<div class="quarto-category">

gemma

</div>

</div>

</div>

<div>

<div class="description">

What PyTensor is missing for LLM inference—and how straightforward it is to build. A Python-native exploration of symbolic graphs, GGUF weights, C, Numba, MLX, and the path to a composable LLM runtime.

</div>

</div>

<div class="quarto-title-meta">

<div>

<div class="quarto-title-meta-heading">

Author

</div>

<div class="quarto-title-meta-contents">

Carlos Trujillo

</div>

</div>

<div>

<div class="quarto-title-meta-heading">

Published

</div>

<div class="quarto-title-meta-contents">

July 12, 2026

</div>

</div>

</div>

<div id="introduction" class="section level1">

# Introduction

Most practitioners meet [PyTensor](https://pytensor.readthedocs.io/) through PyMC. We write a probabilistic model, PyMC builds a symbolic graph, and PyTensor compiles the mathematics into something a machine can execute.

That description is accurate. It is also incomplete.

PyTensor is a general symbolic tensor compiler. It does not know what a prior is, and it does not require a likelihood. It is the graph compiler at the heart of probabilistic programming, but its architecture was never limited to that domain. This article explores what happens when we push it toward something the authors may not have originally intended—local LLM inference—and what that tells us about where PyTensor, and the [pytensor-ml](https://github.com/pymc-labs/pytensor-ml) project, could go next.

Here is what PyTensor already has: multi-backend execution (JAX, MLX, Numba, C), symbolic graph optimization that simplifies computation before it hits hardware, GGUF dequantization via the `gguf` library, working inference for small models, and the [Alchemize](https://github.com/pymc-labs/alchemize) pipeline that auto-generates PyTensor modules from GGUF metadata.

Here is what is missing: production KV caching, continuous batching, mmap zero-copy loading, and the performance tuning that makes `llama.cpp` serve models at scale. Those gaps define the roadmap. Here we build the complete vertical slice underneath them: model loading, symbolic execution, generation, and independent numerical validation.

The story this article tells is not about replacing `llama.cpp`. It is about discovering how straightforward it is to assemble an LLM inference stack in PyTensor—weights, tokenization, a symbolic transformer, a generation loop, and numerical validation—once the graph compiler is freed from its probabilistic assumptions. And it is about what becomes possible when those pieces come together.

It will be super cool be able to run the following in full pytensor no?

<div id="8dee2c55" class="cell" execution_count="1">

<div id="cb1" class="sourceCode cell-code">

``` sourceCode
from pathlib import Path

from cetagostini.utils.pytensor import Gemma3n, inference

model = Gemma3n.from_snapshot(
    Path("/path/to/gemma-3n-E4B-it-lm-4bit/snapshot"),
    backend="c",
)

result = inference(
    model,
    input="What's causal inference?",
    max_tokens=32,
)

result.output
result.report
```

</div>

</div>

That small interface is our destination.

Alchemize gives us the map: the tensor-name inventory, the block structure, and the exact contracts that are missing. Then we build—one piece at a time.

<div class="callout callout-style-simple callout-note callout-titled">

<div class="callout-header d-flex align-content-center" bs-toggle="collapse" bs-target=".callout-1-contents" aria-controls="callout-1" aria-expanded="false" aria-label="Toggle callout">

<div class="callout-icon-container">

</div>

<div class="callout-title-container flex-fill">

What is Alchemize?

</div>

<div class="callout-btn-toggle d-inline-block border-0 py-1 ps-1 pe-0 float-end">

</div>

</div>

<div id="callout-1" class="callout-1-contents callout-collapse collapse">

<div class="callout-body-container callout-body">

[Alchemize](https://github.com/pymc-labs/alchemize) is an LLM-based, self-correcting transpiler from PyMC Labs. It acts as an AI agent that compiles computational models between frameworks: PyMC, Stan, JAX, PyTorch, and Rust—with numerical validation at every step. The agent reasons about the full computational graph and applies optimizations a domain expert would: loop fusion, memory pre-allocation, cache-friendly access patterns.

In this article we use Alchemize’s ability to read a GGUF model file and auto-generate a PyTensor module matching its architecture. It extracts tensor names, layer structure, and metadata—producing a skeleton that would take hours to write by hand. The generated code gives us the architecture map; the missing runtime contracts (dequantization, tensor orientation, KV caching) are what we build explicitly in the sections that follow.

</div>

</div>

</div>

Weight loading. Tokenization. The symbolic transformer. Generation. We build each piece against Gemma 3n E4B through C/CVM, Numba, and MLX, measure what the current speed tells us, and project forward.

<div class="callout callout-style-default callout-note callout-titled">

<div class="callout-header d-flex align-content-center" bs-toggle="collapse" bs-target=".callout-2-contents" aria-controls="callout-2" aria-expanded="false" aria-label="Toggle callout">

<div class="callout-icon-container">

</div>

<div class="callout-title-container flex-fill">

The argument

</div>

<div class="callout-btn-toggle d-inline-block border-0 py-1 ps-1 pe-0 float-end">

</div>

</div>

<div id="callout-2" class="callout-2-contents callout-collapse collapse">

<div class="callout-body-container callout-body">

PyTensor already has the graph, rewrite, and linker abstractions to become the computational core of a Python-native LLM runtime. What is missing is not the foundation—it is the production engineering on top of it. And building that engineering in PyTensor is surprisingly straightforward.

**What “alternative to llama.cpp” means here?** The competitive field is not “PyTensor replaces llama.cpp.” It is “PyTensor becomes the Python-native alternative that composes with the scientific computing stack.” Today `llama.cpp` is the C++ Swiss army knife for running any GGUF model fast. PyTensor could become the Python equivalent where you do not just run one model but chain LLM inference with Bayesian analysis, optimization, or custom computation graphs—all in the same framework. That composability is the real proposition.

</div>

</div>

</div>

</div>

<div id="the-stack-we-are-building" class="section level1">

# The stack we are building

PyTensor owns one layer of the system. The rest is explicit Python:

| Layer              | Responsibility                                                    |
|--------------------|-------------------------------------------------------------------|
| Weight adapters    | validate, map, dequantize, and orient GGUF or safetensors weights |
| Tokenizer adapters | apply the model’s chat template and produce exact token IDs       |
| Symbolic model     | express normalization, RoPE, attention, residual paths, and MLPs  |
| PyTensor compiler  | rewrite the graph and link it to MLX, C/CVM, or Numba             |
| Generation runtime | run prefill, update KV state, choose tokens, and stop             |
| Report layer       | return text, token IDs, timings, memory, and differential checks  |

</div>

<div id="what-alchemize-reveals" class="section level1">

# What Alchemize reveals

With the destination visible, we can rewind and follow the path that produced it—starting with what is missing.

We begin with `SmolLM2-135M-Instruct-Q4_K_M.gguf`, a roughly 105 MB GGUF file. Our first attempt is to ask [Alchemize](https://github.com/pymc-labs/alchemize) for a PyTensor implementation:

<div id="a0bff661" class="cell" execution_count="2">

Show Alchemize call

<div id="cb2" class="sourceCode cell-code">

``` sourceCode
from alchemize import compile_model

result = compile_model(
    model_path=Path("/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf"),
    target="pytensor",
)
```

</div>

</div>

Alchemize reads the GGUF metadata and generates a module with the right architecture skeleton. That is valuable: it gives us the tensor-name map, the block inventory, and the layer structure without writing any of it by hand.

But the generated implementation cannot run. Its central loading assumption is wrong:

<div id="d3da684e" class="cell" execution_count="3">

Show generated materialize\_tensor

<div id="cb3" class="sourceCode cell-code">

``` sourceCode
def materialize_tensor(tensor):
    data = getattr(tensor, "data", None)
    array = np.asarray(data)

    if np.issubdtype(array.dtype, np.floating):
        return np.asarray(array, dtype=np.float32)

    raise NotImplementedError("a dequantizer is required")
```

</div>

</div>

Most tensors in this GGUF are not floating arrays. They are packed quantized values. The generated implementation recognizes the problem and stops, but it never calls `gguf.dequantize`.

The static audit therefore gives us:

| Check                  | Result                   | Why                                                                       |
|------------------------|--------------------------|---------------------------------------------------------------------------|
| Provenance and syntax  | pass                     | the artifact is pinned and valid Python                                   |
| GGUF dequantization    | `STATIC_FAIL`            | packed weights are never dequantized                                      |
| Attention head reshape | `STATIC_FAIL` audit flag | symbolic reshapes do not preserve the repaired runtime’s static contracts |
| Runtime                | `BLOCKED`                | generated code is never executed                                          |
| Semantics              | `UNVERIFIED`             | valid syntax does not establish correct logits                            |

This is not a failure. It is a map.

Alchemize accelerated architecture discovery and told us exactly what is missing. The generated code gives us the skeleton; the static audit tells us which contracts need hand-built replacements. We keep the tensor-name map and block inventory, then build the pieces that fill the gaps: materialization, orientation, graph construction, caching, execution, and validation.

</div>

<div id="building-gemma-3n-inference-in-pytensor" class="section level1">

# Building Gemma 3n inference in PyTensor

A correct answer depends on a chain of contracts: quantized weight dequantization, tensor orientation, tokenization, rotary embeddings, grouped-query attention, KV caching, and backend compilation. Break one contract and the model may still produce plausible text. So we build each contract explicitly and compose them into a working pipeline.

Our target is Gemma 3n E4B—a much larger and stranger model than a typical first experiment. It adds four AltUp residual streams, a learned low-rank LAuReL path, per-layer token embeddings, sparse and dense activation variants, sliding and full attention, and a 262,400-token vocabulary. Can we express all of that in PyTensor’s symbolic language without changing the framework?

| Component           | E4B configuration |
|---------------------|------------------:|
| Decoder layers      |                35 |
| Hidden size         |             2,048 |
| Query / KV heads    |             8 / 2 |
| Head dimension      |               256 |
| MLP width           |            16,384 |
| Vocabulary          |           262,400 |
| AltUp streams       |                 4 |
| LAuReL rank         |                64 |
| Sliding window      |               512 |
| Final logit softcap |                30 |

This case deliberately validates **multi-token, full-prefix generation**. It does not retain a KV cache between decoding steps. Instead, every step rebuilds the complete prefix, creates the model’s shared-KV topology for that one forward, and discards it afterward.

<div id="stream-weights-instead-of-expanding-the-model" class="section level2">

## Stream weights instead of expanding the model

The 3.86 GB checkpoint stores each affine-4 linear module as packed `uint32` weights plus BF16 scales and biases. Eight four-bit values occupy one word:

<span class="math display"> W\_{r,c} = q\_{r,c}s\_{r,g(c)} + b\_{r,g(c)}. </span>

Here, <span class="math inline">q\_{r,c}</span> is the stored nibble at output row <span class="math inline">r</span> and input column <span class="math inline">c</span>; <span class="math inline">s\_{r,g(c)}</span> and <span class="math inline">b\_{r,g(c)}</span> are the scale and bias for its group.

Fully expanding all logical parameters would need about **25.6 GiB** for FP32 weights alone. We instead:

-   load only requested embedding rows,
-   dequantize one decoder layer at a time,
-   release it before loading the next layer, and
-   project vocabulary logits in chunks of 4,096 rows.

<div id="f25f1c3b" class="cell" execution_count="4">

Show weight streaming

<div id="cb4" class="sourceCode cell-code">

``` sourceCode
from cetagostini.utils.pytensor.weights import Gemma3nWeightLoader

with Gemma3nWeightLoader.from_snapshot(snapshot_path) as loader:
    token_rows = loader.load_input_embedding_rows(token_ids)
    layer_0 = loader.load_layer(0)
```

</div>

</div>

Streaming changes the problem from “hold the expanded model” to “hold the current expanded layer.”

</div>

<div id="write-the-equations-once" class="section level2">

## Write the equations once

The shared normalization is ordinary PyTensor:

<div id="13958912" class="cell" execution_count="5">

Show rmsnorm\_symbolic

<div id="cb5" class="sourceCode cell-code">

``` sourceCode
from cetagostini.utils.pytensor.ops import rmsnorm_symbolic

normalized = rmsnorm_symbolic(hidden, gamma, eps=1e-6)
```

</div>

</div>

The important detail is not the formula. It is that `rmsnorm_symbolic` knows nothing about C, Numba, or MLX.

The same is true for grouped-query attention, RoPE, masks, AltUp, and LAuReL. Gemma’s sparse and dense GELU paths produce two specialized `FunctionGraph`s; full versus sliding attention arrives as mask and RoPE data. The operation order follows the pinned MLX-LM implementation—including LAuReL’s apparent repeated residual and the sparse GELU sparsity pattern read from the checkpoint.

</div>

<div id="choose-the-backend-at-compilation" class="section level2">

## Choose the backend at compilation

Backend selection is now a small, reusable utility:

<div id="eb1bf635" class="cell" execution_count="6">

Show backend selection

<div id="cb6" class="sourceCode cell-code">

``` sourceCode
from cetagostini.utils.pytensor.backends import get_mode

c_mode = get_mode("c")
numba_mode = get_mode("numba")
mlx_mode = get_mode("mlx")

c_layer = pytensor.function(layer_inputs, layer_output, mode=c_mode)
numba_layer = pytensor.function(layer_inputs, layer_output, mode=numba_mode)
mlx_layer = pytensor.function(layer_inputs, layer_output, mode=mlx_mode)
```

</div>

</div>

The model definition did not change. Only the linker and rewrite policy changed.

<div class="callout callout-style-default callout-tip callout-titled">

<div class="callout-header d-flex align-content-center">

<div class="callout-icon-container">

</div>

<div class="callout-title-container flex-fill">

One model, multiple compiler experiments

</div>

</div>

<div class="callout-body-container callout-body">

If we change a symbolic equation, every backend inherits it. If we change only a rewrite or linker, the model remains fixed. That separation is PyTensor’s main contribution to this experiment.

</div>

</div>

</div>

<div id="run-gemma-inference-from-python" class="section level2">

## Run Gemma inference from Python

The same public entry point now targets a different artifact and backend:

<div id="d73f8a52" class="cell" execution_count="7">

<div id="cb7" class="sourceCode cell-code">

``` sourceCode
from pathlib import Path

from cetagostini.utils.pytensor import Gemma3n, inference

gemma = Gemma3n.from_snapshot(
    Path("/path/to/gemma-3n-E4B-it-lm-4bit/snapshot"),
    backend="c",
)

result = inference(
    gemma,
    input="What's causal inference?",
    max_tokens=32,
)

result.output, result.output_token_ids
```

</div>

</div>

``` text
('## Causal Inference: Understanding "Why" vs. "Correlation"\n\n'
 'Causal inference is a branch of statistics and statistics is a field '
 'of statistics that aims',
 (1408, 565, 90718, 157036, 236787, 40411, 623, 11355, 236775,
  7728, 236761, 623, 131426, 236775, 108, 236780, 90718, 34711,
  563, 496, 9911, 529, 14906, 532, 14906, 563, 496, 2135,
  529, 14906, 600, 17269))
```

That is an actual continuation, not one next-token prediction. It is also not polished prose: greedy decoding reaches the 32-token cap mid-sentence and becomes repetitive after the differential path separates. The point is to make generation inspectable, not to present a language-quality benchmark.

<div id="da3cda88" class="cell" execution_count="8">

Show validation report

<div id="cb9" class="sourceCode cell-code">

``` sourceCode
result.report.validation
```

</div>

</div>

``` text
{
    'final_top1_match': True,
    'all_top1_match': False,
    'pearson_mean': 0.998932415848074,
    'top10_overlap_mean': 9.533333333333333,
    'thresholds_passed': True,
    'scope': 'all_generated_prefixes_fresh_cache',
    'all_generated_tokens_match': False,
}
```

The initial 15-token prompt passes the publication thresholds and selects the same next token in PyTensor and MLX-LM. Generation then validates each growing PyTensor prefix against an MLX-LM forward over that exact prefix. Each oracle call creates a fresh shared-KV object and discards it; no decode state survives into the next step.

The first 20 generated decisions agree. At token 21, PyTensor chooses `9911` while the fresh-prefix MLX oracle chooses `2135`. The report records that divergence and keeps evaluating MLX on the subsequent **PyTensor prefix**, so later comparisons remain well-defined. It does not compare two independently drifting streams, and it no longer throws away an expensive completed generation because cached `stream_generate` follows a different path.

| Multi-token result              |        Value |
|---------------------------------|-------------:|
| Visible tokens                  |           32 |
| Matching fresh-prefix decisions |      31 / 32 |
| First divergence                |     token 21 |
| Prefix graph compilations       |           32 |
| Additional generation wall time |  1,612.038 s |
| Peak process RSS                | 9,668.61 MiB |
| Stop reason                     | `max_tokens` |

The cost is the lesson. After the initial validated prompt pass, the remaining full-prefix generation loop took **1,612.038 seconds**, or about **26.9 minutes**. This is an educational execution strategy that exposes every graph and comparison; it is not an efficient decoder.

<div id="ccvm-vs-numba-vs-mlx-performance-comparison" class="section level3">

### C/CVM vs Numba vs MLX: performance comparison

All three backends compile the same symbolic equations and select the same top-1 token as the independent MLX-LM oracle at all 20 prompt positions. Their mean all-logit Pearson correlation with the oracle is 0.9986, and their mean top-10 overlap is 9.7 out of 10.

| Metric                          |         C/CVM |     Numba |          MLX |
|---------------------------------|--------------:|----------:|-------------:|
| Graph compilation               |       7.738 s |   0.775 s |  **0.755 s** |
| One full forward (20 positions) |      51.441 s |  53.800 s | **50.211 s** |
| Mean per-layer time             |       1.272 s |   1.266 s |  **1.246 s** |
| Logit projection                |   **3.024 s** |   4.700 s |      3.086 s |
| Whole-process peak RSS          | **4,523 MiB** | 4,663 MiB |    5,111 MiB |

These are single-run measurements of an intentionally streamed educational pipeline, not a throughput benchmark. MLX compiles about 10 times faster than C and records the shortest forward, but only by 2.4%. C still wins the 262,400-vocabulary projection. Weight dequantization, Python orchestration, and per-layer streaming dominate enough that the three full forwards remain in the same range.

<div class="callout callout-style-simple callout-note callout-titled">

<div class="callout-header d-flex align-content-center" bs-toggle="collapse" bs-target=".callout-4-contents" aria-controls="callout-4" aria-expanded="false" aria-label="Toggle callout">

<div class="callout-icon-container">

</div>

<div class="callout-title-container flex-fill">

How the failed MLX probe became a working backend

</div>

<div class="callout-btn-toggle d-inline-block border-0 py-1 ps-1 pe-0 float-end">

</div>

</div>

<div id="callout-4" class="callout-4-contents callout-collapse collapse">

<div class="callout-body-container callout-body">

The first MLX probe failed because the generated draft relied on shapes and operations that the pinned linker could not lower safely. We did not patch the installed PyTensor package or mutate a process-global dispatch registry. Instead, we expressed projections as rank-2 matrix multiplies with explicit reshapes, kept attention in ordinary tensor primitives, and replaced Gemma’s AltUp clip sites with a repository-local symbolic helper built from comparisons and `where`.

The resulting graph compiles through PyTensor 3.1.2’s built-in `pytensor.compile.mode.MLX`. MLX evaluates lazily, so the runner materializes 103 ordered boundaries: two initial projections, 35 decoder layers, one final unembed, and 65 vocabulary chunks. Intermediate tensors remain device-resident; only completed logit chunks cross back to owning NumPy arrays. The measured MLX allocator peak was 455 MiB beyond its near-zero baseline, while whole-process RSS reached 5,111 MiB.

</div>

</div>

</div>

The complete reports are available for the independent [MLX-LM oracle](results/gemma3n_mlx_lm_oracle.json), [C/CVM](results/gemma3n_pytensor_c.json), [Numba](results/gemma3n_pytensor_numba.json), and [MLX](results/gemma3n_pytensor_mlx.json). A separate validator reloaded the temporary raw logits, recomputed every metric instead of trusting the reports, and passed [896 of 896 gates](results/gemma3n_report_validation.json). The large raw arrays are not committed; the reports preserve their shapes, dtypes, byte counts, and cryptographic hashes. The earlier 32-token generation run remains in [`gemma3n_pytensor_generation.json`](results/gemma3n_pytensor_generation.json).

</div>

</div>

</div>

<div id="what-the-current-speed-tells-us" class="section level1">

# What the current speed tells us

The numbers above are not benchmarks. They are measurements of an educational pipeline, and they should be read that way. But they are honest, and they tell us something precise about where the engineering effort needs to go.

Gemma 3n through full-prefix regeneration runs at roughly **0.02 tokens per second**. That is the cost of recompiling and re-evaluating the entire prefix for every generated token. It is deliberately expensive—it validates correctness—but it is not how a production system would work.

The engineering roadmap from here is clear:

| Bottleneck        | Current state                              | What unlocks it                                                |
|-------------------|--------------------------------------------|----------------------------------------------------------------|
| KV cache          | fixed-capacity, O(C) write per layer       | paged or ring-buffer cache, continuous batching                |
| Weight loading    | per-layer streaming (dequantize on demand) | mmap zero-copy, quantized kernels                              |
| Backend execution | C/CVM, Numba, and MLX on Apple Silicon     | backend-native quantized kernels and less host-device movement |
| Graph compilation | recompilation per prefix length            | cached compiled functions per shape, or JIT                    |

`llama.cpp` has spent years on every row of that table. PyTensor has the graph compiler and the multi-backend architecture; it does not yet have the serving infrastructure. The question is not whether PyTensor can match `llama.cpp`’s throughput today—it cannot—but whether the pieces are in place to build that infrastructure in Python. The answer, after this experiment, is yes.

<div class="callout callout-style-default callout-tip callout-titled">

<div class="callout-header d-flex align-content-center">

<div class="callout-icon-container">

</div>

<div class="callout-title-container flex-fill">

The composability advantage

</div>

</div>

<div class="callout-body-container callout-body">

`llama.cpp` is a self-contained inference engine. PyTensor is a graph compiler that can compose with JAX transformations, NumPy operations, SciPy optimizers, and the rest of the scientific Python stack. The moment LLM inference lives inside that ecosystem, you can chain it with Bayesian analysis, gradient-based optimization, or custom symbolic computation—workflows that a standalone C++ engine was never designed to support.

</div>

</div>

</div>

<div id="where-pytensor-and-llama.cpp-differ-and-why-that-matters" class="section level1">

# Where PyTensor and llama.cpp differ — and why that matters

Now we can make the comparison precise:

| Capability                           | This PyTensor stack              | `llama.cpp`                                        |
|--------------------------------------|----------------------------------|----------------------------------------------------|
| Inspectable symbolic graph           | yes                              | not its primary user abstraction                   |
| User-defined graph rewrites          | yes                              | no equivalent Python rewrite database              |
| C, Numba, and MLX experiments        | demonstrated on Gemma 3n E4B     | purpose-built CPU, Metal, CUDA, and other backends |
| GGUF parsing                         | supplied by `gguf-py` adapter    | built in                                           |
| Native quantized matmul              | not implemented here             | built in                                           |
| Tokenization and chat templates      | Python adapters                  | built in                                           |
| Autoregressive loop                  | Python, model-specific           | built in                                           |
| Production KV cache and batching     | no                               | built in                                           |
| Broad architecture support           | one validated fixture (Gemma 3n) | broad, maintained model coverage                   |
| Composability with scientific Python | native                           | not designed for it                                |

`llama.cpp` wins on integrated inference engineering. That is exactly what it is built for.

But the local LLM ecosystem is not just about running one model fast. It is about what you do with the model afterward.

The recent history of Ollama—[documented thoroughly by sleepingrobots](https://sleepingrobots.com/dreams/stop-using-ollama/)—shows what happens when a wrapper obscures its dependencies and pivots to cloud. The ecosystem needs alternatives built on honest, transparent foundations. PyTensor and `llama.cpp` are those foundations.

-   **`llama.cpp`** is the C++ engine for running any GGUF model fast. It owns the full serving stack: quantized kernels, KV cache management, continuous batching, broad architecture support. If you need to serve models at production scale today, `llama.cpp` is the answer.
-   **PyTensor** is the Python-native graph compiler for studying, modifying, and composing LLM inference. You do not just run a model—you can ask what a rewrite changed, compile the same equations through another linker, chain inference with a Bayesian posterior or a custom optimization loop, and validate every contract numerically.

The honest claim is not “PyTensor replaces `llama.cpp`.”

It is this:

<span style="color:var(--green-strong)">PyTensor</span> plus explicit Python adapters is a credible *Python-native alternative* to `llama.cpp` that **composes with the scientific computing stack**—Bayesian analysis, optimization, custom computation graphs—in ways a standalone C++ engine was never designed for.

If someone builds KV caching, mmap zero-copy loading, and continuous batching on PyTensor’s JAX backend, and [Alchemize](https://github.com/pymc-labs/alchemize) matures for auto-generating new architectures from GGUF metadata, we could see PyTensor become the “language” for running multiple LLMs the way `llama.cpp` is today. Not faster at inference—or maybe at some point—but more capable as a general-purpose LLM runtime that plugs into workflows `llama.cpp` was never built for.

</div>

<div id="conclusions" class="section level1">

# Conclusions

1.  **PyTensor is a general graph compiler.** PyMC is its most visible consumer, not the boundary of what it can express.
2.  **Building LLM inference in PyTensor is straightforward.** Weight loaders, symbolic transformers, KV caches, generation loops, and numerical validation—all assembled from Python without modifying the framework.
3.  **One symbolic definition survives multiple backends.** Gemma 3n executes through C/CVM, Numba, and MLX without a second model implementation.
4.  **Independent logits beat plausible prose.** Exact tokens and all-position numerical agreement are stronger evidence than plausible-looking text.
5.  **PyTensor and `llama.cpp` are complementary.** `llama.cpp` owns production serving. PyTensor owns graph transparency and composability with scientific Python.
6.  **The remaining gaps are engineering, not architecture.** Native quantized kernels, paged KV caching, mmap zero-copy loading, and broader backend coverage are all buildable on PyTensor’s existing foundation. The pytensor-ml project has already begun this work.

Recommended readings:

1.  [PyTensor documentation](https://pytensor.readthedocs.io/)
2.  [PyTensor graph rewriting](https://pytensor.readthedocs.io/en/latest/extending/graph_rewriting.html)
3.  [PyTensor compilation modes](https://pytensor.readthedocs.io/en/latest/library/compile/mode.html)
4.  [llama.cpp](https://github.com/ggml-org/llama.cpp)
5.  [MLX-LM](https://github.com/ml-explore/mlx-lm)
6.  [Alchemize](https://github.com/pymc-labs/alchemize)
7.  [pytensor-ml](https://github.com/pymc-labs/pytensor-ml)
8.  [Friends Don’t Let Friends Use Ollama](https://sleepingrobots.com/dreams/stop-using-ollama/)

The recorded experiment used:

| Component   | Version or pin                             |
|-------------|--------------------------------------------|
| Python      | `3.13.14`                                  |
| PyTensor    | `3.1.2`                                    |
| PyTensor-ML | `f6ecf81d58da180cce50b77a43cf5d2c3d95e470` |
| MLX         | `0.32.0`                                   |
| Numba       | `0.65.1`                                   |
| MLX-LM      | `0.31.3`                                   |
| Machine     | Apple M3 Max, 128 GB unified memory        |

The exact environment is recorded in [`environment.yml`](environment.yml). Code cells are not executed during the website build because the model artifacts are intentionally excluded from Git; the committed JSON reports preserve the artifact hashes and measured outputs. The PyTensor suite passed 934 tests with 30 skips, and the gated real-model runner tests passed all 189 checks; the compact record is in [`gemma3n_test_summary.json`](results/gemma3n_test_summary.json).

------------------------------------------------------------------------

The public package, model-specific implementations, tests, generated draft, audits, and sanitized result reports are available in the [site repository](https://github.com/cetagostini/cetagostini.github.io).

</div>
