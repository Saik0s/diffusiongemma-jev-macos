# What a DiffusionGemma Splash port requires

Splash 1.0 cannot gain DiffusionGemma support through a model manifest alone.
Its public runtime has two Qwen target layouts and one autoregressive DFlash
execution contract. DiffusionGemma needs a separate block-diffusion model path.

This note describes a real port, meaning that Splash loads DiffusionGemma
weights and executes its model semantics. Running the existing MLX adapter on
the same Mac does not count as Splash support.

## The architecture boundary

Splash's reusable HTTP, admission, cancellation, memory-pressure, and prefix
cache infrastructure is useful. Its current native model path is not directly
reusable:

- Splash's model factory accepts only `Qwen3_8Weights` and
  `Qwen3_6MoeWeights`.
- Its execution limits describe an eight-row DFlash proposal/verification
  cycle. DFlash is mandatory for the supported autoregressive models.
- Its cache and scheduler advance token-by-token through target and draft
  state.
- Its public API emits selected tokens and rejects token log probabilities.

DiffusionGemma uses one 30-layer Gemma-4 MoE backbone in two phases:

1. Causal prefill writes prompt or committed-block keys and values to a cache.
2. A bidirectional decoder repeatedly evaluates a 16–256 token canvas against
   that cache. For full text generation, an entropy rule accepts and re-noises
   positions until the block converges, then the committed block is causally
   prefilled. For this project's structured reads, one or more canvas forwards
   return selected-label probabilities without autoregressive generation.

Splash 1.0 publishes no DiffusionGemma-compatible DFlash draft. A functional
port must make the draft optional and add a block-diffusion execution contract.

## Required work

### 1. Define and pack the model

Add a `DiffusionGemmaLayout`, weight variant, package schema, and strict package
validation. Build a converter from the pinned Hugging Face or OptiQ checkpoint
to that format. The converter must preserve the checkpoint's mixed precision,
tensor identity, alignment, file sizes, and hashes.

The layout must cover the 262,144 by 2,816 tied embedding, 30 layers, 25
sliding and 5 full-attention layers, the different 256/512 head dimensions,
128 experts with top 8 selected per token, the always-on dense feed-forward
path, seven layer norms, layer scalars, and the self-conditioning MLP. Full
attention aliases values from the raw key projection rather than carrying a
separate value projection.

Splash publishes package validators, but its repository does not publish a
general Hugging Face-to-Splash converter. This converter is new work.

### 2. Implement the native forward paths

Add model-specific Metal execution for:

- causal prefill with the model's sliding and full-attention masks;
- bidirectional canvas attention over cached prompt rows plus all active canvas
  rows;
- the dense feed-forward and routed top-8 MoE paths;
- self-conditioning from the previous canvas distribution;
- tied output projection and logit softcap;
- mixed 4-bit, 8-bit, and BF16 weights used by the selected checkpoint.

Existing Splash linear, normalization, MoE, RoPE, and attention operators may
be reusable where their shapes and numerical contracts match. The surrounding
graph and several shapes differ, so reuse has to be proven operator by
operator. Shape-specialized kernels then need M3/M4/M5 tuning presets.

### 3. Add block-diffusion state and scheduling

Create a model/runtime contract for a persistent causal prefix plus ephemeral
canvas state. It needs separate work types for prefill, canvas scoring, denoise,
and committed-block prefill. Admission must budget the logits, self-conditioning
buffers, MoE scratch, and up to the active canvas width.

The current DFlash proposal, verification, acceptance, and recurrent draft
state must be bypassed for this model. Prefix reuse can retain Splash's cache
ownership model, but snapshots must represent DiffusionGemma's per-layer KV
state and block boundaries. Cancellation and memory-pressure recovery must be
safe between GPU submissions without publishing a partially denoised block.

For full generation, implement the pinned sampler: seeded discrete noise,
temperature schedule, entropy-bound acceptance, re-noising, convergence rules,
and 256-token block commit. These semantics are part of the model, not an API
option.

### 4. Expose structured scoring

This project's API needs probabilities over a small set of label tokens.
Splash's current token-stream callback cannot represent that result. Add a
native score operation that gathers label logits at requested canvas positions,
applies a float32 softmax, and averages independent noise samples. Then add a
typed server route or a local adapter for `noul`, `choice`, and `score` results.

JSON-schema-constrained text generation is not equivalent: it selects a text
answer but does not expose the probability distribution used by this API.

### 5. Prove correctness before optimizing

The minimum gates are:

- loader rejection for missing, altered, misaligned, or wrong-shape artifacts;
- per-operator CPU or MLX oracle tests for norms, RoPE, attention, router,
  experts, self-conditioning, and projection;
- per-layer and full-forward comparisons against the pinned OptiQ runtime;
- deterministic repeatability for fixed seed on one machine;
- structured-read agreement on every frozen development case, including the
  selected-label probabilities within a declared tolerance;
- generation agreement on block boundaries, stop behavior, and denoise-step
  accounting;
- memory-pressure, cancellation, prefix-reuse, and concurrent-request tests;
- M4 measurements for wall time, peak footprint, swap growth, and the existing
  accuracy suite before any speed claim.

Cross-chip token identity cannot be assumed. The existing M2 and M4 OptiQ runs
already produce deterministic results on each machine but different rankings
between machines.

## Practical implementation routes

The shortest prototype is to keep Splash's server surface and call an existing
native DiffusionGemma engine through a process or foreign-function boundary.
The public `mmastrac/diffgemma` project already implements Metal prefill,
denoising, a quantized packer, and structured scoring. That can prove product
behavior quickly, but it does not exercise Splash's native scheduler, cache, or
memory plan. Its repository metadata declares MIT, but the inspected revision
does not contain a license text, so permission should be clarified before code
is copied.

A true Splash backend ports or reimplements those model-specific pieces behind
new Splash runtime interfaces. This retains Splash's scheduling and cache
infrastructure and is the path for an upstream-quality contribution. It is a
substantial native-runtime project rather than an adapter change.

As an engineering estimate, an API-compatible prototype around the existing
native engine is about **one to two focused weeks** once model weights are
available. A production Splash backend, including package tooling, numerical
parity, scheduler integration, memory recovery, kernel tuning, and evaluation,
is closer to **six to ten focused engineer-weeks** for someone already fluent
in Metal inference. These are planning ranges, not measured delivery dates;
upstream help and reusable kernels are the largest variables.

Inco explicitly describes Splash as model-specific and invites requests for
new models. Coordinating the architecture boundary and package format with
Inco before implementing it avoids maintaining an incompatible fork. Their
involvement is also the only current route to a first-party tuned package;
Splash's repository does not contain a DiffusionGemma package or draft.

## Evidence inspected

- [Splash 1.0 model layouts](https://github.com/incoai/splash/blob/c675ed23e6942b5353961246e68b08cd63fb4ee9/runtime/model/ModelDescriptor.hpp)
- [Splash 1.0 package dispatch](https://github.com/incoai/splash/blob/c675ed23e6942b5353961246e68b08cd63fb4ee9/runtime/model/ModelDescriptor.mm)
- [Splash development and package contract](https://github.com/incoai/splash/blob/c675ed23e6942b5353961246e68b08cd63fb4ee9/DEVELOPMENT.md)
- [Splash's model-specialization design](https://inco.ai/blog/splash/)
- [Independent native DiffusionGemma architecture](https://github.com/mmastrac/diffgemma/blob/6f6c825dc9a7bad15451f8dc35174209c14480a3/ARCHITECTURE.md)
