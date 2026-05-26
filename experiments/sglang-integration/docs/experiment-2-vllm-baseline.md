# Experiment 2: vLLM + llm-d Routing Baseline

## Why this experiment

Two reasons:

1. **SGLang is blocked** on this hardware (see `blockers.md`). We need progress
   on something measurable.
2. **vLLM is the reference engine** for llm-d. Before we can claim "llm-d
   routing helps SGLang by X%", we need to know what llm-d routing actually
   does for vLLM on the same workload, on the same hardware. That number is
   the ceiling SGLang would need to reach to be worth shipping.

## Goal

Measure the gap between:

- **Baseline:** vLLM behind a plain Kubernetes Service (round-robin)
- **Heuristic routing:** vLLM behind llm-d EPP with the prefix-cache-scorer (no KVEvents)
- **Precise routing:** vLLM behind llm-d EPP with KVEvents enabled (the
  precise-prefix-cache-scorer)

Across the same three traffic profiles (A, B, C) we defined for Experiment 1.

## Hardware constraint

We have 1× GH200 96GB. To run multi-pod routing experiments we need to
partition the GPU. Two viable options:

| Option | GPU layout | Model | Notes |
|--------|-----------|-------|-------|
| **A — Single pod** | 1× full GPU | Qwen2.5-7B-Instruct | Validates infrastructure + measures single-pod cache behavior. Cannot compare routing strategies. |
| **B — MIG (3× 2g.24gb)** | 3 slices, 24GB each | Qwen2.5-7B-Instruct | Real multi-pod routing. Each slice fits the model. |
| **C — MIG (2× 3g.48gb)** | 2 slices, 48GB each | Qwen2.5-7B-Instruct or Qwen2.5-14B | Smaller pod count but each slice has more headroom for KV cache. |

We start with **Option A** (single pod) to:
- Confirm vLLM runs end-to-end on this hardware
- Validate the load generator and metrics scraper work
- Get a clean "infinite cache, single pod" reference for cache hit rates

Then we move to **Option B** to exercise actual routing decisions.

## What we measure

Identical to Experiment 1:

- **Cache hit rate** — from vLLM's `vllm:gpu_prefix_cache_hit_rate` metric (ground truth)
- **TTFT** — Time to First Token (streaming response)
- **Total latency**
- **Per-pod request distribution** (multi-pod runs only)

vLLM exposes these on `/metrics`:

```
vllm:gpu_prefix_cache_hit_rate
vllm:gpu_prefix_cache_queries_total
vllm:gpu_prefix_cache_hits_total
vllm:num_requests_running
vllm:num_requests_waiting
vllm:time_to_first_token_seconds (histogram)
```

## Phase 1 — Single-pod baseline

### Setup

- 1× vLLM pod, full 96GB GPU
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Image: `vllm/vllm-openai:v0.7.3` (arm64-compatible, CUDA 12.4 base — works with driver 570)
- vLLM args:
  - `--enable-prefix-caching` (enables APC — automatic prefix caching)
  - `--block-size 16`
  - `--gpu-memory-utilization 0.90`

### Profiles

Same as Experiment 1: A (high prefix sharing), B (moderate), C (none).

### Decisions

- **If Profile A hit rate < 60%** → APC isn't working as expected. Investigate config.
- **If Profile A hit rate > 80%** → APC works as expected at single-pod level. Proceed to Phase 2.
- **If Profile C hit rate > 5%** → measurement contamination. Fix before Phase 2.

## Phase 2 — Multi-pod routing comparison

### Setup

Re-enable MIG with `3× 2g.24gb` slices. Deploy:

- 3× vLLM pods, each on a 24GB MIG slice
- Same vLLM args as Phase 1 but with `--gpu-memory-utilization 0.85` (MIG slices are tighter)

### Routing configurations to compare

| Config | Description |
|--------|-------------|
| **Round-robin** | Plain ClusterIP service, no scoring |
| **llm-d heuristic** | EPP with `prefix-cache-scorer` (shadow tree) |
| **llm-d precise** | EPP with `precise-prefix-cache-scorer` + vLLM KVEvents over ZMQ |

### Decisions (same framework as Experiment 1)

- **Gap < 10%** between RR and heuristic: routing doesn't matter for this workload class on vLLM either. Re-evaluate experiment design.
- **Gap 10-30%, TTFT 20-40% better**: heuristic routing is meaningful. KVEvents adds incremental value. Same priority for SGLang KVEvents work.
- **Gap > 30%, TTFT > 50% better**: routing is critical. KVEvents native publisher is high-priority for SGLang.
- **Precise beats heuristic by < 5%**: KVEvents adds little over the shadow tree. SGLang KVEvents work is lower priority.
- **Precise beats heuristic by > 20%**: KVEvents is the bigger win. SGLang KVEvents is critical path.

## Reusable artifacts

We reuse from Experiment 1:

- `benchmarks/load-generator/traffic_profiles.py` — unchanged
- `benchmarks/load-generator/run_benchmark.py` — unchanged (OpenAI-compatible API)
- `benchmarks/load-generator/scrape_metrics.py` — extended to handle vLLM metric names
- `benchmarks/load-generator/analyze_results.py` — same decision framework

New artifacts:

- `configs/vllm-single-gpu/` — Phase 1 deployment
- `configs/vllm-mig/` — Phase 2 deployment (created when Phase 1 passes)
- `benchmarks/run-experiment-2.sh` — orchestration

## Branch policy

Same `experiment/sglang-integration` branch — even though we're running vLLM,
this work directly informs the SGLang decisions. The vLLM numbers become the
target for SGLang once it's unblocked.
