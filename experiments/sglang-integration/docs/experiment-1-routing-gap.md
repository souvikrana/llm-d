# Experiment 1: Routing Gap Baseline

## Goal

Before building anything, establish the actual performance cost of llm-d's
current heuristic routing vs what precise KV-cache-aware routing could
theoretically deliver for SGLang workloads. This is the foundational experiment
— its outcome sets the priority of everything else.

## What We Are Measuring

Three things independently:

| Metric | Why |
|--------|-----|
| **Cache hit rate** | Fraction of prompt blocks already cached on the routed pod (ground truth from SGLang's RadixCache internal stats, not the router's estimate) |
| **TTFT (Time To First Token)** | Directly reflects how much prefill work was avoided |
| **GPU utilization** | Whether routing distributes compute efficiently or creates hot spots |

## Setup

### Cluster

- Minimum **4 SGLang worker pods**, same model
- Recommended model: **Llama-3-8B** or **Qwen-2.5-7B** (small enough to iterate
  fast, large enough that KV reuse is meaningful)
- All pods on identical hardware

### Traffic Profiles

| Profile | Description | Purpose |
|---------|-------------|---------|
| **A — High prefix sharing** | 80% of requests share a fixed 1000-token system prompt, followed by varying user messages | Simulates chatbot/RAG. Cache hit rate should be high if routing works. |
| **B — Moderate prefix sharing** | Requests share prefixes from a pool of 10 different system prompts | Simulates multi-tenant API. |
| **C — No prefix sharing** | Fully random prompts | Control baseline. Routing strategy should make no difference. Confirms measurement setup is unbiased. |

### Load Generator

- Tool: `locust` or custom script sending requests at controlled QPS
- Start at **10 req/s**, scale to **50 req/s**
- Keep `max_tokens` short (**50–100**) so TTFT dominates total latency

### Routing Configurations

Run each independently, same traffic profile, same duration (**minimum 10 minutes
each** to warm up caches):

| Config | Description |
|--------|-------------|
| **A — Round-robin** | Pure round-robin via Envoy, no EPP scoring |
| **B — llm-d heuristic** | EPP with prefix-cache-scorer (shadow tree, no KVEvents) |
| **C — SGLang native router** | SGLang's own `sgl-router` in cache-aware mode |

## Instrumentation

### SGLang Pod Metrics (expose and scrape)

```
# SGLang exposes these on /metrics
sglang:cache_hit_tokens_total
sglang:cache_miss_tokens_total
sglang:running_requests
sglang:waiting_requests
```

Compute true cache hit rate as:

```
hit_rate = cache_hit_tokens / (cache_hit_tokens + cache_miss_tokens)
```

This is **ground truth** — the engine knows what was actually in its RadixCache
when the request arrived, regardless of what the router predicted.

### Load Generator Side (per-request)

- TTFT (time from request sent to first token received)
- Total latency
- Which pod was routed to (add `x-pod-id` response header from Envoy)

### Aggregate Metrics Per Configuration

| Metric | Aggregations |
|--------|-------------|
| Cache hit rate | mean, p50, p90 |
| TTFT | mean, p50, p90, p99 |
| Per-pod cache hit rate distribution | (detect hot/cold pod imbalance) |
| GPU utilization per pod | nvidia-smi or DCGM exporter |

## Execution Steps

1. **Step 1** — Deploy 4 SGLang pods, no routing intelligence (round-robin). Run
   Profile A for 15 min. Record all metrics. Clear all pod caches between runs
   (restart pods or call `/flush_cache`).

2. **Step 2** — Deploy llm-d EPP with heuristic prefix scorer only (no KVEvents,
   no LMCache). Same traffic. Same duration.

3. **Step 3** — Deploy SGLang native `sgl-router` in cache-aware mode. Same
   traffic.

4. **Step 4** — Run Profile B and C for each configuration.

> Profile C is your control — if you see differences across configs on random
> traffic, your measurement setup has a bug.

## Decision Framework

### Case 1: Gap < 10% between round-robin and heuristic routing (Profile A)

Cache hit rates are similar regardless of routing. Either the workload doesn't
have enough prefix sharing to matter, or the model's KV cache is large enough
that most pods end up with the hot prefixes anyway.

**Decision:** KVEvents investment is low priority. The heuristic is good enough
for this workload class. Redirect effort to P/D disaggregation or latency
predictor experiments instead. Revisit with longer prompts or higher QPS where
cache pressure is real.

### Case 2: Gap 10–30% on cache hit rate, TTFT difference 20–40%

Heuristic routing is meaningfully better than round-robin but leaves significant
value on the table. Precise KV routing would recover most of that gap.

**Decision:** KVEvents native publisher is high priority but not a blocker for
shipping. Ship llm-d with heuristic routing for SGLang now. Build KVEvents in
parallel as the next milestone. This gap justifies the engineering investment.

### Case 3: Gap > 30% on cache hit rate, TTFT difference > 50%

Heuristic routing is substantially better than round-robin, but the gap to what
precise routing could deliver is large. This typically happens at high QPS with a
large prompt pool — the shadow tree drifts quickly as evictions happen.

**Decision:** KVEvents native publisher is critical path. Do not ship SGLang
support without it — heuristic-only SGLang support would be a regression compared
to users' current setup with SGLang's own router. Prioritize Experiment 2
(LMCache stability) immediately to find the fastest path to a working KVEvent
pipeline.

### Case 4: SGLang native router significantly beats llm-d heuristic (Profile A)

SGLang's shadow tree router knows SGLang's RadixCache better than llm-d's
generic heuristic because it was built specifically for SGLang's eviction
behavior.

**Decision:** Do not ship llm-d heuristic routing for SGLang as a claimed
improvement — it would be worse than what users already have. Two sub-paths:

- (a) Analyze why SGLang's router beats llm-d's heuristic and fix the
  heuristic's SGLang-specific blind spots, OR
- (b) Treat this as further evidence that KVEvents is the only path to being
  better than the native router, and double down on Experiment 2.

### Case 5: Profile C shows differences across routing configs

Random prompts should produce identical cache hit rates regardless of routing,
because there's nothing to hit. If you see differences, your cache wasn't fully
cleared between runs, or your "random" prompts still share a common prefix (e.g.
a shared tokenizer BOS token or system prompt you forgot to remove).

**Decision:** Fix the experiment before drawing any conclusions. This is a
measurement artifact, not a real signal.

## Expected Outcomes (Practical Estimates)

For a realistic chatbot-style workload (Profile A, 1000-token shared system
prompt, 10+ req/s):

| Configuration | Expected Cache Hit Rate |
|---------------|------------------------|
| Round-robin | ~20–35% |
| llm-d heuristic | ~50–65% |
| SGLang native router | ~55–70% |
| Theoretical ceiling (precise KV routing) | ~80–90% |

The gap between heuristic and theoretical ceiling — roughly **20–30 percentage
points** — is what this experiment is measuring, and what the rest of the SGLang
work is trying to close.
