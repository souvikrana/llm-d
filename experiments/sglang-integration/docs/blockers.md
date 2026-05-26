# SGLang Experiment Blockers

## Blocker 1 — GH200 / arm64 + driver 570 / CUDA 12.8

**Date discovered:** 2026-05-26
**Hardware:** NVIDIA GH200 480GB (Grace ARM64 + Hopper GPU)
**Driver version:** 570.211.01 (supports CUDA ≤ 12.8)

### Summary

We cannot run any official `lmsysorg/sglang` Docker image on this hardware
without upgrading the driver.

### Compatibility matrix observed

| SGLang version | CUDA | arm64 build available? | Compatible with driver 570? |
|----------------|------|------------------------|-----------------------------|
| v0.4.4 – v0.4.7 | cu124 | **No** (amd64 only) | yes (if image existed) |
| v0.5.0 – v0.5.10 | cu125, cu126 | only `b200/gb200` (Blackwell) variants | n/a |
| v0.5.10+ | cu129 | yes (multi-arch) | **No** — requires driver 575+ |

The intersection of `arm64-built` and `CUDA ≤ 12.8` is empty across all
published SGLang tags.

### Errors observed

- With `v0.5.10.post1` (cu129 amd64): `Error 802: system not yet initialized` — driver 570 doesn't support CUDA 12.9 runtime
- With `v0.4.7-cu124`: `exec /usr/bin/python3: exec format error` — image is amd64-only, host CPU is arm64

### Resolution paths

1. **Upgrade driver to ≥575** — unblocks `lmsysorg/sglang:v0.5.10.post1-cu129` (arm64). Requires reboot.
2. **Build SGLang from source for arm64 + CUDA 12.8** — feasible but ~1 day of build time on the host.
3. **Run on different hardware** — any x86_64 host with H100/A100 + recent driver works with all published tags.

### Decision

Park SGLang work until driver upgrade is acceptable. Pivot to a vLLM
benchmark experiment that exercises the same llm-d routing infrastructure
on the same hardware, so our findings remain useful when SGLang is
unblocked.
