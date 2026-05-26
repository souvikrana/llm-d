# Experiment Blockers

## Blocker — GH200 / arm64 + driver 570 / CUDA 12.8

**Date discovered:** 2026-05-26
**Hardware:** NVIDIA GH200 480GB (Grace ARM64 + Hopper GPU)
**Driver version:** 570.211.01 (supports CUDA ≤ 12.8)

### Summary

We cannot run any official inference engine container (SGLang or vLLM) on this
hardware without upgrading the driver. The published Docker images for both
engines assume either:

- **arm64 build with CUDA ≥ 12.9** — requires driver 575+, which we don't have
- **older CUDA build (≤ 12.8) but amd64-only** — won't run on Grace ARM64 CPU

### Compatibility matrix

| Engine | arm64 + CUDA ≤ 12.8? | arm64 + CUDA 12.9+? | amd64 + CUDA ≤ 12.8? |
|--------|----------------------|---------------------|----------------------|
| SGLang `lmsysorg/sglang` | **No** (none published) | yes (`v0.5.10.post1-cu129` etc.) | yes (older versions) |
| vLLM `vllm/vllm-openai` | **No** (none published) | yes (`v0.20.2-aarch64`, `v0.21.0-cu129`) | yes (`v0.7.x`, `v0.20.x`) |

The intersection of `arm64-built` and `CUDA ≤ 12.8` is **empty across both
projects**.

### Errors observed

- **SGLang `v0.5.10.post1` (cu129 amd64):** `Error 802: system not yet initialized` — driver 570 doesn't support CUDA 12.9 runtime
- **SGLang `v0.4.7-cu124`:** `exec /usr/bin/python3: exec format error` — image is amd64-only, host CPU is arm64
- **vLLM `v0.20.2-aarch64`:** `No CUDA runtime is found` + `Failed to infer device type` — image's CUDA runtime is newer than driver supports

### Resolution paths

1. **Upgrade driver to ≥ 575 (recommended).** Unblocks all current arm64 builds (`v0.5.10.post1-cu129`, `vllm:v0.21.0-cu129` etc.). Requires `sudo apt install nvidia-driver-575` + reboot.
2. **Build vLLM or SGLang from source for arm64 + CUDA 12.8.** Feasible but ~1 day of build time on the host.
3. **Run on different hardware.** Any x86_64 host with H100/A100 and a recent driver works with all published tags.

### Decision pending

Need explicit user decision on whether to:
- (a) upgrade the driver and continue
- (b) request a different host
- (c) accept the build-from-source delay

Until then, all benchmark experiments on this hardware are blocked at the
container-startup stage.
