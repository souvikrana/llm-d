# Experiment Blockers

## Blocker 1 — GH200 / arm64 + driver 570 / CUDA 12.8 [RESOLVED]

**Resolution:** Upgraded the NVIDIA driver to 580.159.04 (open kernel module
variant required for self-hosted GH200). All cu129+ arm64 images now run on
this driver.

## Blocker 2 — Official vLLM arm64 images broken on GH200 [WORKAROUND]

**Date discovered:** 2026-05-28
**Symptom:** `torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are
busy or unavailable` on `cudaMemGetInfo` during EngineCore init, even with
driver 580 + working `nvidia-smi` inside the same container image.

**Root cause:** The upstream `vllm/vllm-openai` aarch64 Docker images are not
built correctly for Grace Hopper. The CUDA libraries shipped in the image
mismatch the host driver in subtle ways (LD_LIBRARY_PATH conflicts, missing
SBSA-specific kernels). Confirmed via [vllm issue #10459](https://github.com/vllm-project/vllm/issues/10459).

**Workaround:** Use a community-maintained GH200-specific build:

| Image | Version | Notes |
|-------|---------|-------|
| `rajesh550/gh200-vllm:0.11.0` | vLLM 0.11.0 (Aug 2025) | **Recommended.** Latest, arm64 + GH200-tested. Earlier versions also include LMCache 0.3.0. |
| `substratusai/vllm-gh200:v0.8.2` | vLLM 0.8.2 | Older but well-tested. |
| `LambdaLabsML/vllm-builder` | various | Official LambdaLabs builds. |

All require `VLLM_WORKER_MULTIPROC_METHOD=spawn` env var.

**Long-term fix:** Wait for upstream vLLM to fix aarch64 Docker support.

## Operational notes

- The GH200 requires the **NVIDIA open kernel modules** (use `nvidia-driver-580-open`, not `nvidia-driver-580`).
- After driver upgrade, regenerate the containerd CDI/runtime config:
  ```
  sudo nvidia-ctk runtime configure --runtime=containerd --config=/etc/containerd/config.toml
  sudo sed -i 's/default_runtime_name = "runc"/default_runtime_name = "nvidia"/' /etc/containerd/conf.d/99-nvidia.toml
  sudo systemctl restart containerd
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  ```
- The NVIDIA k8s device plugin needs explicit `migStrategy: none` (or `mixed` if MIG is enabled). Default `auto` reports 0 allocatable GPUs.
- MIG `1g.12gb` slices break vLLM's `cudaMemGetInfo` even on community-built GH200 images. For multi-pod experiments use larger MIG profiles (`3g.48gb`) or run multiple pods sharing the full GPU via process-level partitioning (out of scope for this experiment phase).
