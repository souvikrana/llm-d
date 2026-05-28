# Experiment Blockers

## Blocker — GH200 / arm64 + driver 570 / CUDA 12.8 [RESOLVED]

**Date discovered:** 2026-05-26
**Date resolved:** 2026-05-28
**Hardware:** NVIDIA GH200 480GB (Grace ARM64 + Hopper GPU)

### Summary

We could not run any official inference engine container (SGLang or vLLM) on
this hardware with the original driver. The published Docker images for both
engines for arm64 require CUDA ≥ 12.9, which needs driver 575+.

### Resolution

Upgraded the NVIDIA driver from 570.211 to 580.159 (open kernel module variant
required for self-hosted GH200). All current cu129+ arm64 images now run.

### Compatibility matrix (final)

| Engine | Working tag | Notes |
|--------|-------------|-------|
| SGLang `lmsysorg/sglang` | `v0.5.10.post1-cu129` | arm64 + cu129, works with driver 580 |
| vLLM `vllm/vllm-openai` | `v0.21.0-cu129` | arm64 + cu129, works with driver 580 |

### Operational notes

- The GH200 requires the **NVIDIA open kernel modules**. The standard
  `nvidia-driver-575` package installs closed modules and silently fails with
  "self-hosted GPU requires open kernel modules" in dmesg.
  Use `nvidia-driver-580-open` (or `nvidia-open-580`) instead.
- After driver upgrade, the containerd config at
  `/etc/containerd/conf.d/99-nvidia.toml` must be regenerated:
  ```
  sudo nvidia-ctk runtime configure --runtime=containerd --config=/etc/containerd/config.toml
  sudo sed -i 's/default_runtime_name = "runc"/default_runtime_name = "nvidia"/' /etc/containerd/conf.d/99-nvidia.toml
  sudo systemctl restart containerd
  ```
- MIG is enabled on this host (7× 1g.12gb slices). The default NVIDIA k8s
  device plugin does not handle MIG out of the box — use the MIG-aware manifest
  at `configs/cluster-setup/nvidia-device-plugin-mig.yaml` (sets
  `migStrategy: mixed` and `privileged: true` for memory introspection).

### Hardware constraints that remain

- **One physical GPU** (1× GH200) — partitioned via MIG into 7× 1g.12gb slices
- **Each MIG slice is 12GB** — fits Qwen2.5-1.5B comfortably; Qwen2.5-7B needs the full GPU (MIG must be disabled for that path)
