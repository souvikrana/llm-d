#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Experiment 1 (Single-GPU Variant): Cache Hit Rate Measurement
# =============================================================================
#
# Measures ground-truth cache hit rates on a single SGLang instance across
# three traffic profiles. This validates:
#   1. How much prefix reuse actually matters (theoretical ceiling)
#   2. That our measurement infrastructure works correctly
#   3. Profile C produces ~0% hit rate (control validation)
#
# Prerequisites:
#   - kubectl connected to cluster with 1 GPU
#   - HF token secret created
#   - Python deps installed
#
# Usage:
#   ./run-single-gpu.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../configs/single-gpu"
RESULTS_DIR="${SCRIPT_DIR}/results/single-gpu-$(date +%Y%m%d-%H%M%S)"
LOAD_GEN_DIR="${SCRIPT_DIR}/load-generator"

NAMESPACE="llm-d-exp1"
MODEL="Qwen/Qwen2.5-7B-Instruct"
QPS="${QPS:-10}"
DURATION="${DURATION:-600}"  # 10 minutes per profile
MAX_TOKENS="${MAX_TOKENS:-64}"

mkdir -p "${RESULTS_DIR}"

echo "=============================================="
echo "Experiment 1 (Single-GPU): Cache Hit Baseline"
echo "=============================================="
echo "Namespace:    ${NAMESPACE}"
echo "Model:        ${MODEL}"
echo "QPS:          ${QPS}"
echo "Duration:     ${DURATION}s per profile"
echo "Results dir:  ${RESULTS_DIR}"
echo "=============================================="
echo ""

# --- Get endpoint ---
echo "Getting SGLang service IP..."
SVC_IP=$(kubectl get service sglang-exp1 -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')
POD_IP=$(kubectl get pods -l app=sglang-exp1 -n "${NAMESPACE}" -o jsonpath='{.items[0].status.podIP}')

echo "  Service IP: ${SVC_IP}"
echo "  Pod IP: ${POD_IP}"
echo ""

# Quick health check
echo "Checking SGLang health..."
kubectl exec -n "${NAMESPACE}" deploy/sglang-exp1 -- curl -s http://localhost:8000/health || {
    echo "ERROR: SGLang not healthy. Check pod logs:"
    echo "  kubectl logs -n ${NAMESPACE} deploy/sglang-exp1 --tail=50"
    exit 1
}
echo "  SGLang is healthy."
echo ""

# --- Run profiles ---
for PROFILE in A B C; do
    echo "====== Profile ${PROFILE} ======"

    # Flush cache before each profile
    echo "  Flushing RadixCache..."
    kubectl exec -n "${NAMESPACE}" deploy/sglang-exp1 -- \
        curl -s -X POST http://localhost:8000/flush_cache || true
    sleep 5

    # Start metrics scraper in background
    echo "  Starting metrics scraper..."
    python3 "${LOAD_GEN_DIR}/scrape_metrics.py" \
        --pods "${POD_IP}:8000" \
        --interval 10 \
        --duration "${DURATION}" \
        --output "${RESULTS_DIR}/profile${PROFILE}-metrics.json" &
    SCRAPER_PID=$!

    # Run load generator (from within cluster via kubectl exec, or port-forward)
    echo "  Running load test: profile=${PROFILE}, qps=${QPS}, duration=${DURATION}s"
    python3 "${LOAD_GEN_DIR}/run_benchmark.py" \
        --endpoint "http://${SVC_IP}:8000" \
        --profile "${PROFILE}" \
        --qps "${QPS}" \
        --duration "${DURATION}" \
        --max-tokens "${MAX_TOKENS}" \
        --model "${MODEL}" \
        --output "${RESULTS_DIR}/profile${PROFILE}-results.json"

    # Wait for scraper
    wait "${SCRAPER_PID}" 2>/dev/null || true

    echo "  Profile ${PROFILE} complete."
    echo ""
done

echo "=============================================="
echo "ALL PROFILES COMPLETE"
echo "Results: ${RESULTS_DIR}"
echo "=============================================="
echo ""
echo "Analyze with:"
echo "  python3 ${LOAD_GEN_DIR}/analyze_results.py --results-dir ${RESULTS_DIR}"
