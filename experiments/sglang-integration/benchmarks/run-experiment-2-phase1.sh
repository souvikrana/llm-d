#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Experiment 2 Phase 1: vLLM single-pod baseline on GH200
# =============================================================================
#
# Validates:
#   - vLLM runs end-to-end on this hardware (arm64 + driver 570)
#   - APC (automatic prefix caching) achieves expected hit rates on Profile A
#   - Load generator and metrics scraper work correctly
#   - Profile C produces near-zero hit rate (control)
#
# Prerequisites:
#   - kubectl pointing at the cluster
#   - Namespace llm-d-exp1 exists, llm-d-hf-token secret exists
#   - Python deps installed: pip install -r load-generator/requirements.txt
#
# Usage:
#   ./run-experiment-2-phase1.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../configs/vllm-single-gpu"
RESULTS_DIR="${SCRIPT_DIR}/results/exp2-phase1-$(date +%Y%m%d-%H%M%S)"
LOAD_GEN_DIR="${SCRIPT_DIR}/load-generator"

NAMESPACE="llm-d-exp1"
MODEL="Qwen/Qwen2.5-7B-Instruct"
QPS="${QPS:-10}"
DURATION="${DURATION:-600}"  # 10 minutes per profile
MAX_TOKENS="${MAX_TOKENS:-64}"

mkdir -p "${RESULTS_DIR}"

echo "=============================================="
echo "Experiment 2 Phase 1: vLLM single-pod baseline"
echo "=============================================="
echo "Namespace:    ${NAMESPACE}"
echo "Model:        ${MODEL}"
echo "QPS:          ${QPS}"
echo "Duration:     ${DURATION}s per profile"
echo "Results dir:  ${RESULTS_DIR}"
echo "=============================================="
echo ""

# --- Wait for vLLM to be ready ---
echo "Waiting for vLLM pod to be ready..."
kubectl wait --for=condition=ready pod \
    -l app=vllm-exp2 \
    -n "${NAMESPACE}" \
    --timeout=900s

SVC_IP=$(kubectl get service vllm-exp2 -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')
POD_IP=$(kubectl get pods -l app=vllm-exp2 -n "${NAMESPACE}" -o jsonpath='{.items[0].status.podIP}')

echo "  Service IP: ${SVC_IP}"
echo "  Pod IP: ${POD_IP}"
echo ""

# Quick health check
echo "Checking vLLM health..."
kubectl exec -n "${NAMESPACE}" deploy/vllm-exp2 -- curl -sf http://localhost:8000/health > /dev/null && \
    echo "  vLLM is healthy." || {
    echo "ERROR: vLLM not healthy. Check pod logs:"
    echo "  kubectl logs -n ${NAMESPACE} deploy/vllm-exp2 --tail=50"
    exit 1
}
echo ""

# --- Run profiles ---
for PROFILE in A B C; do
    echo "====== Profile ${PROFILE} ======"

    # vLLM doesn't expose /flush_cache by default, so we restart the pod for a clean cache
    echo "  Restarting pod for clean cache state..."
    kubectl rollout restart deployment/vllm-exp2 -n "${NAMESPACE}"
    kubectl rollout status deployment/vllm-exp2 -n "${NAMESPACE}" --timeout=300s

    # Refresh IPs (pod has new address)
    SVC_IP=$(kubectl get service vllm-exp2 -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')
    POD_IP=$(kubectl get pods -l app=vllm-exp2 -n "${NAMESPACE}" -o jsonpath='{.items[0].status.podIP}')

    # Wait an extra 10s for vLLM to fully warm up
    sleep 10

    # Start metrics scraper in background
    echo "  Starting metrics scraper for pod ${POD_IP}..."
    python3 "${LOAD_GEN_DIR}/scrape_metrics.py" \
        --pods "${POD_IP}:8000" \
        --interval 10 \
        --duration "${DURATION}" \
        --output "${RESULTS_DIR}/profile${PROFILE}-metrics.json" &
    SCRAPER_PID=$!

    # Run load generator
    echo "  Running load test: profile=${PROFILE}, qps=${QPS}, duration=${DURATION}s"
    python3 "${LOAD_GEN_DIR}/run_benchmark.py" \
        --endpoint "http://${SVC_IP}:8000" \
        --profile "${PROFILE}" \
        --qps "${QPS}" \
        --duration "${DURATION}" \
        --max-tokens "${MAX_TOKENS}" \
        --model "${MODEL}" \
        --output "${RESULTS_DIR}/profile${PROFILE}-results.json"

    wait "${SCRAPER_PID}" 2>/dev/null || true

    echo "  Profile ${PROFILE} complete."
    echo ""
done

echo "=============================================="
echo "PHASE 1 COMPLETE"
echo "Results: ${RESULTS_DIR}"
echo "=============================================="
echo ""
echo "Quick check — print final cache hit rates:"
for PROFILE in A B C; do
    if [[ -f "${RESULTS_DIR}/profile${PROFILE}-metrics.json" ]]; then
        rate=$(python3 -c "
import json
with open('${RESULTS_DIR}/profile${PROFILE}-metrics.json') as f:
    d = json.load(f)
print(f\"{d['aggregate']['aggregate_cache_hit_rate']:.1%}\")
")
        echo "  Profile ${PROFILE} cache hit rate: ${rate}"
    fi
done
echo ""
echo "Detailed analysis:"
echo "  python3 ${LOAD_GEN_DIR}/analyze_results.py --results-dir ${RESULTS_DIR}"
