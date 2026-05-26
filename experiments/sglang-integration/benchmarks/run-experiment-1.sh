#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Experiment 1: Routing Gap Baseline — Full Orchestration Script
# =============================================================================
#
# Prerequisites:
#   - kubectl configured and pointing to your GPU cluster
#   - Namespace created: kubectl create namespace $NAMESPACE
#   - HF token secret created:
#       kubectl create secret generic llm-d-hf-token \
#         --from-literal=HF_TOKEN=<your-token> -n $NAMESPACE
#   - Python 3.10+ with: pip install -r load-generator/requirements.txt
#   - Gateway API CRDs installed (for Step 2 only)
#
# Usage:
#   export NAMESPACE=llm-d-exp1
#   export GAIE_VERSION=v1.5.0
#   ./run-experiment-1.sh [step1|step2|step3|all]
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../configs"
RESULTS_DIR="${SCRIPT_DIR}/results/$(date +%Y%m%d-%H%M%S)"
LOAD_GEN_DIR="${SCRIPT_DIR}/load-generator"

NAMESPACE="${NAMESPACE:-llm-d-exp1}"
GAIE_VERSION="${GAIE_VERSION:-v1.5.0}"
MODEL="Qwen/Qwen2.5-7B-Instruct"
QPS="${QPS:-10}"
DURATION="${DURATION:-900}"  # 15 minutes
MAX_TOKENS="${MAX_TOKENS:-64}"

mkdir -p "${RESULTS_DIR}"

echo "=============================================="
echo "Experiment 1: Routing Gap Baseline"
echo "=============================================="
echo "Namespace:    ${NAMESPACE}"
echo "Model:        ${MODEL}"
echo "QPS:          ${QPS}"
echo "Duration:     ${DURATION}s"
echo "Results dir:  ${RESULTS_DIR}"
echo "=============================================="
echo ""

# --- Helper functions ---

wait_for_pods_ready() {
    local label=$1
    local expected=$2
    local timeout=${3:-600}

    echo "  Waiting for ${expected} pods with label '${label}' to be ready (timeout: ${timeout}s)..."
    kubectl wait --for=condition=ready pod \
        -l "${label}" \
        -n "${NAMESPACE}" \
        --timeout="${timeout}s" 2>/dev/null || {
        echo "  WARNING: Not all pods ready within timeout. Checking status..."
        kubectl get pods -l "${label}" -n "${NAMESPACE}"
        return 1
    }
    echo "  All ${expected} pods ready."
}

get_pod_ips() {
    local label=$1
    kubectl get pods -l "${label}" -n "${NAMESPACE}" \
        -o jsonpath='{range .items[*]}{.status.podIP}:8000,{end}' | sed 's/,$//'
}

get_service_ip() {
    local svc_name=$1
    kubectl get service "${svc_name}" -n "${NAMESPACE}" \
        -o jsonpath='{.spec.clusterIP}'
}

flush_caches() {
    local label=$1
    echo "  Flushing caches on all pods..."
    local pods
    pods=$(kubectl get pods -l "${label}" -n "${NAMESPACE}" -o jsonpath='{range .items[*]}{.metadata.name} {end}')
    for pod in ${pods}; do
        kubectl exec -n "${NAMESPACE}" "${pod}" -- \
            curl -s -X POST http://localhost:8000/flush_cache 2>/dev/null || true
    done
    echo "  Caches flushed. Waiting 10s for stabilization..."
    sleep 10
}

run_load_test() {
    local endpoint=$1
    local profile=$2
    local output_file=$3
    local pod_ips=$4

    echo "  Running load test: profile=${profile}, qps=${QPS}, duration=${DURATION}s"
    echo "  Endpoint: ${endpoint}"

    # Start metrics scraper in background
    local metrics_file="${output_file%.json}-metrics.json"
    python3 "${LOAD_GEN_DIR}/scrape_metrics.py" \
        --pods "${pod_ips}" \
        --interval 10 \
        --duration "${DURATION}" \
        --output "${metrics_file}" &
    local scraper_pid=$!

    # Run the load generator
    python3 "${LOAD_GEN_DIR}/run_benchmark.py" \
        --endpoint "${endpoint}" \
        --profile "${profile}" \
        --qps "${QPS}" \
        --duration "${DURATION}" \
        --max-tokens "${MAX_TOKENS}" \
        --model "${MODEL}" \
        --output "${output_file}"

    # Wait for scraper to finish
    wait "${scraper_pid}" 2>/dev/null || true

    echo "  Load test complete. Results: ${output_file}"
    echo ""
}

cleanup_step() {
    local step=$1
    echo "  Cleaning up step ${step} resources..."
    case ${step} in
        1)
            kubectl delete -n "${NAMESPACE}" -k "${CONFIG_DIR}/step1-round-robin" --ignore-not-found=true 2>/dev/null || true
            ;;
        2)
            helm uninstall exp1-heuristic -n "${NAMESPACE}" 2>/dev/null || true
            kubectl delete -n "${NAMESPACE}" -k "${CONFIG_DIR}/step2-llmd-heuristic" --ignore-not-found=true 2>/dev/null || true
            ;;
        3)
            kubectl delete -n "${NAMESPACE}" -k "${CONFIG_DIR}/step3-sglang-native" --ignore-not-found=true 2>/dev/null || true
            ;;
    esac
    echo "  Cleanup done. Waiting 15s for resources to terminate..."
    sleep 15
}

# =============================================================================
# STEP 1: Round-Robin (no routing intelligence)
# =============================================================================
run_step1() {
    echo ""
    echo "====== STEP 1: Round-Robin (no EPP) ======"
    echo ""

    # Deploy SGLang pods with simple service
    echo "  Deploying 4 SGLang pods with round-robin service..."
    kubectl apply -n "${NAMESPACE}" -k "${CONFIG_DIR}/step1-round-robin"

    wait_for_pods_ready "llm-d.ai/routing-config=round-robin" 4

    local svc_ip
    svc_ip=$(get_service_ip "exp1-sglang-rr-sglang-rr-service")
    local pod_ips
    pod_ips=$(get_pod_ips "llm-d.ai/routing-config=round-robin")

    echo "  Service IP: ${svc_ip}"
    echo "  Pod IPs: ${pod_ips}"

    # Run all three profiles
    for profile in A B C; do
        echo ""
        echo "  --- Profile ${profile} ---"
        flush_caches "llm-d.ai/routing-config=round-robin"
        run_load_test \
            "http://${svc_ip}:8000" \
            "${profile}" \
            "${RESULTS_DIR}/step1-rr-profile${profile}.json" \
            "${pod_ips}"
    done

    cleanup_step 1
    echo "====== STEP 1 COMPLETE ======"
}

# =============================================================================
# STEP 2: llm-d EPP with heuristic prefix-cache-scorer
# =============================================================================
run_step2() {
    echo ""
    echo "====== STEP 2: llm-d Heuristic Routing ======"
    echo ""

    # Deploy SGLang pods
    echo "  Deploying 4 SGLang pods..."
    kubectl apply -n "${NAMESPACE}" -k "${CONFIG_DIR}/step2-llmd-heuristic"

    wait_for_pods_ready "llm-d.ai/routing-config=llmd-heuristic" 4

    # Deploy EPP with heuristic scorer
    echo "  Deploying llm-d EPP with prefix-cache-scorer..."
    helm install exp1-heuristic \
        oci://registry.k8s.io/gateway-api-inference-extension/charts/standalone \
        -f "${REPO_ROOT}/guides/recipes/router/base.values.yaml" \
        -f "${CONFIG_DIR}/step2-llmd-heuristic/router-values.yaml" \
        -n "${NAMESPACE}" --version "${GAIE_VERSION}"

    echo "  Waiting for EPP to be ready..."
    kubectl wait --for=condition=available deployment/exp1-heuristic-epp \
        -n "${NAMESPACE}" --timeout=120s 2>/dev/null || true

    local epp_ip
    epp_ip=$(get_service_ip "exp1-heuristic-epp")
    local pod_ips
    pod_ips=$(get_pod_ips "llm-d.ai/routing-config=llmd-heuristic")

    echo "  EPP IP: ${epp_ip}"
    echo "  Pod IPs: ${pod_ips}"

    # Run all three profiles
    for profile in A B C; do
        echo ""
        echo "  --- Profile ${profile} ---"
        flush_caches "llm-d.ai/routing-config=llmd-heuristic"
        run_load_test \
            "http://${epp_ip}" \
            "${profile}" \
            "${RESULTS_DIR}/step2-heuristic-profile${profile}.json" \
            "${pod_ips}"
    done

    cleanup_step 2
    echo "====== STEP 2 COMPLETE ======"
}

# =============================================================================
# STEP 3: SGLang native router (sgl-router, cache-aware mode)
# =============================================================================
run_step3() {
    echo ""
    echo "====== STEP 3: SGLang Native Router ======"
    echo ""

    # Deploy SGLang workers + native router
    echo "  Deploying 4 SGLang pods + sgl-router..."
    kubectl apply -n "${NAMESPACE}" -k "${CONFIG_DIR}/step3-sglang-native"

    # Wait for workers first
    wait_for_pods_ready "app=sglang-native-worker" 4

    # Then wait for router
    kubectl wait --for=condition=available deployment/exp1-sglang-native-router \
        -n "${NAMESPACE}" --timeout=120s 2>/dev/null || true

    local router_ip
    router_ip=$(get_service_ip "exp1-sglang-native-router")
    local pod_ips
    pod_ips=$(get_pod_ips "app=sglang-native-worker")

    echo "  Router IP: ${router_ip}"
    echo "  Pod IPs: ${pod_ips}"

    # Run all three profiles
    for profile in A B C; do
        echo ""
        echo "  --- Profile ${profile} ---"
        flush_caches "app=sglang-native-worker"
        run_load_test \
            "http://${router_ip}:8080" \
            "${profile}" \
            "${RESULTS_DIR}/step3-native-profile${profile}.json" \
            "${pod_ips}"
    done

    cleanup_step 3
    echo "====== STEP 3 COMPLETE ======"
}

# =============================================================================
# Main
# =============================================================================

case "${1:-all}" in
    step1) run_step1 ;;
    step2) run_step2 ;;
    step3) run_step3 ;;
    all)
        run_step1
        run_step2
        run_step3
        echo ""
        echo "=============================================="
        echo "ALL STEPS COMPLETE"
        echo "Results directory: ${RESULTS_DIR}"
        echo "=============================================="
        echo ""
        echo "Next: Run the analysis script to compare results:"
        echo "  python3 ${LOAD_GEN_DIR}/analyze_results.py --results-dir ${RESULTS_DIR}"
        ;;
    *)
        echo "Usage: $0 [step1|step2|step3|all]"
        exit 1
        ;;
esac
