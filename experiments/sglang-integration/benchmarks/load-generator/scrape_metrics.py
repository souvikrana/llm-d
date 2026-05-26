#!/usr/bin/env python3
"""
Scrape SGLang /metrics endpoint from each pod to get ground-truth cache hit rates.

Usage:
  # Scrape all pods in the experiment namespace
  python scrape_metrics.py \
    --pods "10.0.1.10:8000,10.0.1.11:8000,10.0.1.12:8000,10.0.1.13:8000" \
    --output results/step1-rr-profileA-metrics.json

  # Or use kubectl port-forward and scrape localhost
  python scrape_metrics.py \
    --pods "localhost:8001,localhost:8002,localhost:8003,localhost:8004" \
    --output results/metrics.json

  # Continuous scraping during benchmark (every 10s for 15min)
  python scrape_metrics.py \
    --pods "10.0.1.10:8000,10.0.1.11:8000,10.0.1.12:8000,10.0.1.13:8000" \
    --interval 10 \
    --duration 900 \
    --output results/step1-rr-profileA-metrics.json
"""

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp
import asyncio


@dataclass
class PodMetrics:
    pod_address: str
    timestamp: float
    cache_hit_tokens: float
    cache_miss_tokens: float
    cache_hit_rate: float
    running_requests: float
    waiting_requests: float
    raw_metrics: dict  # All parsed metrics


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Parse Prometheus text format into a dict of metric_name -> value."""
    metrics = {}
    for line in text.strip().split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        # Match: metric_name{labels} value  OR  metric_name value
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?[^}]*\}?\s+([0-9eE.+\-]+|NaN|Inf|-Inf)$', line)
        if match:
            name = match.group(1)
            try:
                value = float(match.group(2))
            except ValueError:
                value = 0.0
            # Accumulate if same metric appears multiple times (different labels)
            if name in metrics:
                metrics[name] += value
            else:
                metrics[name] = value
    return metrics


async def scrape_pod(session: aiohttp.ClientSession, pod_address: str) -> PodMetrics:
    """Scrape /metrics from a single pod."""
    url = f"http://{pod_address}/metrics"
    timestamp = time.time()

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return PodMetrics(
                    pod_address=pod_address,
                    timestamp=timestamp,
                    cache_hit_tokens=0,
                    cache_miss_tokens=0,
                    cache_hit_rate=0,
                    running_requests=0,
                    waiting_requests=0,
                    raw_metrics={},
                )
            text = await resp.text()
    except Exception as e:
        print(f"  Warning: Failed to scrape {pod_address}: {e}")
        return PodMetrics(
            pod_address=pod_address,
            timestamp=timestamp,
            cache_hit_tokens=0,
            cache_miss_tokens=0,
            cache_hit_rate=0,
            running_requests=0,
            waiting_requests=0,
            raw_metrics={},
        )

    metrics = parse_prometheus_metrics(text)

    # Engine-agnostic: try SGLang metric names first, then vLLM names.
    # SGLang exposes per-token counters; vLLM exposes per-query counters
    # plus a hit_rate gauge — we compute hit rate from whichever pair is present.

    # SGLang token-level counters (preferred — most precise)
    cache_hit = (
        metrics.get("sglang_cache_hit_tokens_total", 0)
        or metrics.get("sglang:cache_hit_tokens_total", 0)
        or metrics.get("cache_hit_tokens_total", 0)
        # vLLM query/block-level counters
        or metrics.get("vllm:gpu_prefix_cache_hits_total", 0)
        or metrics.get("vllm:gpu_prefix_cache_hit_tokens", 0)
    )
    cache_miss = (
        metrics.get("sglang_cache_miss_tokens_total", 0)
        or metrics.get("sglang:cache_miss_tokens_total", 0)
        or metrics.get("cache_miss_tokens_total", 0)
        # vLLM doesn't expose misses directly; compute from queries - hits
    )

    # vLLM exposes total queries, so derive miss count if we have it
    if cache_miss == 0 and cache_hit > 0:
        vllm_queries = (
            metrics.get("vllm:gpu_prefix_cache_queries_total", 0)
            or metrics.get("vllm:gpu_prefix_cache_query_tokens", 0)
        )
        if vllm_queries > 0:
            cache_miss = max(0, vllm_queries - cache_hit)

    total = cache_hit + cache_miss
    hit_rate = cache_hit / total if total > 0 else 0.0

    # Fallback: vLLM also exposes a precomputed hit_rate gauge
    if hit_rate == 0:
        hit_rate = metrics.get("vllm:gpu_prefix_cache_hit_rate", 0)

    running = (
        metrics.get("sglang_running_requests", 0)
        or metrics.get("sglang:running_requests", 0)
        or metrics.get("vllm:num_requests_running", 0)
    )
    waiting = (
        metrics.get("sglang_waiting_requests", 0)
        or metrics.get("sglang:waiting_requests", 0)
        or metrics.get("vllm:num_requests_waiting", 0)
    )

    return PodMetrics(
        pod_address=pod_address,
        timestamp=timestamp,
        cache_hit_tokens=cache_hit,
        cache_miss_tokens=cache_miss,
        cache_hit_rate=hit_rate,
        running_requests=running,
        waiting_requests=waiting,
        raw_metrics=metrics,
    )


async def scrape_all_pods(pod_addresses: list[str]) -> list[PodMetrics]:
    """Scrape all pods concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [scrape_pod(session, addr) for addr in pod_addresses]
        return await asyncio.gather(*tasks)


def compute_aggregate(snapshots: list[list[PodMetrics]]) -> dict:
    """Compute aggregate cache hit rates across all snapshots."""
    if not snapshots:
        return {}

    # Use the last snapshot for final state (counters are cumulative)
    final = snapshots[-1]

    total_hits = sum(p.cache_hit_tokens for p in final)
    total_misses = sum(p.cache_miss_tokens for p in final)
    total = total_hits + total_misses

    per_pod = {}
    for p in final:
        pod_total = p.cache_hit_tokens + p.cache_miss_tokens
        per_pod[p.pod_address] = {
            "cache_hit_tokens": p.cache_hit_tokens,
            "cache_miss_tokens": p.cache_miss_tokens,
            "cache_hit_rate": p.cache_hit_tokens / pod_total if pod_total > 0 else 0,
            "running_requests": p.running_requests,
            "waiting_requests": p.waiting_requests,
        }

    return {
        "aggregate_cache_hit_rate": total_hits / total if total > 0 else 0,
        "total_cache_hit_tokens": total_hits,
        "total_cache_miss_tokens": total_misses,
        "per_pod": per_pod,
        "num_snapshots": len(snapshots),
    }


async def main_async(args):
    pod_addresses = [addr.strip() for addr in args.pods.split(",")]
    print(f"Scraping {len(pod_addresses)} pods: {pod_addresses}")

    all_snapshots = []

    if args.interval and args.duration:
        # Continuous scraping mode
        print(f"Continuous mode: every {args.interval}s for {args.duration}s")
        start = time.time()
        while time.time() - start < args.duration:
            snapshot = await scrape_all_pods(pod_addresses)
            all_snapshots.append(snapshot)
            elapsed = time.time() - start
            print(f"  [{int(elapsed)}s] Scraped. Hit rates: "
                  + ", ".join(f"{p.cache_hit_rate:.2%}" for p in snapshot))
            await asyncio.sleep(args.interval)
    else:
        # Single scrape
        snapshot = await scrape_all_pods(pod_addresses)
        all_snapshots.append(snapshot)
        for p in snapshot:
            print(f"  {p.pod_address}: hit_rate={p.cache_hit_rate:.2%} "
                  f"(hits={p.cache_hit_tokens}, misses={p.cache_miss_tokens})")

    # Compute aggregates
    aggregate = compute_aggregate(all_snapshots)

    print(f"\nAggregate cache hit rate: {aggregate.get('aggregate_cache_hit_rate', 0):.2%}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "aggregate": aggregate,
        "snapshots": [
            [asdict(p) for p in snapshot]
            for snapshot in all_snapshots
        ],
    }

    # Remove raw_metrics from output to keep file size reasonable
    for snapshot in output_data["snapshots"]:
        for pod in snapshot:
            pod.pop("raw_metrics", None)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Metrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape SGLang pod metrics")
    parser.add_argument("--pods", required=True, help="Comma-separated pod_ip:port list")
    parser.add_argument("--interval", type=int, default=0, help="Scrape interval in seconds (0=single)")
    parser.add_argument("--duration", type=int, default=0, help="Total scraping duration in seconds")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
