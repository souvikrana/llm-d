#!/usr/bin/env python3
"""
Experiment 1 Load Generator: Routing Gap Baseline

Sends requests at controlled QPS to an SGLang endpoint, measures:
  - TTFT (Time To First Token) via streaming
  - Total latency
  - Pod routing (from x-pod-id header if available)
  - Cache hit/miss (scraped from /metrics endpoint on each pod)

Usage:
  python run_benchmark.py \
    --endpoint http://<service-ip>:8000 \
    --profile A \
    --qps 10 \
    --duration 900 \
    --output results/step1-rr-profileA.json

  # For Step 3 (sgl-router), use port 8080:
  python run_benchmark.py \
    --endpoint http://<sgl-router-ip>:8080 \
    --profile A \
    --qps 10 \
    --duration 900 \
    --output results/step3-native-profileA.json
"""

import argparse
import asyncio
import json
import time
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp
import numpy as np

from traffic_profiles import (
    generate_profile_a,
    generate_profile_b,
    generate_profile_c,
)


@dataclass
class RequestResult:
    request_id: str
    profile: str
    ttft_ms: float  # Time to first token in milliseconds
    total_latency_ms: float  # Total request latency in milliseconds
    pod_id: str  # From x-pod-id header or "unknown"
    status: int  # HTTP status code
    error: str  # Empty string if no error
    timestamp: float  # Unix timestamp when request was sent
    tokens_generated: int


@dataclass
class BenchmarkConfig:
    endpoint: str
    profile: str
    qps: float
    duration_seconds: int
    max_tokens: int
    model: str


async def send_streaming_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    prompt: str,
    request_id: str,
    profile: str,
    max_tokens: int,
    model: str,
) -> RequestResult:
    """Send a single streaming request and measure TTFT + total latency."""

    url = f"{endpoint}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }

    start_time = time.perf_counter()
    timestamp = time.time()
    ttft = None
    tokens_generated = 0
    pod_id = "unknown"
    error = ""

    try:
        async with session.post(url, json=payload) as resp:
            status = resp.status
            pod_id = resp.headers.get("x-pod-id", "unknown")

            if status != 200:
                body = await resp.text()
                return RequestResult(
                    request_id=request_id,
                    profile=profile,
                    ttft_ms=-1,
                    total_latency_ms=(time.perf_counter() - start_time) * 1000,
                    pod_id=pod_id,
                    status=status,
                    error=f"HTTP {status}: {body[:200]}",
                    timestamp=timestamp,
                    tokens_generated=0,
                )

            # Stream SSE chunks
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded or not decoded.startswith("data:"):
                    continue

                data_str = decoded[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("text", ""):
                        tokens_generated += 1
                        if ttft is None:
                            ttft = (time.perf_counter() - start_time) * 1000
                except json.JSONDecodeError:
                    continue

    except asyncio.TimeoutError:
        error = "timeout"
        status = 0
    except Exception as e:
        error = str(e)[:200]
        status = 0

    total_latency = (time.perf_counter() - start_time) * 1000

    return RequestResult(
        request_id=request_id,
        profile=profile,
        ttft_ms=ttft if ttft is not None else -1,
        total_latency_ms=total_latency,
        pod_id=pod_id,
        status=status if status else 0,
        error=error,
        timestamp=timestamp,
        tokens_generated=tokens_generated,
    )


async def run_benchmark(config: BenchmarkConfig) -> list[RequestResult]:
    """Run the benchmark at controlled QPS for the specified duration."""

    # Generate enough requests for the full duration
    total_requests = int(config.qps * config.duration_seconds * 1.1)  # 10% buffer

    print(f"Generating {total_requests} requests for profile {config.profile}...")
    if config.profile == "A":
        requests = generate_profile_a(total_requests, config.max_tokens)
    elif config.profile == "B":
        requests = generate_profile_b(total_requests, config.max_tokens)
    elif config.profile == "C":
        requests = generate_profile_c(total_requests, config.max_tokens)
    else:
        raise ValueError(f"Unknown profile: {config.profile}")

    print(f"Starting benchmark: {config.qps} req/s for {config.duration_seconds}s")
    print(f"Endpoint: {config.endpoint}")
    print(f"Model: {config.model}")
    print(f"Max tokens: {config.max_tokens}")
    print()

    results: list[RequestResult] = []
    interval = 1.0 / config.qps
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        start = time.perf_counter()
        request_idx = 0

        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= config.duration_seconds:
                break

            if request_idx >= len(requests):
                request_idx = 0  # Wrap around

            req = requests[request_idx]
            task = asyncio.create_task(
                send_streaming_request(
                    session=session,
                    endpoint=config.endpoint,
                    prompt=req.prompt,
                    request_id=req.request_id,
                    profile=req.profile,
                    max_tokens=req.max_tokens,
                    model=config.model,
                )
            )
            tasks.append(task)
            request_idx += 1

            # Rate limiting: sleep until next request should be sent
            next_send = start + (len(tasks) * interval)
            sleep_time = next_send - time.perf_counter()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            # Print progress every 30 seconds
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                completed = sum(1 for t in tasks if t.done())
                print(f"  [{int(elapsed)}s] Sent: {len(tasks)}, Completed: {completed}")

        # Wait for all in-flight requests to complete
        print(f"\nAll requests sent ({len(tasks)} total). Waiting for completion...")
        results = await asyncio.gather(*tasks)

    return list(results)


def compute_stats(results: list[RequestResult]) -> dict:
    """Compute aggregate statistics from results."""

    successful = [r for r in results if r.status == 200 and r.ttft_ms > 0]
    failed = [r for r in results if r.status != 200 or r.ttft_ms <= 0]

    if not successful:
        return {"error": "No successful requests", "total": len(results), "failed": len(failed)}

    ttfts = np.array([r.ttft_ms for r in successful])
    latencies = np.array([r.total_latency_ms for r in successful])

    # Per-pod distribution
    pod_counts: dict[str, int] = {}
    for r in successful:
        pod_counts[r.pod_id] = pod_counts.get(r.pod_id, 0) + 1

    stats = {
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "ttft_ms": {
            "mean": float(np.mean(ttfts)),
            "p50": float(np.percentile(ttfts, 50)),
            "p90": float(np.percentile(ttfts, 90)),
            "p99": float(np.percentile(ttfts, 99)),
            "min": float(np.min(ttfts)),
            "max": float(np.max(ttfts)),
        },
        "total_latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "pod_distribution": pod_counts,
        "effective_qps": len(successful) / (
            (max(r.timestamp for r in successful) - min(r.timestamp for r in successful))
            if len(successful) > 1 else 1
        ),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Experiment 1 Load Generator")
    parser.add_argument("--endpoint", required=True, help="Base URL (e.g. http://svc-ip:8000)")
    parser.add_argument("--profile", required=True, choices=["A", "B", "C"])
    parser.add_argument("--qps", type=float, default=10.0, help="Requests per second")
    parser.add_argument("--duration", type=int, default=900, help="Duration in seconds (default 15min)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per request")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    config = BenchmarkConfig(
        endpoint=args.endpoint,
        profile=args.profile,
        qps=args.qps,
        duration_seconds=args.duration,
        max_tokens=args.max_tokens,
        model=args.model,
    )

    results = asyncio.run(run_benchmark(config))

    # Compute stats
    stats = compute_stats(results)

    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULTS: Profile {config.profile} @ {config.qps} req/s")
    print("=" * 60)
    print(f"Total requests:      {stats.get('total_requests', 0)}")
    print(f"Successful:          {stats.get('successful_requests', 0)}")
    print(f"Failed:              {stats.get('failed_requests', 0)}")
    if "ttft_ms" in stats:
        print(f"TTFT mean:           {stats['ttft_ms']['mean']:.1f} ms")
        print(f"TTFT p50:            {stats['ttft_ms']['p50']:.1f} ms")
        print(f"TTFT p90:            {stats['ttft_ms']['p90']:.1f} ms")
        print(f"TTFT p99:            {stats['ttft_ms']['p99']:.1f} ms")
        print(f"Total latency mean:  {stats['total_latency_ms']['mean']:.1f} ms")
        print(f"Effective QPS:       {stats.get('effective_qps', 0):.2f}")
    if "pod_distribution" in stats:
        print(f"Pod distribution:    {stats['pod_distribution']}")
    print("=" * 60)

    # Save full results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config),
        "stats": stats,
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
