#!/usr/bin/env python3
"""
Analyze Experiment 1 results and produce a decision recommendation.

Usage:
  python analyze_results.py --results-dir ./results/20260526-143000
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_metrics(path: Path) -> dict | None:
    metrics_path = Path(str(path).replace(".json", "-metrics.json"))
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def format_table(headers: list[str], rows: list[list[str]], col_widths: list[int] = None) -> str:
    """Format a simple ASCII table."""
    if not col_widths:
        col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) + 2
                      for i, h in enumerate(headers)]

    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    sep_line = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"

    lines = [header_line, sep_line]
    for row in rows:
        lines.append("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)) + " |")

    return "\n".join(lines)


def analyze(results_dir: Path):
    print("=" * 70)
    print("EXPERIMENT 1: ROUTING GAP BASELINE — ANALYSIS")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print()

    # Load all results
    configs = {
        "Round-Robin": "step1-rr",
        "llm-d Heuristic": "step2-heuristic",
        "SGLang Native": "step3-native",
    }
    profiles = ["A", "B", "C"]

    # --- Profile A comparison (primary decision metric) ---
    print("=" * 70)
    print("PROFILE A: High Prefix Sharing (Primary Decision Metric)")
    print("=" * 70)
    print()

    headers = ["Config", "Cache Hit Rate", "TTFT mean (ms)", "TTFT p90 (ms)", "TTFT p99 (ms)", "Requests"]
    rows = []

    profile_a_data = {}

    for config_name, prefix in configs.items():
        result = load_result(results_dir / f"{prefix}-profileA.json")
        metrics = load_metrics(results_dir / f"{prefix}-profileA.json")

        if result is None:
            rows.append([config_name, "N/A", "N/A", "N/A", "N/A", "N/A"])
            continue

        stats = result.get("stats", {})
        ttft = stats.get("ttft_ms", {})
        cache_hit_rate = "N/A"

        if metrics:
            agg = metrics.get("aggregate", {})
            cache_hit_rate = f"{agg.get('aggregate_cache_hit_rate', 0):.1%}"
            profile_a_data[config_name] = {
                "cache_hit_rate": agg.get("aggregate_cache_hit_rate", 0),
                "ttft_mean": ttft.get("mean", 0),
                "ttft_p90": ttft.get("p90", 0),
            }
        else:
            profile_a_data[config_name] = {
                "cache_hit_rate": 0,
                "ttft_mean": ttft.get("mean", 0),
                "ttft_p90": ttft.get("p90", 0),
            }

        rows.append([
            config_name,
            cache_hit_rate,
            f"{ttft.get('mean', 0):.1f}",
            f"{ttft.get('p90', 0):.1f}",
            f"{ttft.get('p99', 0):.1f}",
            str(stats.get("successful_requests", 0)),
        ])

    print(format_table(headers, rows))
    print()

    # --- Profile B and C ---
    for profile in ["B", "C"]:
        profile_label = {
            "B": "PROFILE B: Moderate Prefix Sharing",
            "C": "PROFILE C: No Prefix Sharing (Control)",
        }[profile]

        print(f"\n{'=' * 70}")
        print(profile_label)
        print("=" * 70)
        print()

        rows = []
        for config_name, prefix in configs.items():
            result = load_result(results_dir / f"{prefix}-profile{profile}.json")
            metrics = load_metrics(results_dir / f"{prefix}-profile{profile}.json")

            if result is None:
                rows.append([config_name, "N/A", "N/A", "N/A", "N/A", "N/A"])
                continue

            stats = result.get("stats", {})
            ttft = stats.get("ttft_ms", {})
            cache_hit_rate = "N/A"
            if metrics:
                agg = metrics.get("aggregate", {})
                cache_hit_rate = f"{agg.get('aggregate_cache_hit_rate', 0):.1%}"

            rows.append([
                config_name,
                cache_hit_rate,
                f"{ttft.get('mean', 0):.1f}",
                f"{ttft.get('p90', 0):.1f}",
                f"{ttft.get('p99', 0):.1f}",
                str(stats.get("successful_requests", 0)),
            ])

        print(format_table(headers, rows))

    # --- Decision recommendation ---
    print(f"\n\n{'=' * 70}")
    print("DECISION RECOMMENDATION")
    print("=" * 70)
    print()

    rr = profile_a_data.get("Round-Robin", {})
    heuristic = profile_a_data.get("llm-d Heuristic", {})
    native = profile_a_data.get("SGLang Native", {})

    if not rr or not heuristic:
        print("Insufficient data to make a recommendation.")
        print("Ensure Step 1 and Step 2 results are available.")
        return

    # Compute gaps
    rr_hit = rr.get("cache_hit_rate", 0)
    heuristic_hit = heuristic.get("cache_hit_rate", 0)
    native_hit = native.get("cache_hit_rate", 0) if native else 0

    gap_rr_to_heuristic = (heuristic_hit - rr_hit) * 100  # percentage points
    gap_rr_to_native = (native_hit - rr_hit) * 100 if native else 0

    ttft_improvement = 0
    if rr.get("ttft_mean", 0) > 0:
        ttft_improvement = ((rr["ttft_mean"] - heuristic.get("ttft_mean", 0)) / rr["ttft_mean"]) * 100

    print(f"Cache hit rate gap (RR → Heuristic):  {gap_rr_to_heuristic:+.1f} pp")
    print(f"Cache hit rate gap (RR → Native):     {gap_rr_to_native:+.1f} pp")
    print(f"TTFT improvement (RR → Heuristic):    {ttft_improvement:.1f}%")
    if native:
        native_vs_heuristic = (native_hit - heuristic_hit) * 100
        print(f"Cache hit rate gap (Heuristic → Native): {native_vs_heuristic:+.1f} pp")
    print()

    # Apply decision framework
    if gap_rr_to_heuristic < 10:
        print(">>> CASE 1: Gap < 10%")
        print("    KVEvents investment is LOW PRIORITY.")
        print("    The heuristic is good enough for this workload class.")
        print("    Redirect effort to P/D disaggregation or latency predictor.")
    elif gap_rr_to_heuristic < 30 and ttft_improvement < 50:
        print(">>> CASE 2: Gap 10-30%, TTFT improvement 20-40%")
        print("    KVEvents native publisher is HIGH PRIORITY but not a blocker.")
        print("    Ship llm-d with heuristic routing for SGLang now.")
        print("    Build KVEvents in parallel as next milestone.")
    elif gap_rr_to_heuristic >= 30 or ttft_improvement >= 50:
        print(">>> CASE 3: Gap > 30% or TTFT > 50%")
        print("    KVEvents native publisher is CRITICAL PATH.")
        print("    Do NOT ship SGLang support without it.")
        print("    Prioritize Experiment 2 (LMCache stability) immediately.")

    if native and native_hit > heuristic_hit + 0.05:
        print()
        print(">>> CASE 4 DETECTED: SGLang native router beats llm-d heuristic")
        print("    Do NOT ship llm-d heuristic as an improvement over native router.")
        print("    Options: (a) fix heuristic blind spots, or (b) double down on KVEvents.")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment 1 results")
    parser.add_argument("--results-dir", required=True, help="Path to results directory")
    args = parser.parse_args()

    analyze(Path(args.results_dir))


if __name__ == "__main__":
    main()
