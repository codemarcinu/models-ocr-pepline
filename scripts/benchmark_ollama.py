#!/usr/bin/env python3
"""
Ollama Benchmark Script - Test local LLM performance and optimize settings.

Usage:
    python scripts/benchmark_ollama.py                    # Run full benchmark
    python scripts/benchmark_ollama.py --model gemma3:4b  # Test specific model
    python scripts/benchmark_ollama.py --quick            # Quick test (fewer iterations)

Measures:
- Cold start time (first request)
- Warm request time (subsequent requests)
- Tokens per second
- VRAM usage
"""

import argparse
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ollama
import torch
from utils.ollama_optimizer import (
    get_optimal_options,
    get_model_config,
    TaskType,
    MODEL_CONFIGS
)


def get_vram_usage() -> float:
    """Get current VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def get_vram_reserved() -> float:
    """Get reserved VRAM in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024**3)
    return 0.0


def benchmark_model(
    model_name: str,
    iterations: int = 3,
    test_prompts: list = None
) -> dict:
    """
    Benchmark a single model.

    Args:
        model_name: Ollama model name
        iterations: Number of warm iterations
        test_prompts: List of prompts to test

    Returns:
        Dict with benchmark results
    """
    if test_prompts is None:
        test_prompts = [
            ("short", "Napisz jedno zdanie o Pythonie."),
            ("medium", "Wyjaśnij w 3 zdaniach czym jest machine learning i podaj przykład zastosowania."),
            ("json", '{"task": "extract", "text": "Jan Kowalski ma 30 lat i mieszka w Warszawie"} Wyciągnij dane osobowe jako JSON.'),
        ]

    config = get_model_config(model_name)
    results = {
        "model": model_name,
        "config": {
            "num_ctx": config.num_ctx,
            "temperature": config.temperature,
            "vram_estimate": config.vram_estimate_gb
        },
        "tests": []
    }

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"Config: ctx={config.num_ctx}, temp={config.temperature}")
    print(f"{'='*60}")

    # Unload model first
    try:
        ollama.generate(model=model_name, prompt="", keep_alive=0)
    except:
        pass

    for prompt_type, prompt in test_prompts:
        test_result = {
            "type": prompt_type,
            "prompt_len": len(prompt),
            "cold_time_ms": 0,
            "warm_times_ms": [],
            "output_tokens": [],
            "tokens_per_sec": []
        }

        # Cold start (first request)
        print(f"\n  [{prompt_type}] Cold start...", end=" ", flush=True)
        options = get_optimal_options(model_name, TaskType.CHAT)

        start = time.time()
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options=options,
            keep_alive=config.keep_alive
        )
        cold_time = (time.time() - start) * 1000
        test_result["cold_time_ms"] = cold_time

        output = response["message"]["content"]
        output_tokens = len(output.split())
        test_result["output_tokens"].append(output_tokens)

        print(f"{cold_time:.0f}ms ({output_tokens} tokens)")

        # Warm iterations
        for i in range(iterations):
            print(f"  [{prompt_type}] Warm {i+1}/{iterations}...", end=" ", flush=True)

            start = time.time()
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options=options,
                keep_alive=config.keep_alive
            )
            warm_time = (time.time() - start) * 1000
            test_result["warm_times_ms"].append(warm_time)

            output = response["message"]["content"]
            output_tokens = len(output.split())
            test_result["output_tokens"].append(output_tokens)

            # Calculate tokens/sec
            if warm_time > 0:
                tps = output_tokens / (warm_time / 1000)
                test_result["tokens_per_sec"].append(tps)

            print(f"{warm_time:.0f}ms ({output_tokens} tokens, {tps:.1f} tok/s)")

        results["tests"].append(test_result)

    # Calculate averages
    all_warm_times = []
    all_tps = []
    for test in results["tests"]:
        all_warm_times.extend(test["warm_times_ms"])
        all_tps.extend(test["tokens_per_sec"])

    results["summary"] = {
        "avg_warm_time_ms": sum(all_warm_times) / len(all_warm_times) if all_warm_times else 0,
        "avg_tokens_per_sec": sum(all_tps) / len(all_tps) if all_tps else 0,
        "min_warm_time_ms": min(all_warm_times) if all_warm_times else 0,
        "max_warm_time_ms": max(all_warm_times) if all_warm_times else 0,
    }

    return results


def compare_context_sizes(model_name: str, sizes: list = None):
    """
    Compare performance with different context window sizes.

    Larger context = more VRAM but can handle longer conversations.
    """
    if sizes is None:
        sizes = [1024, 2048, 4096, 8192]

    print(f"\n{'='*60}")
    print(f"Context Window Comparison: {model_name}")
    print(f"{'='*60}")

    prompt = "Napisz krótkie zdanie."
    results = []

    for ctx_size in sizes:
        print(f"\n  Testing num_ctx={ctx_size}...", end=" ", flush=True)

        # Unload model
        try:
            ollama.generate(model=model_name, prompt="", keep_alive=0)
        except:
            pass

        options = {"num_ctx": ctx_size, "num_gpu": -1}

        times = []
        for _ in range(3):
            start = time.time()
            ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options=options,
                keep_alive="30s"
            )
            times.append((time.time() - start) * 1000)

        avg_time = sum(times) / len(times)
        results.append({"ctx": ctx_size, "avg_ms": avg_time})
        print(f"{avg_time:.0f}ms avg")

    return results


def print_summary(results: list):
    """Print benchmark summary."""
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    for result in results:
        summary = result.get("summary", {})
        print(f"\n{result['model']}:")
        print(f"  Avg warm time: {summary.get('avg_warm_time_ms', 0):.0f}ms")
        print(f"  Avg tokens/sec: {summary.get('avg_tokens_per_sec', 0):.1f}")
        print(f"  Min/Max: {summary.get('min_warm_time_ms', 0):.0f}ms / {summary.get('max_warm_time_ms', 0):.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama models")
    parser.add_argument("--model", "-m", help="Specific model to test")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick test (1 iteration)")
    parser.add_argument("--context", "-c", action="store_true", help="Test context window sizes")
    parser.add_argument("--all", "-a", action="store_true", help="Test all configured models")
    args = parser.parse_args()

    iterations = 1 if args.quick else 3

    if args.context:
        model = args.model or "gemma3:4b"
        compare_context_sizes(model)
        return

    if args.model:
        models = [args.model]
    elif args.all:
        # Get models from Ollama
        available = ollama.list()
        models = [m.model for m in available.models][:5]  # Limit to 5
    else:
        # Default test models
        models = ["gemma3:4b"]

    results = []
    for model in models:
        try:
            result = benchmark_model(model, iterations=iterations)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {model}: {e}")

    print_summary(results)


if __name__ == "__main__":
    main()
