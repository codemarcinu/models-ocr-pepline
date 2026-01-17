#!/usr/bin/env python3
"""
OCR Benchmark Runner
Executes benchmarking suite and generates reports.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ocr_benchmark_engine import (
    OCRBenchmark,
    OCRProvider,
    GoogleVisionExtractor,
    GPT4oMiniExtractor,
    DeepSeekR1Extractor,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkReporter:
    """Generate reports and visualizations."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate_comparison_report(self, summary: dict) -> str:
        """Generate comprehensive comparison report."""
        report = []
        report.append("="*80)
        report.append("OCR BENCHMARKING REPORT")
        report.append("="*80)
        report.append("")

        # Overall statistics
        report.append("BENCHMARK OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total Tests: {summary['total_tests']}")
        report.append(f"Timestamp: {summary['timestamp']}")
        report.append("")

        # Provider comparison
        report.append("PROVIDER COMPARISON")
        report.append("-" * 40)

        providers_data = []
        for provider_name, metrics in summary["providers"].items():
            providers_data.append({
                "Provider": provider_name,
                "Tests": metrics["count"],
                "Field Accuracy": f"{metrics['avg_field_accuracy']*100:.2f}%",
                "Fuzzy Accuracy": f"{metrics['avg_fuzzy_accuracy']*100:.2f}%",
                "Char Error Rate": f"{metrics['avg_char_error_rate']*100:.2f}%",
                "Word Error Rate": f"{metrics['avg_word_error_rate']*100:.2f}%",
                "Avg Time (s)": f"{metrics['avg_processing_time']:.3f}",
                "Total Cost": f"${metrics['total_cost']:.4f}",
                "Completeness": f"{metrics['avg_field_completeness']*100:.2f}%",
                "Numerical Accuracy": f"{metrics['avg_numerical_accuracy']*100:.2f}%",
                "Consistency": f"{metrics['avg_consistency_score']:.2f}",
            })

        df = pd.DataFrame(providers_data)
        report.append(df.to_string(index=False))
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS FOR DEEPSEEK R1 OPTIMIZATION")
        report.append("-" * 40)

        if summary["providers"]:
            # Find best performer in each category
            best_accuracy = max(
                summary["providers"].items(),
                key=lambda x: x[1]["avg_field_accuracy"]
            )
            best_speed = min(
                summary["providers"].items(),
                key=lambda x: x[1]["avg_processing_time"]
            )
            best_cost = min(
                summary["providers"].items(),
                key=lambda x: x[1]["total_cost"]
            )

            report.append(f"\nBest Accuracy: {best_accuracy[0]}")
            report.append(f"  - Field Accuracy: {best_accuracy[1]['avg_field_accuracy']*100:.2f}%")
            report.append(f"  - Actions: Review prompt engineering, improve fuzzy matching")

            report.append(f"\nBest Speed: {best_speed[0]}")
            report.append(f"  - Avg Time: {best_speed[1]['avg_processing_time']:.3f}s")
            report.append(f"  - Actions: Optimize tokenization, batch processing")

            report.append(f"\nBest Cost: {best_cost[0]}")
            report.append(f"  - Total Cost: ${best_cost[1]['total_cost']:.4f}")
            report.append(f"  - Local model has natural cost advantage")

            report.append(f"\nDeepSeek R1 Optimization Strategy:")
            report.append(f"  1. Use same prompts as GPT-4o mini (benchmark)")
            report.append(f"  2. Implement fuzzy matching for ~80% similarity threshold")
            report.append(f"  3. Add validation layer for business rule consistency")
            report.append(f"  4. Optimize for speed while maintaining accuracy")
            report.append(f"  5. Deploy on local GPU for zero API costs")

        report.append("")
        report.append("="*80)

        return "\n".join(report)

    def create_visualizations(self, summary: dict):
        """Create visualization plots."""
        if not summary["providers"]:
            logger.warning("No data to visualize")
            return

        # Prepare data
        providers = list(summary["providers"].keys())
        metrics = summary["providers"]

        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle("OCR Benchmark Comparison", fontsize=16, fontweight="bold")

        # 1. Accuracy comparison
        ax = axes[0, 0]
        field_acc = [metrics[p]["avg_field_accuracy"] * 100 for p in providers]
        ax.bar(providers, field_acc, color="steelblue")
        ax.set_ylabel("Field Accuracy (%)")
        ax.set_title("Field-Level Accuracy")
        ax.set_ylim(0, 105)
        for i, v in enumerate(field_acc):
            ax.text(i, v + 2, f"{v:.1f}%", ha="center")

        # 2. Fuzzy accuracy
        ax = axes[0, 1]
        fuzzy_acc = [metrics[p]["avg_fuzzy_accuracy"] * 100 for p in providers]
        ax.bar(providers, fuzzy_acc, color="darkgreen")
        ax.set_ylabel("Fuzzy Accuracy (%)")
        ax.set_title("Fuzzy Matching Accuracy (>80%)")
        ax.set_ylim(0, 105)
        for i, v in enumerate(fuzzy_acc):
            ax.text(i, v + 2, f"{v:.1f}%", ha="center")

        # 3. Error rates
        ax = axes[0, 2]
        char_err = [metrics[p]["avg_char_error_rate"] * 100 for p in providers]
        word_err = [metrics[p]["avg_word_error_rate"] * 100 for p in providers]
        x = range(len(providers))
        width = 0.35
        ax.bar([i - width/2 for i in x], char_err, width, label="Char Error", color="coral")
        ax.bar([i + width/2 for i in x], word_err, width, label="Word Error", color="lightcoral")
        ax.set_ylabel("Error Rate (%)")
        ax.set_title("Error Rates")
        ax.set_xticks(x)
        ax.set_xticklabels(providers)
        ax.legend()

        # 4. Processing time
        ax = axes[1, 0]
        proc_time = [metrics[p]["avg_processing_time"] for p in providers]
        ax.bar(providers, proc_time, color="orange")
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Average Processing Time")
        for i, v in enumerate(proc_time):
            ax.text(i, v + 0.01, f"{v:.3f}s", ha="center")

        # 5. Cost comparison
        ax = axes[1, 1]
        costs = [metrics[p]["total_cost"] for p in providers]
        ax.bar(providers, costs, color="crimson")
        ax.set_ylabel("Cost ($)")
        ax.set_title("Total Cost")
        for i, v in enumerate(costs):
            ax.text(i, v + max(costs)*0.01, f"${v:.4f}", ha="center")

        # 6. Field completeness
        ax = axes[1, 2]
        completeness = [metrics[p]["avg_field_completeness"] * 100 for p in providers]
        ax.bar(providers, completeness, color="mediumslateblue")
        ax.set_ylabel("Completeness (%)")
        ax.set_title("Field Extraction Completeness")
        ax.set_ylim(0, 105)
        for i, v in enumerate(completeness):
            ax.text(i, v + 2, f"{v:.1f}%", ha="center")

        # 7. Numerical accuracy
        ax = axes[2, 0]
        num_acc = [metrics[p]["avg_numerical_accuracy"] * 100 for p in providers]
        ax.bar(providers, num_acc, color="teal")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Numerical Field Accuracy")
        ax.set_ylim(0, 105)
        for i, v in enumerate(num_acc):
            ax.text(i, v + 2, f"{v:.1f}%", ha="center")

        # 8. Consistency score
        ax = axes[2, 1]
        consistency = [metrics[p]["avg_consistency_score"] for p in providers]
        ax.bar(providers, consistency, color="darkviolet")
        ax.set_ylabel("Score (0-1)")
        ax.set_title("Business Logic Consistency")
        ax.set_ylim(0, 1.1)
        for i, v in enumerate(consistency):
            ax.text(i, v + 0.03, f"{v:.2f}", ha="center")

        # 9. Overall score (weighted average)
        ax = axes[2, 2]
        overall_scores = []
        for p in providers:
            score = (
                metrics[p]["avg_field_accuracy"] * 0.3 +
                metrics[p]["avg_fuzzy_accuracy"] * 0.2 +
                (1 - metrics[p]["avg_char_error_rate"]) * 0.2 +
                metrics[p]["avg_field_completeness"] * 0.15 +
                metrics[p]["avg_consistency_score"] * 0.15
            ) * 100
            overall_scores.append(score)
        
        colors = ["gold" if p == "gpt4o_mini" else ("lightgreen" if p == "deepseek_r1" else "lightblue") for p in providers]
        ax.bar(providers, overall_scores, color=colors)
        ax.set_ylabel("Score (%)")
        ax.set_title("Overall Performance Score")
        ax.set_ylim(0, 105)
        for i, v in enumerate(overall_scores):
            ax.text(i, v + 2, f"{v:.1f}%", ha="center")

        plt.tight_layout()
        plt.savefig(self.results_dir / "benchmark_comparison.png", dpi=300, bbox_inches="tight")
        logger.info(f"Visualization saved to {self.results_dir / 'benchmark_comparison.png'}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run OCR benchmarking suite"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("benchmarking/test_receipts"),
        help="Directory with test receipt images"
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=Path("benchmarking/ground_truth"),
        help="Directory with ground truth JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarking/results"),
        help="Output directory for results and reports"
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["google_vision", "gpt4o_mini", "deepseek_r1"],
        default=["gpt4o_mini", "deepseek_r1"],
        help="Providers to benchmark"
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization generation"
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        help="OpenAI API key (or use OPENAI_API_KEY env var)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting OCR benchmark...")
    logger.info(f"Test images: {args.image_dir}")
    logger.info(f"Output: {args.output_dir}")

    # Initialize benchmark
    benchmark = OCRBenchmark(ground_truth_dir=args.ground_truth_dir)

    # Register selected providers
    provider_map = {
        "gpt4o_mini": (
            OCRProvider.GPT4O_MINI,
            lambda: GPT4oMiniExtractor(api_key=args.openai_key)
        ),
        "deepseek_r1": (
            OCRProvider.DEEPSEEK_R1,
            lambda: DeepSeekR1Extractor()
        ),
        "google_vision": (
            OCRProvider.GOOGLE_VISION,
            lambda: GoogleVisionExtractor()
        ),
    }

    for provider_key in args.providers:
        if provider_key in provider_map:
            provider, extractor_fn = provider_map[provider_key]
            try:
                benchmark.register_extractor(provider, extractor_fn())
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_key}: {e}")

    # Run benchmark
    try:
        results = benchmark.run_benchmark(
            image_dir=args.image_dir,
            output_dir=args.output_dir
        )

        # Generate reports
        reporter = BenchmarkReporter(args.output_dir)
        report = reporter.generate_comparison_report(results)
        
        # Save report
        report_file = args.output_dir / "benchmark_report.txt"
        with open(report_file, "w") as f:
            f.write(report)
        
        print(report)
        logger.info(f"Report saved to {report_file}")

        # Generate visualizations
        if not args.skip_visualization:
            try:
                reporter.create_visualizations(results)
            except Exception as e:
                logger.error(f"Visualization failed: {e}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
