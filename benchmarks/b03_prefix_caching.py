# -*- coding: utf-8 -*-
"""
Feature 3: Prefix Caching (Radix Cache)
Latency and throughput with vs without automatic prefix caching.
"""
import os

import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams

from config import *
from utils import benchmark, cleanup_engine, save_results


def run_benchmark():
    """Run the benchmark. Return results dict."""
    print("=" * 60)
    print("Prefix Caching (Radix Cache)")
    print("=" * 60)

    suffixes = [
        "What is dark matter and why is it important?",
        "How do mRNA vaccines work?",
        "Explain the concept of neural plasticity.",
        "What causes the northern lights?",
        "How does nuclear fusion generate energy?",
        "What is the role of mitochondria in cells?",
        "Explain how GPS satellites determine position.",
        "What is the double-slit experiment?",
    ]

    sp = SamplingParams(max_tokens=PREFIX_CACHE_MAX_TOKENS, temperature=0.7)

    print("\n--- Without Prefix Caching ---")
    llm_no_cache = LLM(
        model=MODEL_ID,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=False,
        trust_remote_code=True,
    )

    prompts_with_prefix = [SHARED_PREFIX + s for s in suffixes]

    res_no_cache_1 = benchmark(
        llm_no_cache, prompts_with_prefix, sp, label="No cache (run 1)"
    )
    res_no_cache_2 = benchmark(
        llm_no_cache, prompts_with_prefix, sp, label="No cache (run 2)"
    )
    cleanup_engine(llm_no_cache)

    print("\n--- With Prefix Caching (APC) ---")
    llm_cache = LLM(
        model=MODEL_ID,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )

    res_cache_1 = benchmark(
        llm_cache, prompts_with_prefix, sp, label="Cached (run 1 - cold)"
    )
    res_cache_2 = benchmark(
        llm_cache, prompts_with_prefix, sp, label="Cached (run 2 - warm)"
    )

    prefix_results = {
        "no_cache_run1": res_no_cache_1,
        "no_cache_run2": res_no_cache_2,
        "cache_run1": res_cache_1,
        "cache_run2": res_cache_2,
    }

    speedup = res_no_cache_2["elapsed_s"] / max(res_cache_2["elapsed_s"], 1e-6)
    print(f"\n   🚀 Warm cache speedup: {speedup:.2f}x")

    results = {
        "_summary": {
            "feature": "Prefix Caching",
            "tokens_per_sec": res_cache_2["tokens_per_sec"],
        },
        "prefix_results": prefix_results,
    }

    cleanup_engine(llm_cache)
    print("\nPrefix Caching benchmark complete")

    save_results("03_prefix_caching", results)
    return results


def plot_results(results=None):
    """Generate and save the plot."""
    if results is None:
        from utils import load_results

        results = load_results("03_prefix_caching")

    prefix_results = results["prefix_results"]

    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Prefix Caching (Radix Cache) — With vs Without",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text"],
        y=1.02,
    )

    labels = [
        "No Cache\n(Run 1)",
        "No Cache\n(Run 2)",
        "APC\n(Cold)",
        "APC\n(Warm)",
    ]
    times = [
        prefix_results["no_cache_run1"]["elapsed_s"],
        prefix_results["no_cache_run2"]["elapsed_s"],
        prefix_results["cache_run1"]["elapsed_s"],
        prefix_results["cache_run2"]["elapsed_s"],
    ]
    colors = [
        COLORS["secondary"],
        COLORS["secondary"],
        COLORS["primary"],
        COLORS["accent1"],
    ]
    bars = ax1.bar(labels, times, color=colors, edgecolor="white", linewidth=0.5, width=0.6)
    for bar, val in zip(bars, times):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.02,
            f"{val:.3f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=COLORS["text"],
        )
    ax1.set_ylabel("Latency (seconds)")
    ax1.set_title("Latency Comparison")
    ax1.grid(axis="y", alpha=0.3)

    tps_vals = [
        prefix_results["no_cache_run1"]["tokens_per_sec"],
        prefix_results["no_cache_run2"]["tokens_per_sec"],
        prefix_results["cache_run1"]["tokens_per_sec"],
        prefix_results["cache_run2"]["tokens_per_sec"],
    ]
    bars = ax2.bar(labels, tps_vals, color=colors, edgecolor="white", linewidth=0.5, width=0.6)
    for bar, val in zip(bars, tps_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(tps_vals) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=COLORS["text"],
        )
    ax2.set_ylabel("Throughput (tokens/s)")
    ax2.set_title("Throughput Comparison")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "plot_03_prefix_caching.png")
    plt.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=COLORS["bg_dark"],
    )
    plt.close(fig)
    print(f"Plot saved: {out_path}")


if __name__ == "__main__":
    from config import ensure_hf_login

    ensure_hf_login()
    r = run_benchmark()
    plot_results(r)
