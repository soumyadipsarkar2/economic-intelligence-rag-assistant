# -*- coding: utf-8 -*-
"""
Feature 2: Continuous Batching
Throughput scaling and latency across batch sizes with mixed workloads.
"""
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from vllm import LLM, SamplingParams

from config import *
from utils import benchmark, cleanup_engine, make_test_prompts, save_results


def run_benchmark():
    """Run the benchmark. Return results dict."""
    print("=" * 60)
    print("  2. Continuous Batching")
    print("=" * 60)

    llm = LLM(
        model=MODEL_ID,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=256,
        trust_remote_code=True,
    )

    batch_sizes = CB_BATCH_SIZES
    cb_results = []

    for n in batch_sizes:
        prompts = make_test_prompts(n)
        sp_mixed = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.7)
        res = benchmark(llm, prompts, sp_mixed, label=f"batch={n}")
        res["batch_size"] = n
        cb_results.append(res)

    for r in cb_results:
        r["tps_per_seq"] = r["tokens_per_sec"] / r["batch_size"]

    results = {
        "_summary": {
            "feature": "Continuous Batching",
            "tokens_per_sec": cb_results[-1]["tokens_per_sec"],
        },
        "cb_results": cb_results,
    }

    cleanup_engine(llm)
    print("\nContinuous Batching benchmark complete")

    save_results("02_continuous_batching", results)
    return results


def plot_results(results=None):
    """Generate and save the plot."""
    if results is None:
        from utils import load_results

        results = load_results("02_continuous_batching")

    cb_results = results["cb_results"]

    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Continuous Batching — Throughput Scaling",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text"],
        y=1.02,
    )

    bs_vals = [r["batch_size"] for r in cb_results]
    tps_vals = [r["tokens_per_sec"] for r in cb_results]
    ax1.plot(
        bs_vals,
        tps_vals,
        "o-",
        color=COLORS["primary"],
        linewidth=2.5,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )
    ax1.fill_between(bs_vals, tps_vals, alpha=0.15, color=COLORS["primary"])
    ax1.set_xlabel("Batch Size (concurrent sequences)")
    ax1.set_ylabel("Total Throughput (tokens/s)")
    ax1.set_title("Aggregate Throughput")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

    latencies = [r["elapsed_s"] for r in cb_results]
    ax2.plot(
        bs_vals,
        latencies,
        "s-",
        color=COLORS["secondary"],
        linewidth=2.5,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )
    ax2.fill_between(bs_vals, latencies, alpha=0.15, color=COLORS["secondary"])
    ax2.set_xlabel("Batch Size (concurrent sequences)")
    ax2.set_ylabel("Total Wall Time (s)")
    ax2.set_title("Latency vs. Batch Size")
    ax2.set_xscale("log", base=2)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "plot_02_continuous_batching.png")
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
