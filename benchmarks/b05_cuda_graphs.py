# -*- coding: utf-8 -*-
"""
Feature 5: CUDA Graphs
Eager vs CUDA graph mode throughput and batch scaling.
"""
import os

import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams

from config import *
from utils import benchmark, cleanup_engine, make_test_prompts, save_results


def run_benchmark():
    """Run the benchmark. Return results dict."""
    print("=" * 60)
    print("CUDA Graphs")
    print("=" * 60)

    sp = SamplingParams(max_tokens=CUDA_GRAPHS_MAX_TOKENS, temperature=0.7)
    cuda_prompts = make_test_prompts(16)

    print("\n--- Without CUDA Graphs ---")
    llm_no_cg = LLM(
        model=MODEL_ID,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        trust_remote_code=True,
    )
    res_no_cg = benchmark(llm_no_cg, cuda_prompts, sp, label="Eager (no CUDA graphs)")
    cleanup_engine(llm_no_cg)

    print("\n--- With CUDA Graphs ---")
    llm_cg = LLM(
        model=MODEL_ID,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=False,
        trust_remote_code=True,
    )
    res_cg = benchmark(llm_cg, cuda_prompts, sp, label="CUDA Graphs enabled")

    cg_batch_results = []
    for n in CG_BATCH_SIZES:
        p = make_test_prompts(n)
        res = benchmark(llm_cg, p, sp, label=f"CG batch={n}")
        res["batch_size"] = n
        cg_batch_results.append(res)

    cuda_graph_results = {
        "eager": res_no_cg,
        "cuda_graph": res_cg,
        "batch_scaling": cg_batch_results,
    }

    speedup = res_cg["tokens_per_sec"] / max(res_no_cg["tokens_per_sec"], 1)
    print(f"\n  CUDA Graphs speedup: {speedup:.2f}x")

    results = {
        "_summary": {"feature": "CUDA Graphs", "tokens_per_sec": res_cg["tokens_per_sec"]},
        "cuda_graph_results": cuda_graph_results,
    }

    cleanup_engine(llm_cg)
    print("\nCUDA Graphs benchmark complete")

    save_results("05_cuda_graphs", results)
    return results


def plot_results(results=None):
    """Generate and save the plot."""
    if results is None:
        from utils import load_results

        results = load_results("05_cuda_graphs")

    cuda_graph_results = results["cuda_graph_results"]
    cg_batch_results = cuda_graph_results["batch_scaling"]
    res_no_cg = cuda_graph_results["eager"]
    res_cg = cuda_graph_results["cuda_graph"]
    speedup = res_cg["tokens_per_sec"] / max(res_no_cg["tokens_per_sec"], 1)

    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "CUDA Graphs — Eager vs. Graph Mode",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text"],
        y=1.02,
    )

    labels = ["Eager Mode\n(no CUDA graphs)", "CUDA Graphs\nEnabled"]
    tps_vals = [res_no_cg["tokens_per_sec"], res_cg["tokens_per_sec"]]
    colors = [COLORS["secondary"], COLORS["accent1"]]
    bars = ax1.bar(labels, tps_vals, color=colors, edgecolor="white", linewidth=0.5, width=0.5)
    for bar, val in zip(bars, tps_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(tps_vals) * 0.02,
            f"{val:.0f} tok/s",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=COLORS["text"],
        )
    ax1.set_ylabel("Throughput (tokens/s)")
    ax1.set_title(f"Speedup: {speedup:.2f}x")
    ax1.grid(axis="y", alpha=0.3)

    bs = [r["batch_size"] for r in cg_batch_results]
    tps = [r["tokens_per_sec"] for r in cg_batch_results]
    ax2.plot(
        bs,
        tps,
        "o-",
        color=COLORS["accent1"],
        linewidth=2.5,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="CUDA Graphs",
    )
    ax2.fill_between(bs, tps, alpha=0.15, color=COLORS["accent1"])
    ax2.axhline(
        y=res_no_cg["tokens_per_sec"],
        color=COLORS["secondary"],
        linestyle="--",
        linewidth=1.5,
        label=f"Eager baseline ({res_no_cg['tokens_per_sec']:.0f} tok/s)",
    )
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Throughput (tokens/s)")
    ax2.set_title("CUDA Graphs Batch Scaling")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "plot_05_cuda_graphs.png")
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
