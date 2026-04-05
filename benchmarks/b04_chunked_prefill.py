# -*- coding: utf-8 -*-
"""
Feature 4: Chunked Prefill
Impact of chunked prefill and max_num_batched_tokens on throughput and latency.
"""
import os

import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams

from config import *
from utils import benchmark, cleanup_engine, save_results


def run_benchmark():
    """Run the benchmark. Return results dict."""
    print("=" * 60)
    print(" Chunked Prefill + Disaggregated Prefill/Decode")
    print("=" * 60)

    chunk_configs = [
        {
            "name": "No Chunking\n(default)",
            "enable_chunked_prefill": False,
            "max_num_batched_tokens": None,
        },
    ]
    for cs in CHUNK_SIZES:
        chunk_configs.append(
            {
                "name": f"Chunk\n{cs} tok",
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": cs,
            }
        )

    long_prefix = "Provide a comprehensive analysis of the following topic. " * 30
    chunk_prompts = [
        long_prefix + t
        for t in [
            "Renewable energy trends in 2025.",
            "The future of autonomous vehicles.",
            "Advances in quantum computing.",
            "Impact of AI on healthcare.",
            "Space exploration milestones.",
            "Climate change mitigation strategies.",
            "The evolution of programming languages.",
            "Blockchain applications beyond cryptocurrency.",
        ]
    ]

    sp = SamplingParams(max_tokens=CHUNKED_PREFILL_MAX_TOKENS, temperature=0.7)
    chunk_results = []

    for cfg in chunk_configs:
        print(f"\n--- Config: {cfg['name'].replace(chr(10), ' ')} ---")
        kwargs = {
            "model": MODEL_ID,
            "dtype": DTYPE,
            "gpu_memory_utilization": GPU_MEMORY_UTIL,
            "max_model_len": MAX_MODEL_LEN,
            "enable_chunked_prefill": cfg["enable_chunked_prefill"],
            "trust_remote_code": True,
        }
        if cfg["max_num_batched_tokens"] is not None:
            kwargs["max_num_batched_tokens"] = cfg["max_num_batched_tokens"]

        engine = LLM(**kwargs)
        res = benchmark(engine, chunk_prompts, sp, label=cfg["name"].replace("\n", " "))
        res["config"] = cfg["name"]
        chunk_results.append(res)
        cleanup_engine(engine)

    best_tps = max(r["tokens_per_sec"] for r in chunk_results)
    results = {
        "_summary": {"feature": "Chunked Prefill", "tokens_per_sec": best_tps},
        "chunk_results": chunk_results,
    }

    print("\nChunked Prefill benchmark complete")

    save_results("04_chunked_prefill", results)
    return results


def plot_results(results=None):
    """Generate and save the plot."""
    if results is None:
        from utils import load_results

        results = load_results("04_chunked_prefill")

    chunk_results = results["chunk_results"]

    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Chunked Prefill — Chunk Size Impact",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text"],
        y=1.02,
    )

    cfg_names = [r["config"] for r in chunk_results]
    tps_vals = [r["tokens_per_sec"] for r in chunk_results]
    lat_vals = [r["elapsed_s"] for r in chunk_results]
    n_chunked = max(0, len(cfg_names) - 1)

    bars = ax1.bar(
        range(len(cfg_names)),
        tps_vals,
        color=[COLORS["accent3"]] + [COLORS["primary"]] * n_chunked,
        edgecolor="white",
        linewidth=0.5,
        width=0.6,
    )
    ax1.set_xticks(range(len(cfg_names)))
    ax1.set_xticklabels(cfg_names, fontsize=9)
    ax1.set_ylabel("Throughput (tokens/s)")
    ax1.set_title("Throughput by Chunk Size")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, tps_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(tps_vals) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLORS["text"],
        )

    bars = ax2.bar(
        range(len(cfg_names)),
        lat_vals,
        color=[COLORS["accent3"]] + [COLORS["secondary"]] * n_chunked,
        edgecolor="white",
        linewidth=0.5,
        width=0.6,
    )
    ax2.set_xticks(range(len(cfg_names)))
    ax2.set_xticklabels(cfg_names, fontsize=9)
    ax2.set_ylabel("Latency (seconds)")
    ax2.set_title("Latency by Chunk Size")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, lat_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(lat_vals) * 0.02,
            f"{val:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLORS["text"],
        )

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "plot_04_chunked_prefill.png")
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
