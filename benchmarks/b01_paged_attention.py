# -*- coding: utf-8 -*-
"""
Feature 1: PagedAttention + Block Manager
Memory efficiency, throughput scaling, and block table heatmap.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from vllm import LLM, SamplingParams

from config import *
from utils import benchmark, cleanup_engine, make_test_prompts, save_results


def run_benchmark():
    """Run the benchmark. Return results dict."""
    print("=" * 60)
    print("  PagedAttention + Block Manager")
    print("=" * 60)

    llm = LLM(
        model=MODEL_ID,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=256,
        block_size=16,
        trust_remote_code=True,
    )

    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.7, top_p=0.9)

    _ = llm.generate(["Hello"], SamplingParams(max_tokens=1))
    torch.cuda.synchronize()

    model_base_mem = torch.cuda.memory_reserved() / 1e9

    batch_counts = BATCH_SIZES
    paged_results = []

    for n in batch_counts:
        prompts = make_test_prompts(n)
        res = benchmark(llm, prompts, sp, label=f"batch={n}")
        res["batch_size"] = n
        post_mem = torch.cuda.memory_reserved() / 1e9
        tokens_processed = res["total_tokens"] + n * 20
        kv_estimate_gb = tokens_processed * 32 * 2 * 128 * 8 * 2 / 1e9
        res["gpu_peak_mem_gb"] = post_mem
        res["kv_cache_estimate_gb"] = kv_estimate_gb
        res["model_mem_gb"] = model_base_mem
        paged_results.append(res)
        print(
            f"   GPU mem: {post_mem:.2f} GB | Model base: {model_base_mem:.2f} GB | "
            f"KV est: {kv_estimate_gb:.2f} GB"
        )

    heatmap_prompts = make_test_prompts(16)

    varied_outputs = []
    varied_max_tokens = [
        32,
        64,
        96,
        128,
        160,
        200,
        48,
        80,
        110,
        150,
        180,
        60,
        100,
        140,
        70,
        190,
    ]
    for i, prompt in enumerate(heatmap_prompts):
        hm_sp = SamplingParams(max_tokens=varied_max_tokens[i], temperature=0.8)
        out = llm.generate([prompt], hm_sp)
        varied_outputs.extend(out)

    block_size = 16
    seq_block_data = []
    for i, out in enumerate(varied_outputs):
        prompt_tokens = len(out.prompt_token_ids)
        gen_tokens = len(out.outputs[0].token_ids)
        total_tokens = prompt_tokens + gen_tokens
        num_blocks = (total_tokens + block_size - 1) // block_size
        seq_block_data.append(
            {
                "seq_id": i,
                "prompt_tokens": prompt_tokens,
                "gen_tokens": gen_tokens,
                "total_tokens": total_tokens,
                "blocks_used": num_blocks,
            }
        )

    results = {
        "_summary": {
            "feature": "PagedAttention",
            "tokens_per_sec": paged_results[-1]["tokens_per_sec"],
        },
        "paged_results": paged_results,
        "seq_block_data": seq_block_data,
    }

    cleanup_engine(llm)
    print("\nPagedAttention benchmark complete")

    save_results("01_paged_attention", results)
    return results


def plot_results(results=None):
    """Generate and save the plot."""
    if results is None:
        from utils import load_results

        results = load_results("01_paged_attention")

    paged_results = results["paged_results"]
    seq_block_data = results["seq_block_data"]

    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "PagedAttention + Block Manager",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text"],
        y=1.02,
    )

    ax = axes[0]
    bs = [r["batch_size"] for r in paged_results]
    tps = [r["tokens_per_sec"] for r in paged_results]
    bars = ax.bar(
        range(len(bs)),
        tps,
        color=PALETTE[: len(bs)],
        edgecolor="white",
        linewidth=0.5,
        width=0.7,
    )
    ax.set_xticks(range(len(bs)))
    ax.set_xticklabels(bs)
    ax.set_xlabel("Concurrent Sequences")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput Scaling")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, tps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(tps) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS["text"],
            fontweight="bold",
        )

    ax = axes[1]
    peak_mem = [r["gpu_peak_mem_gb"] for r in paged_results]
    kv_mem = [r["kv_cache_estimate_gb"] for r in paged_results]
    model_base = paged_results[0]["model_mem_gb"]
    x = np.arange(len(bs))

    ax.bar(
        x,
        peak_mem,
        0.6,
        color=COLORS["accent5"],
        alpha=0.7,
        label="Total Reserved",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x,
        kv_mem,
        0.6,
        color=COLORS["accent1"],
        alpha=0.8,
        label="KV Cache (est.)",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(
        y=model_base,
        color=COLORS["primary"],
        linestyle="--",
        linewidth=2,
        label=f"Model Weights ({model_base:.1f} GB)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bs)
    ax.set_xlabel("Concurrent Sequences")
    ax.set_ylabel("GPU Memory (GB)")
    ax.set_title("Memory Efficiency")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    for i, val in enumerate(peak_mem):
        ax.text(
            x[i],
            val + max(peak_mem) * 0.02,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["text"],
        )

    ax = axes[2]

    total_blocks_needed = sum(s["blocks_used"] for s in seq_block_data)
    total_gpu_blocks = total_blocks_needed + 10
    num_seqs = len(seq_block_data)

    heatmap_matrix = np.zeros((total_gpu_blocks, num_seqs))

    block_cursor = 0
    for s in seq_block_data:
        prompt_blocks = (s["prompt_tokens"] + 16 - 1) // 16
        for b in range(s["blocks_used"]):
            if block_cursor < total_gpu_blocks:
                if b < prompt_blocks:
                    heatmap_matrix[block_cursor][s["seq_id"]] = 1
                else:
                    heatmap_matrix[block_cursor][s["seq_id"]] = 2
                block_cursor += 1

    heatmap_matrix = heatmap_matrix.T

    cmap = ListedColormap(
        [COLORS["bg_dark"], COLORS["primary"], COLORS["accent1"]]
    )
    sns.heatmap(
        heatmap_matrix,
        ax=ax,
        cmap=cmap,
        linewidths=0.5,
        linecolor=COLORS["grid"],
        cbar=False,
        xticklabels=5,
        yticklabels=1,
        vmin=0,
        vmax=2,
    )
    ax.set_xlabel("GPU Block Pool Slot")
    ax.set_ylabel("Concurrent Request")
    ax.set_title("Block Table — Concurrent Allocation")
    legend_elements = [
        Patch(facecolor=COLORS["bg_dark"], edgecolor="white", label="Free"),
        Patch(facecolor=COLORS["primary"], edgecolor="white", label="Prompt KV"),
        Patch(facecolor=COLORS["accent1"], edgecolor="white", label="Decode KV"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    print("\nBlock Table Summary:")
    print(f"{'Seq':>4} {'Prompt':>7} {'Gen':>5} {'Total':>6} {'Blocks':>7}")
    for s in seq_block_data:
        print(
            f"{s['seq_id']:>4} {s['prompt_tokens']:>7} {s['gen_tokens']:>5} "
            f"{s['total_tokens']:>6} {s['blocks_used']:>7}"
        )

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "plot_01_paged_attention.png")
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
