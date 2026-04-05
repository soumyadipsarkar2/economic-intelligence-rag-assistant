# -*- coding: utf-8 -*-
"""
Load benchmark JSON results and render the combined white-theme dashboard
(same layout as the notebook LinkedIn export).
"""
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from config import (
    COLORS_WHITE,
    GPU_MEMORY_UTIL,
    MAX_MODEL_LEN,
    MODEL_ID,
    PALETTE_WHITE,
    PLOTS_DIR,
    setup_plot_style_white,
)
from utils import load_results


def _require_results():
    """Load all eight result files; raise if any missing."""
    keys = [
        "01_paged_attention",
        "02_continuous_batching",
        "03_prefix_caching",
        "04_chunked_prefill",
        "05_cuda_graphs",
        "06_speculative_decoding",
        "07_quantization",
        "08_multi_lora",
    ]
    data = {}
    missing = []
    for k in keys:
        try:
            data[k] = load_results(k)
        except OSError:
            missing.append(k)
    if missing:
        raise FileNotFoundError(
            "Missing result JSON(s): "
            + ", ".join(missing)
            + ". Run benchmarks first (python run_all.py)."
        )
    return data


def generate_dashboard():
    """Build ALL_RESULTS from JSON files and save plots/vllm_benchmark_linkedin.png."""
    data = _require_results()

    ALL_RESULTS = {}
    for k in data:
        s = data[k]["_summary"]
        ALL_RESULTS[s["feature"]] = s["tokens_per_sec"]

    paged_results = data["01_paged_attention"]["paged_results"]
    cb_results = data["02_continuous_batching"]["cb_results"]
    prefix_results = data["03_prefix_caching"]["prefix_results"]
    chunk_results = data["04_chunked_prefill"]["chunk_results"]
    cuda_graph_results = data["05_cuda_graphs"]["cuda_graph_results"]
    cg_batch_results = cuda_graph_results["batch_scaling"]
    spec_k_results = data["06_speculative_decoding"]["spec_k_results"]
    quant_results = data["07_quantization"]["quant_results"]
    lora_results = data["08_multi_lora"]["lora_results"]

    setup_plot_style_white()
    W = COLORS_WHITE
    WP = PALETTE_WHITE

    fig = plt.figure(figsize=(24, 28))
    fig.patch.set_facecolor(W["bg"])

    fig.text(
        0.5,
        0.97,
        "vLLM Feature Benchmark Dashboard",
        fontsize=26,
        fontweight="bold",
        ha="center",
        color=W["text"],
    )
    fig.text(
        0.5,
        0.955,
        "Llama-3.1-8B-Instruct  |  A100 80GB  |  vLLM 0.19.0  |  "
        f"gpu_mem={GPU_MEMORY_UTIL*100:.0f}%  |  max_len={MAX_MODEL_LEN}",
        fontsize=11,
        ha="center",
        color=W["text_light"],
    )

    fig.text(
        0.5,
        0.925,
        "Overall Throughput Ranking",
        fontsize=14,
        fontweight="bold",
        ha="center",
        color=W["text"],
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=W["cyan"],
            alpha=0.15,
            edgecolor="none",
        ),
    )

    ax_s = fig.add_axes([0.08, 0.78, 0.84, 0.13])
    ax_s.set_facecolor(W["card"])
    sorted_r = sorted(ALL_RESULTS.items(), key=lambda x: x[1], reverse=True)
    feats = [r[0] for r in sorted_r]
    tputs = [r[1] for r in sorted_r]
    bars = ax_s.barh(
        range(len(feats)), tputs, color=WP[: len(feats)], edgecolor="white", linewidth=0.8, height=0.6
    )
    mx = max(tputs)
    for bar, val in zip(bars, tputs):
        ax_s.text(
            val + mx * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f" {val:,.0f} tok/s",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=W["text"],
        )
    ax_s.set_yticks(range(len(feats)))
    ax_s.set_yticklabels(feats, fontsize=11)
    ax_s.invert_yaxis()
    ax_s.set_xlim(0, mx * 1.22)
    ax_s.set_xlabel("Throughput (tokens/second)")
    ax_s.grid(axis="x", alpha=0.3)

    fig.text(
        0.29,
        0.75,
        "1. PagedAttention",
        fontsize=13,
        fontweight="bold",
        ha="center",
        color=W["text"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=W["blue"], alpha=0.12, edgecolor="none"),
    )
    fig.text(
        0.71,
        0.75,
        "2. Continuous Batching",
        fontsize=13,
        fontweight="bold",
        ha="center",
        color=W["text"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=W["blue"], alpha=0.12, edgecolor="none"),
    )

    ax1 = fig.add_subplot(5, 4, 5)
    bs = [r["batch_size"] for r in paged_results]
    tps = [r["tokens_per_sec"] for r in paged_results]
    bars = ax1.bar(range(len(bs)), tps, color=WP[: len(bs)], edgecolor="white", width=0.7)
    ax1.set_xticks(range(len(bs)))
    ax1.set_xticklabels(bs)
    ax1.set_xlabel("Sequences")
    ax1.set_ylabel("tok/s")
    ax1.set_title("Throughput Scaling")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, tps):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(tps) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    ax2 = fig.add_subplot(5, 4, 6)
    peak_m = [r["gpu_peak_mem_gb"] for r in paged_results]
    kv_m = [r["kv_cache_estimate_gb"] for r in paged_results]
    mb = paged_results[0]["model_mem_gb"]
    x = np.arange(len(bs))
    ax2.bar(x, peak_m, 0.6, color=W["purple"], alpha=0.4, label="Reserved", edgecolor="white")
    ax2.bar(x, kv_m, 0.6, color=W["green"], alpha=0.8, label="KV Cache", edgecolor="white")
    ax2.axhline(y=mb, color=W["blue"], linestyle="--", linewidth=1.5, label=f"Model ({mb:.1f}GB)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bs)
    ax2.set_xlabel("Sequences")
    ax2.set_ylabel("GB")
    ax2.set_title("Memory Efficiency")
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(5, 4, 7)
    bs_v = [r["batch_size"] for r in cb_results]
    tps_v = [r["tokens_per_sec"] for r in cb_results]
    ax3.plot(
        bs_v,
        tps_v,
        "o-",
        color=W["blue"],
        linewidth=2,
        markersize=5,
        markeredgecolor="white",
    )
    ax3.fill_between(bs_v, tps_v, alpha=0.1, color=W["blue"])
    ax3.set_xlabel("Batch Size")
    ax3.set_ylabel("tok/s")
    ax3.set_title("Aggregate Throughput")
    ax3.set_xscale("log", base=2)
    ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(5, 4, 8)
    lats = [r["elapsed_s"] for r in cb_results]
    ax4.plot(
        bs_v,
        lats,
        "s-",
        color=W["red"],
        linewidth=2,
        markersize=5,
        markeredgecolor="white",
    )
    ax4.fill_between(bs_v, lats, alpha=0.1, color=W["red"])
    ax4.set_xlabel("Batch Size")
    ax4.set_ylabel("Seconds")
    ax4.set_title("Latency vs Batch Size")
    ax4.set_xscale("log", base=2)
    ax4.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax4.grid(True, alpha=0.3)

    fig.text(
        0.29,
        0.56,
        "3. Prefix Caching (APC)",
        fontsize=13,
        fontweight="bold",
        ha="center",
        color=W["text"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=W["green"], alpha=0.12, edgecolor="none"),
    )
    fig.text(
        0.71,
        0.56,
        "4. Chunked Prefill",
        fontsize=13,
        fontweight="bold",
        ha="center",
        color=W["text"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=W["green"], alpha=0.12, edgecolor="none"),
    )

    ax5 = fig.add_subplot(5, 4, 9)
    pc_lb = ["No Cache\nR1", "No Cache\nR2", "APC\nCold", "APC\nWarm"]
    pc_t = [
        prefix_results[k]["elapsed_s"]
        for k in ["no_cache_run1", "no_cache_run2", "cache_run1", "cache_run2"]
    ]
    pc_c = [W["red"], W["red"], W["blue"], W["green"]]
    bars = ax5.bar(pc_lb, pc_t, color=pc_c, edgecolor="white", width=0.6)
    for bar, val in zip(bars, pc_t):
        ax5.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(pc_t) * 0.02,
            f"{val:.3f}s",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    ax5.set_ylabel("Seconds")
    ax5.set_title("Latency Comparison")
    ax5.grid(axis="y", alpha=0.3)

    ax6 = fig.add_subplot(5, 4, 10)
    pc_tp = [
        prefix_results[k]["tokens_per_sec"]
        for k in ["no_cache_run1", "no_cache_run2", "cache_run1", "cache_run2"]
    ]
    bars = ax6.bar(pc_lb, pc_tp, color=pc_c, edgecolor="white", width=0.6)
    for bar, val in zip(bars, pc_tp):
        ax6.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(pc_tp) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    ax6.set_ylabel("tok/s")
    ax6.set_title("Throughput Comparison")
    ax6.grid(axis="y", alpha=0.3)

    ax7 = fig.add_subplot(5, 4, 11)
    cn = [r["config"] for r in chunk_results]
    ct = [r["tokens_per_sec"] for r in chunk_results]
    n_chunked = max(0, len(cn) - 1)
    bars = ax7.bar(
        range(len(cn)),
        ct,
        color=[W["cyan"]] + [W["blue"]] * n_chunked,
        edgecolor="white",
        width=0.6,
    )
    ax7.set_xticks(range(len(cn)))
    ax7.set_xticklabels(cn, fontsize=7)
    ax7.set_ylabel("tok/s")
    ax7.set_title("Throughput by Chunk")
    ax7.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, ct):
        ax7.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ct) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    ax8 = fig.add_subplot(5, 4, 12)
    cl = [r["elapsed_s"] for r in chunk_results]
    bars = ax8.bar(
        range(len(cn)),
        cl,
        color=[W["cyan"]] + [W["red"]] * n_chunked,
        edgecolor="white",
        width=0.6,
    )
    ax8.set_xticks(range(len(cn)))
    ax8.set_xticklabels(cn, fontsize=7)
    ax8.set_ylabel("Seconds")
    ax8.set_title("Latency by Chunk")
    ax8.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, cl):
        ax8.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(cl) * 0.02,
            f"{val:.2f}s",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    fig.text(
        0.29,
        0.37,
        "5. CUDA Graphs",
        fontsize=13,
        fontweight="bold",
        ha="center",
        color=W["text"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=W["yellow"], alpha=0.15, edgecolor="none"),
    )
    fig.text(
        0.71,
        0.37,
        "6. Speculative Decoding",
        fontsize=13,
        fontweight="bold",
        ha="center",
        color=W["text"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=W["yellow"], alpha=0.15, edgecolor="none"),
    )

    ax9 = fig.add_subplot(5, 4, 13)
    cg_lb = ["Eager", "CUDA Graphs"]
    cg_tp = [
        cuda_graph_results["eager"]["tokens_per_sec"],
        cuda_graph_results["cuda_graph"]["tokens_per_sec"],
    ]
    cg_c = [W["red"], W["green"]]
    bars = ax9.bar(cg_lb, cg_tp, color=cg_c, edgecolor="white", width=0.5)
    for bar, val in zip(bars, cg_tp):
        ax9.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(cg_tp) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    sp_cg = cg_tp[1] / max(cg_tp[0], 1)
    ax9.set_ylabel("tok/s")
    ax9.set_title(f"Speedup: {sp_cg:.2f}x")
    ax9.grid(axis="y", alpha=0.3)

    ax10 = fig.add_subplot(5, 4, 14)
    cg_bs = [r["batch_size"] for r in cg_batch_results]
    cg_bt = [r["tokens_per_sec"] for r in cg_batch_results]
    ax10.plot(
        cg_bs,
        cg_bt,
        "o-",
        color=W["green"],
        linewidth=2,
        markersize=5,
        markeredgecolor="white",
        label="CUDA Graphs",
    )
    ax10.fill_between(cg_bs, cg_bt, alpha=0.1, color=W["green"])
    ax10.axhline(
        y=cg_tp[0],
        color=W["red"],
        linestyle="--",
        linewidth=1.5,
        label=f"Eager ({cg_tp[0]:.0f})",
    )
    ax10.set_xlabel("Batch Size")
    ax10.set_ylabel("tok/s")
    ax10.set_title("Batch Scaling")
    ax10.legend(fontsize=7)
    ax10.grid(True, alpha=0.3)

    ax11 = fig.add_subplot(5, 4, 15)
    k_v = [r["k"] for r in spec_k_results]
    sd_t = [r["tokens_per_sec"] for r in spec_k_results]
    k_lb = [f"k={k}" if k > 0 else "Base" for k in k_v]
    k_c = [W["red"] if k == 0 else W["green"] for k in k_v]
    bars = ax11.bar(k_lb, sd_t, color=k_c, edgecolor="white", width=0.5)
    for bar, val in zip(bars, sd_t):
        ax11.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sd_t) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax11.set_ylabel("tok/s")
    ax11.set_title("Throughput by k")
    ax11.grid(axis="y", alpha=0.3)

    ax12 = fig.add_subplot(5, 4, 16)
    base_sd = spec_k_results[0]["tokens_per_sec"]
    sd_sp = [r["tokens_per_sec"] / base_sd for r in spec_k_results]
    bars = ax12.bar(k_lb, sd_sp, color=k_c, edgecolor="white", width=0.5)
    ax12.axhline(y=1.0, color=W["text_light"], linestyle="--", linewidth=1, alpha=0.5)
    for bar, val in zip(bars, sd_sp):
        ax12.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax12.set_ylabel("Speedup")
    ax12.set_title("vs Baseline")
    ax12.grid(axis="y", alpha=0.3)

    fig.text(
        0.29,
        0.18,
        "7. Quantization",
        fontsize=13,
        fontweight="bold",
        ha="center",
        color=W["text"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=W["purple"], alpha=0.1, edgecolor="none"),
    )
    fig.text(
        0.71,
        0.18,
        "8. Multi-LoRA Serving",
        fontsize=13,
        fontweight="bold",
        ha="center",
        color=W["text"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=W["purple"], alpha=0.1, edgecolor="none"),
    )

    ax13 = fig.add_subplot(5, 4, 17)
    ok_q = [r for r in quant_results if r["status"] == "success"]
    qn = [r["name"] for r in ok_q]
    qt = [r["tokens_per_sec"] for r in ok_q]
    qm = [r["model_weight_gb"] for r in ok_q]
    bars = ax13.bar(range(len(qn)), qt, color=WP[: len(qn)], edgecolor="white", width=0.6)
    ax13.set_xticks(range(len(qn)))
    ax13.set_xticklabels(qn, fontsize=7)
    ax13.set_ylabel("tok/s")
    ax13.set_title("Throughput")
    ax13.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, qt):
        ax13.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(qt) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    ax14 = fig.add_subplot(5, 4, 18)
    bars = ax14.bar(range(len(qn)), qm, color=WP[: len(qn)], edgecolor="white", width=0.6)
    ax14.set_xticks(range(len(qn)))
    ax14.set_xticklabels(qn, fontsize=7)
    ax14.set_ylabel("GB")
    ax14.set_title("Model Weight Size")
    ax14.grid(axis="y", alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, qm)):
        ax14.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(qm) * 0.02,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
        if i > 0:
            sav = (1 - val / qm[0]) * 100
            ax14.text(i, val / 2, f"-{sav:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold", color=W["red"])

    ax15 = fig.add_subplot(5, 4, 19)
    lr_lb = ["Base", "LoRA\nEnabled", "With\nAdapter"]
    lr_tp = [
        lora_results["no_lora"]["tokens_per_sec"],
        lora_results["lora_infra"]["tokens_per_sec"],
        lora_results["with_adapter"]["tokens_per_sec"],
    ]
    lr_c = [W["blue"], W["yellow"], W["green"]]
    bars = ax15.bar(lr_lb, lr_tp, color=lr_c, edgecolor="white", width=0.5)
    for bar, val in zip(bars, lr_tp):
        ax15.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(lr_tp) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax15.set_ylabel("tok/s")
    ax15.set_title("LoRA Overhead")
    ax15.grid(axis="y", alpha=0.3)

    ax16 = fig.add_subplot(5, 4, 20)
    lr_lt = [
        lora_results["no_lora"]["elapsed_s"],
        lora_results["lora_infra"]["elapsed_s"],
        lora_results["with_adapter"]["elapsed_s"],
    ]
    bars = ax16.bar(lr_lb, lr_lt, color=lr_c, edgecolor="white", width=0.5)
    for bar, val in zip(bars, lr_lt):
        ax16.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(lr_lt) * 0.02,
            f"{val:.3f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax16.set_ylabel("Seconds")
    ax16.set_title("Latency")
    ax16.grid(axis="y", alpha=0.3)

    plt.subplots_adjust(hspace=0.55, wspace=0.35, top=0.93, bottom=0.03, left=0.06, right=0.97)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "vllm_benchmark_linkedin.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=W["bg"])
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Model: {MODEL_ID}")


if __name__ == "__main__":
    generate_dashboard()
