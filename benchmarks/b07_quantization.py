# -*- coding: utf-8 -*-
"""
Feature 7: Quantization
FP16/BF16 baseline vs FP8 and 4-bit quantized checkpoints.
"""
import os

import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams

from config import *
from utils import benchmark, cleanup_engine, make_test_prompts, save_results


def run_benchmark():
    """Run the benchmark. Return results dict."""
    print("=" * 60)
    print("  Quantization Comparison")
    print("=" * 60)

    sp = SamplingParams(max_tokens=QUANTIZATION_MAX_TOKENS, temperature=0.7)
    quant_prompts = make_test_prompts(16)

    quant_results = []

    quant_configs = QUANT_CONFIGS
    for name, model_id, extra_kw in quant_configs:
        print(f"\n--- {name.replace(chr(10), ' ')} ({model_id.split('/')[-1]}) ---")
        try:
            engine = LLM(
                model=model_id,
                dtype=DTYPE,
                gpu_memory_utilization=GPU_MEMORY_UTIL,
                max_model_len=MAX_MODEL_LEN,
                trust_remote_code=True,
                **extra_kw,
            )

            res = benchmark(engine, quant_prompts, sp, label=name.replace("\n", " "))
            res["name"] = name
            res["model_weight_gb"] = KNOWN_WEIGHT_SIZES.get(name, 0)
            res["status"] = "success"
            quant_results.append(res)
            print(f"   Model Weights: ~{res['model_weight_gb']:.1f} GB")
            cleanup_engine(engine)
        except Exception as e:
            print(f"   Skipped: {e}")
            quant_results.append(
                {
                    "name": name,
                    "tokens_per_sec": 0,
                    "elapsed_s": 0,
                    "model_weight_gb": 0,
                    "status": "failed",
                }
            )
            cleanup_engine()

    successful = [r for r in quant_results if r["status"] == "success"]
    if successful:
        best_q = max(successful, key=lambda x: x["tokens_per_sec"])
        summary_tps = best_q["tokens_per_sec"]
    else:
        summary_tps = 0.0

    results = {
        "_summary": {"feature": "Quantization (best)", "tokens_per_sec": summary_tps},
        "quant_results": quant_results,
    }

    print("\nQuantization benchmark complete")

    save_results("07_quantization", results)
    return results


def plot_results(results=None):
    """Generate and save the plot."""
    if results is None:
        from utils import load_results

        results = load_results("07_quantization")

    quant_results = results["quant_results"]
    ok_results = [r for r in quant_results if r["status"] == "success"]

    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Quantization — Precision vs. Performance",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text"],
        y=1.02,
    )

    names = [r["name"] for r in ok_results]
    tps_vals = [r["tokens_per_sec"] for r in ok_results]
    mem_weights = [r["model_weight_gb"] for r in ok_results]
    q_colors = PALETTE[: len(ok_results)]

    bars = ax1.bar(
        range(len(names)),
        tps_vals,
        color=q_colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.6,
    )
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("Throughput (tokens/s)")
    ax1.set_title("Throughput Comparison")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, tps_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(tps_vals) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=COLORS["text"],
        )

    bars = ax2.bar(
        range(len(names)),
        mem_weights,
        color=q_colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.6,
    )
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("Model Weight Size (GB)")
    ax2.set_title("Memory Footprint")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, mem_weights):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mem_weights) * 0.02,
            f"{val:.1f} GB",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=COLORS["text"],
        )

    if len(mem_weights) >= 2:
        baseline_mem = mem_weights[0]
        for i in range(1, len(mem_weights)):
            savings = (1 - mem_weights[i] / baseline_mem) * 100
            ax2.text(
                i,
                mem_weights[i] / 2,
                f"-{savings:.0f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "plot_07_quantization.png")
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
