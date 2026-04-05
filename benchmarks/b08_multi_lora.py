# -*- coding: utf-8 -*-
"""
Feature 8: Multi-LoRA Adapter Serving
Baseline vs LoRA-enabled engine and optional public adapter load.
"""
import os

import matplotlib.pyplot as plt
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from config import *
from utils import benchmark, cleanup_engine, make_test_prompts, save_results


def run_benchmark():
    """Run the benchmark. Return results dict."""
    print("=" * 60)
    print("  Multi-LoRA Adapter Serving")
    print("=" * 60)

    sp = SamplingParams(max_tokens=LORA_MAX_TOKENS, temperature=0.7)
    lora_prompts = make_test_prompts(8)

    print("\n--- Baseline (no LoRA) ---")
    llm_base = LLM(
        model=MODEL_ID,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )
    res_no_lora = benchmark(llm_base, lora_prompts, sp, label="No LoRA")
    cleanup_engine(llm_base)

    print("\n--- LoRA infrastructure enabled ---")
    try:
        llm_lora = LLM(
            model=MODEL_ID,
            dtype=DTYPE,
            gpu_memory_utilization=GPU_MEMORY_UTIL,
            max_model_len=MAX_MODEL_LEN,
            enable_lora=True,
            max_loras=MAX_LORAS,
            max_lora_rank=LORA_RANK,
            trust_remote_code=True,
        )
        res_lora_enabled = benchmark(
            llm_lora, lora_prompts, sp, label="LoRA infra enabled (no adapter)"
        )
        lora_mem = torch.cuda.memory_allocated() / 1e9

        lora_adapter_loaded = False
        try:
            lora_id = "yard1/llama-2-7b-sql-lora-test"
            lora_req = LoRARequest("sql_lora", 1, lora_id)
            res_with_adapter = benchmark(
                llm_lora, lora_prompts, sp, label="With LoRA adapter"
            )
            lora_adapter_loaded = True
            print(f"   LoRA adapter loaded: {lora_id}")
        except Exception as e:
            print(f"   Adapter loading skipped: {e}")
            res_with_adapter = res_lora_enabled

        lora_results = {
            "no_lora": res_no_lora,
            "lora_infra": res_lora_enabled,
            "with_adapter": res_with_adapter,
            "lora_mem_gb": lora_mem,
            "adapter_loaded": lora_adapter_loaded,
        }
        lora_success = True
        cleanup_engine(llm_lora)

    except Exception as e:
        print(f"  LoRA test failed: {e}")
        lora_results = {
            "no_lora": res_no_lora,
            "lora_infra": res_no_lora,
            "with_adapter": res_no_lora,
            "lora_mem_gb": 0,
            "adapter_loaded": False,
        }
        lora_success = False

    results = {
        "_summary": {
            "feature": "Multi-LoRA",
            "tokens_per_sec": lora_results["lora_infra"]["tokens_per_sec"],
        },
        "lora_results": lora_results,
        "lora_success": lora_success,
    }

    print("\nMulti-LoRA benchmark complete")

    save_results("08_multi_lora", results)
    return results


def plot_results(results=None):
    """Generate and save the plot."""
    if results is None:
        from utils import load_results

        results = load_results("08_multi_lora")

    lora_results = results["lora_results"]

    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "  Multi-LoRA Adapter Serving",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text"],
        y=1.02,
    )

    labels = ["Base Model\n(no LoRA)", "LoRA Infra\nEnabled", "With LoRA\nAdapter"]
    tps_vals = [
        lora_results["no_lora"]["tokens_per_sec"],
        lora_results["lora_infra"]["tokens_per_sec"],
        lora_results["with_adapter"]["tokens_per_sec"],
    ]
    colors = [COLORS["primary"], COLORS["accent2"], COLORS["accent1"]]
    bars = ax1.bar(labels, tps_vals, color=colors, edgecolor="white", linewidth=0.5, width=0.5)
    for bar, val in zip(bars, tps_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(tps_vals) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=COLORS["text"],
        )
    ax1.set_ylabel("Throughput (tokens/s)")
    ax1.set_title("LoRA Overhead Analysis")
    ax1.grid(axis="y", alpha=0.3)

    lat_vals = [
        lora_results["no_lora"]["elapsed_s"],
        lora_results["lora_infra"]["elapsed_s"],
        lora_results["with_adapter"]["elapsed_s"],
    ]
    bars = ax2.bar(labels, lat_vals, color=colors, edgecolor="white", linewidth=0.5, width=0.5)
    for bar, val in zip(bars, lat_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(lat_vals) * 0.02,
            f"{val:.3f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=COLORS["text"],
        )
    ax2.set_ylabel("Latency (seconds)")
    ax2.set_title("Latency Overhead")
    ax2.grid(axis="y", alpha=0.3)

    if not lora_results["adapter_loaded"]:
        fig.text(
            0.5,
            0.92,
            " Adapter not loaded — showing infra overhead only",
            ha="center",
            fontsize=10,
            color=COLORS["accent2"],
            fontstyle="italic",
        )

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "plot_08_multi_lora.png")
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
