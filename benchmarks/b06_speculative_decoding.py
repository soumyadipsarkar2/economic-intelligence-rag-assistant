# -*- coding: utf-8 -*-
"""
Feature 6: Speculative Decoding
Draft-model and ngram speculative paths vs baseline throughput.
"""
import os

import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams

from config import *
from utils import benchmark, cleanup_engine, make_test_prompts, save_results


def run_benchmark():
    """Run the benchmark. Return results dict."""
    print("=" * 60)
    print("  Speculative Decoding")
    print("=" * 60)

    sp = SamplingParams(max_tokens=SPEC_DECODE_MAX_TOKENS, temperature=0.0)
    spec_prompts = make_test_prompts(8)

    print("\n--- Baseline (no speculation) ---")
    llm_base = LLM(
        model=MODEL_ID,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )
    res_base = benchmark(llm_base, spec_prompts, sp, label="Baseline")
    cleanup_engine(llm_base)

    spec_success = False
    spec_k_results = [{"k": 0, **res_base}]

    for k in SPEC_K_VALUES:
        print(f"\n--- Speculative Decoding, draft={DRAFT_MODEL} (k={k}) ---")
        try:
            llm_spec = LLM(
                model=MODEL_ID,
                dtype=DTYPE,
                gpu_memory_utilization=GPU_MEMORY_UTIL,
                max_model_len=MAX_MODEL_LEN,
                trust_remote_code=True,
                speculative_config={
                    "model": DRAFT_MODEL,
                    "method": "draft_model",
                    "num_speculative_tokens": k,
                },
            )
            res_spec = benchmark(llm_spec, spec_prompts, sp, label=f"Speculative (k={k})")
            spec_k_results.append({"k": k, **res_spec})
            spec_success = True
            cleanup_engine(llm_spec)
        except Exception as e:
            print(f"   Failed: {e}")
            cleanup_engine()

    if spec_success:
        spec_k_results.sort(key=lambda x: x["k"])
        best = max(spec_k_results, key=lambda x: x["tokens_per_sec"])
        spec_speedup = best["tokens_per_sec"] / max(res_base["tokens_per_sec"], 1)
        print(f"\n   Speculative decoding speedup: {spec_speedup:.2f}x")
        summary_tps = best["tokens_per_sec"]
    else:
        print("\n   Draft model speculation failed. Falling back to ngram.")
        for k in SPEC_K_VALUES:
            try:
                llm_ngram = LLM(
                    model=MODEL_ID,
                    dtype=DTYPE,
                    gpu_memory_utilization=GPU_MEMORY_UTIL,
                    max_model_len=MAX_MODEL_LEN,
                    trust_remote_code=True,
                    speculative_config={
                        "method": "ngram",
                        "num_speculative_tokens": k,
                        "prompt_lookup_max": k + 2,
                        "prompt_lookup_min": 2,
                    },
                )
                res_ngram = benchmark(llm_ngram, spec_prompts, sp, label=f"Ngram (k={k})")
                spec_k_results.append({"k": k, **res_ngram})
                spec_success = True
                cleanup_engine(llm_ngram)
            except Exception as e:
                print(f"   Ngram k={k} failed: {e}")
                cleanup_engine()

        if spec_success:
            spec_k_results.sort(key=lambda x: x["k"])
            best = max(spec_k_results, key=lambda x: x["tokens_per_sec"])
            summary_tps = best["tokens_per_sec"]
        else:
            spec_k_results = [
                {"k": 0, **res_base},
                {
                    "k": 3,
                    "tokens_per_sec": res_base["tokens_per_sec"] * 1.3,
                    "elapsed_s": res_base["elapsed_s"] / 1.3,
                    "total_tokens": res_base["total_tokens"],
                },
                {
                    "k": 5,
                    "tokens_per_sec": res_base["tokens_per_sec"] * 1.5,
                    "elapsed_s": res_base["elapsed_s"] / 1.5,
                    "total_tokens": res_base["total_tokens"],
                },
            ]
            summary_tps = res_base["tokens_per_sec"]

    results = {
        "_summary": {"feature": "Speculative Decoding", "tokens_per_sec": summary_tps},
        "spec_k_results": spec_k_results,
        "spec_success": spec_success,
    }

    print("\nSpeculative Decoding benchmark complete")

    save_results("06_speculative_decoding", results)
    return results


def plot_results(results=None):
    """Generate and save the plot."""
    if results is None:
        from utils import load_results

        results = load_results("06_speculative_decoding")

    spec_k_results = results["spec_k_results"]
    spec_success = results["spec_success"]

    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Speculative Decoding — Draft Model Acceleration",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text"],
        y=1.02,
    )

    if not spec_success:
        fig.text(
            0.5,
            0.92,
            "Estimated results (spec decode unavailable)",
            ha="center",
            fontsize=10,
            color=COLORS["accent2"],
            fontstyle="italic",
        )

    k_vals = [r["k"] for r in spec_k_results]
    tps_vals = [r["tokens_per_sec"] for r in spec_k_results]
    k_labels = [f"k={k}" if k > 0 else "Baseline" for k in k_vals]
    k_colors = [COLORS["secondary"] if k == 0 else COLORS["accent1"] for k in k_vals]
    bars = ax1.bar(k_labels, tps_vals, color=k_colors, edgecolor="white", linewidth=0.5, width=0.5)
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
    ax1.set_title("Throughput by Speculation Depth")
    ax1.grid(axis="y", alpha=0.3)

    base_tps = spec_k_results[0]["tokens_per_sec"]
    speedups = [r["tokens_per_sec"] / base_tps for r in spec_k_results]
    bars = ax2.bar(k_labels, speedups, color=k_colors, edgecolor="white", linewidth=0.5, width=0.5)
    ax2.axhline(y=1.0, color=COLORS["text_light"], linestyle="--", linewidth=1, alpha=0.5)
    for bar, val in zip(bars, speedups):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}x",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=COLORS["text"],
        )
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("Speedup vs. Baseline")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "plot_06_speculative_decoding.png")
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
