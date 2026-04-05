# -*- coding: utf-8 -*-
"""Run all 8 benchmarks sequentially."""
import argparse

from benchmarks import (
    b01_paged_attention,
    b02_continuous_batching,
    b03_prefix_caching,
    b04_chunked_prefill,
    b05_cuda_graphs,
    b06_speculative_decoding,
    b07_quantization,
    b08_multi_lora,
)
import config
from config import ensure_hf_login

ALL_BENCHMARKS = [
    ("01_paged_attention", b01_paged_attention),
    ("02_continuous_batching", b02_continuous_batching),
    ("03_prefix_caching", b03_prefix_caching),
    ("04_chunked_prefill", b04_chunked_prefill),
    ("05_cuda_graphs", b05_cuda_graphs),
    ("06_speculative_decoding", b06_speculative_decoding),
    ("07_quantization", b07_quantization),
    ("08_multi_lora", b08_multi_lora),
]

CANONICAL_IDS = {name for name, _ in ALL_BENCHMARKS}

SKIP_ALIASES = {
    "paged_attention": "01_paged_attention",
    "continuous_batching": "02_continuous_batching",
    "prefix_caching": "03_prefix_caching",
    "chunked_prefill": "04_chunked_prefill",
    "cuda_graphs": "05_cuda_graphs",
    "speculative_decoding": "06_speculative_decoding",
    "quantization": "07_quantization",
    "multi_lora": "08_multi_lora",
}


def _parse_skip(s: str) -> set:
    out = set()
    for part in s.split(","):
        part = part.strip().lower().replace("-", "_")
        if not part:
            continue
        if part in SKIP_ALIASES:
            out.add(SKIP_ALIASES[part])
        elif part.isdigit():
            n = int(part)
            if 1 <= n <= 8:
                out.add(ALL_BENCHMARKS[n - 1][0])
        elif part in CANONICAL_IDS:
            out.add(part)
    return out


def _apply_global_args(args):
    if args.model:
        config.MODEL_ID = args.model
        config.refresh_quant_configs_after_model_change()
    if args.gpu_mem is not None:
        config.GPU_MEMORY_UTIL = args.gpu_mem
    if args.max_len is not None:
        config.MAX_MODEL_LEN = args.max_len
    if args.warmup is not None:
        config.WARMUP_RUNS = args.warmup
    if args.runs is not None:
        config.BENCH_RUNS = args.runs
    if args.max_tokens is not None:
        config.apply_max_tokens_override(args.max_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all vLLM feature benchmarks.")
    parser.add_argument("--model", type=str, help="Override config.MODEL_ID")
    parser.add_argument("--gpu-mem", type=float, dest="gpu_mem", help="GPU memory utilization 0–1")
    parser.add_argument("--max-len", type=int, dest="max_len", help="max_model_len")
    parser.add_argument("--warmup", type=int, help="WARMUP_RUNS")
    parser.add_argument("--runs", type=int, help="BENCH_RUNS")
    parser.add_argument("--max-tokens", type=int, dest="max_tokens", help="Override all max_tokens caps")
    parser.add_argument(
        "--skip",
        type=str,
        help="Comma-separated features to skip (name or 1–8), e.g. speculative_decoding",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots and dashboard from saved JSON only",
    )
    args = parser.parse_args()

    _apply_global_args(args)
    skip = _parse_skip(args.skip) if args.skip else set()

    if args.plot_only:
        for name, module in ALL_BENCHMARKS:
            if name in skip:
                print(f"  Skipping (plot): {name}")
                continue
            print(f"\n{'='*60}\n  Plot only: {name}\n{'='*60}")
            try:
                module.plot_results(None)
            except Exception as e:
                print(f"  FAILED: {e}")
        try:
            from dashboard import generate_dashboard

            generate_dashboard()
        except Exception as e:
            print(f"  Dashboard skipped: {e}")
    else:
        ensure_hf_login()
        for name, module in ALL_BENCHMARKS:
            if name in skip:
                print(f"  Skipping: {name}")
                continue
            print(f"\n{'='*60}")
            print(f"  Running: {name}")
            print(f"{'='*60}")
            try:
                results = module.run_benchmark()
                module.plot_results(results)
            except Exception as e:
                print(f"  FAILED: {e}")
                continue

        from dashboard import generate_dashboard

        try:
            generate_dashboard()
        except Exception as e:
            print(f"  Dashboard skipped: {e}")
