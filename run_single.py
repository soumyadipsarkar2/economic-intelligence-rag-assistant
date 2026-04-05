# -*- coding: utf-8 -*-
"""Run a single benchmark by name or number."""
import argparse
import importlib

import config
from config import ensure_hf_login

FEATURE_MAP = {
    "paged_attention": "benchmarks.b01_paged_attention",
    "continuous_batching": "benchmarks.b02_continuous_batching",
    "prefix_caching": "benchmarks.b03_prefix_caching",
    "chunked_prefill": "benchmarks.b04_chunked_prefill",
    "cuda_graphs": "benchmarks.b05_cuda_graphs",
    "speculative_decoding": "benchmarks.b06_speculative_decoding",
    "quantization": "benchmarks.b07_quantization",
    "multi_lora": "benchmarks.b08_multi_lora",
}

MODULES_BY_INDEX = [
    "benchmarks.b01_paged_attention",
    "benchmarks.b02_continuous_batching",
    "benchmarks.b03_prefix_caching",
    "benchmarks.b04_chunked_prefill",
    "benchmarks.b05_cuda_graphs",
    "benchmarks.b06_speculative_decoding",
    "benchmarks.b07_quantization",
    "benchmarks.b08_multi_lora",
]


def _resolve_feature_module(feat: str) -> str:
    s = str(feat).strip().lower().replace("-", "_")
    if s.isdigit():
        n = int(s)
        if 1 <= n <= 8:
            return MODULES_BY_INDEX[n - 1]
        raise SystemExit("feature number must be 1–8")
    if s in FEATURE_MAP:
        return FEATURE_MAP[s]
    raise SystemExit(
        f"Unknown feature {feat!r}. Use 1–8 or one of: {', '.join(sorted(FEATURE_MAP))}"
    )


def _apply_overrides(args):
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
    if args.batch_sizes:
        config.BATCH_SIZES = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    if args.cb_batch_sizes:
        config.CB_BATCH_SIZES = [int(x.strip()) for x in args.cb_batch_sizes.split(",") if x.strip()]
    if args.chunk_sizes:
        config.CHUNK_SIZES = [int(x.strip()) for x in args.chunk_sizes.split(",") if x.strip()]
    if args.spec_tokens:
        config.SPEC_K_VALUES = [int(x.strip()) for x in args.spec_tokens.split(",") if x.strip()]
    if args.draft_model:
        config.DRAFT_MODEL = args.draft_model
    if args.max_loras is not None:
        config.MAX_LORAS = args.max_loras
    if args.lora_rank is not None:
        config.LORA_RANK = args.lora_rank
    if args.quant_methods:
        config.filter_quant_configs(args.quant_methods)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one vLLM feature benchmark.")
    parser.add_argument(
        "--feature",
        required=True,
        help="Feature: 1–8 or name (e.g. paged_attention, cuda_graphs)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate plot from saved results",
    )
    parser.add_argument("--model", type=str, help="Override config.MODEL_ID")
    parser.add_argument("--gpu-mem", type=float, dest="gpu_mem", help="GPU memory utilization 0–1")
    parser.add_argument("--max-len", type=int, dest="max_len", help="max_model_len")
    parser.add_argument("--warmup", type=int, help="WARMUP_RUNS")
    parser.add_argument("--runs", type=int, help="BENCH_RUNS")
    parser.add_argument("--max-tokens", type=int, dest="max_tokens", help="Override all max_tokens caps")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        help="Comma-separated batch sizes for PagedAttention (e.g. 1,8,32)",
    )
    parser.add_argument("--cb-batch-sizes", type=str, dest="cb_batch_sizes", help="Continuous batching sizes")
    parser.add_argument("--chunk-sizes", type=str, dest="chunk_sizes", help="Chunked prefill token sizes")
    parser.add_argument("--spec-tokens", type=str, dest="spec_tokens", help="Speculative k values, e.g. 3,5,7")
    parser.add_argument("--draft-model", type=str, dest="draft_model", help="Draft model for speculative decoding")
    parser.add_argument("--max-loras", type=int, dest="max_loras", help="Multi-LoRA max_loras")
    parser.add_argument("--lora-rank", type=int, dest="lora_rank", help="Multi-LoRA max_lora_rank")
    parser.add_argument(
        "--quant-methods",
        type=str,
        dest="quant_methods",
        help="Comma-separated: bf16,fp8,awq,gptq",
    )
    args = parser.parse_args()

    _apply_overrides(args)

    mod_name = _resolve_feature_module(args.feature)
    module = importlib.import_module(mod_name)

    if not args.plot_only:
        ensure_hf_login()

    if args.plot_only:
        module.plot_results(None)
    else:
        results = module.run_benchmark()
        module.plot_results(results)
