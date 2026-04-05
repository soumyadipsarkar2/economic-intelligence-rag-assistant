# -*- coding: utf-8 -*-
"""Timing, engine lifecycle, and results I/O."""
import gc
import json
import multiprocessing
import os
import time

import numpy as np
import torch

import config as _config

from config import RESULTS_DIR


def to_jsonable(obj):
    """Recursively convert numpy types and other values for JSON."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results(name, data):
    """Save dict as JSON under results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2)
    print(f"   Results saved: {path}")


def load_results(name):
    """Load JSON from results/ (name without .json)."""
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cleanup_engine(engine=None):
    """Destroy a vLLM engine, kill child processes, and free GPU memory."""
    if engine is not None:
        try:
            engine.llm_engine.shutdown()
        except Exception:
            pass
        del engine

    children = multiprocessing.active_children()
    for child in children:
        try:
            child.terminate()
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
                child.join(timeout=3)
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()

    time.sleep(3)
    print(
        f"   Cleanup done. GPU reserved: {torch.cuda.memory_reserved()/1e9:.1f} GB"
    )


def timed_generate(engine, prompts, sampling_params, label=""):
    """
    Run generation and return (outputs, elapsed_seconds).
    Synchronizes CUDA for accurate timing.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(prompts, sampling_params)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if label:
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(
            f"   {label}: {elapsed:.3f}s | {total_tokens} tokens | "
            f"{total_tokens/elapsed:.1f} tok/s"
        )
    return outputs, elapsed


def benchmark(
    engine,
    prompts,
    sampling_params,
    label="",
    warmup=None,
    runs=None,
):
    """
    Benchmark with warmup + multiple runs, return median metrics.
    Returns dict with: elapsed_s, total_tokens, tokens_per_sec
    """
    if warmup is None:
        warmup = _config.WARMUP_RUNS
    if runs is None:
        runs = _config.BENCH_RUNS
    for _ in range(warmup):
        engine.generate(prompts, sampling_params)

    times = []
    all_tok = []
    for _ in range(runs):
        outputs, elapsed = timed_generate(engine, prompts, sampling_params)
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        times.append(elapsed)
        all_tok.append(total_tokens)

    med_time = float(np.median(times))
    med_tok = int(np.median(all_tok))
    tps = med_tok / med_time if med_time > 0 else 0

    if label:
        print(
            f"   {label}: {med_time:.3f}s median | "
            f"{med_tok} tokens | {tps:.1f} tok/s"
        )

    return {
        "elapsed_s": med_time,
        "total_tokens": med_tok,
        "tokens_per_sec": tps,
    }


def make_test_prompts(n, prefix="", length="short"):
    """Generate diverse test prompts."""
    topics = [
        "Explain how neural networks learn through backpropagation.",
        "What are the main differences between TCP and UDP protocols?",
        "Describe the water cycle and its importance to ecosystems.",
        "How does a compiler convert source code into machine code?",
        "Explain the concept of supply and demand in economics.",
        "What is the theory of general relativity in simple terms?",
        "Describe the process of photosynthesis step by step.",
        "How do vaccines work to protect against diseases?",
        "Explain the difference between classical and quantum computing.",
        "What causes tides and how are they predicted?",
        "Describe how blockchain technology ensures data integrity.",
        "What is CRISPR and how is it used in gene editing?",
        "Explain the principles behind magnetic resonance imaging (MRI).",
        "How do operating systems manage memory allocation?",
        "Describe the structure and function of DNA.",
        "What is the greenhouse effect and its role in climate change?",
    ]
    prompts = []
    for i in range(n):
        topic = topics[i % len(topics)]
        prompts.append(f"{prefix}{topic}")
    return prompts
