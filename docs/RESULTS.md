# Sample benchmark results

These runs used **meta-llama/Llama-3.1-8B-Instruct** on an **NVIDIA A100 80GB** with **vLLM ~0.19.x**, **gpu_memory_utilization ≈ 0.85**, and **max_model_len = 8192**. Your numbers will differ with other models, GPUs, and settings. The same figures live in `plots/` (committed samples alongside `results/*.json`).

## Overall throughput ranking

Peak throughput (tokens/s) recorded for each feature benchmark:


| Rank | Feature                                         | Tokens/s |
| ---- | ----------------------------------------------- | -------- |
| 1    | Continuous batching                             | 7,058    |
| 2    | PagedAttention                                  | 4,088    |
| 3    | Quantization (best: AWQ 4-bit)                  | 2,656    |
| 4    | CUDA graphs                                     | 1,407    |
| 5    | Speculative decoding (baseline, no speculation) | 729      |
| 6    | Prefix caching (APC warm)                       | 719      |
| 7    | Multi-LoRA (infra enabled)                      | 715      |
| 8    | Chunked prefill (best config)                   | 713      |


## Combined dashboard

vLLM feature benchmark dashboard

---

## 1. PagedAttention + block manager

Throughput scales with concurrent sequences; KV usage grows predictably; block table shows per-request allocation.


| Concurrent sequences | Throughput (tok/s) |
| -------------------- | ------------------ |
| 1                    | 90                 |
| 4                    | 346                |
| 8                    | 700                |
| 16                   | 1,367              |
| 32                   | 2,446              |
| 64                   | 4,088              |


PagedAttention

---

## 2. Continuous batching

Aggregate throughput rises with batch size; wall time stays relatively flat until higher concurrency.


| Batch size | Throughput (tok/s) | Notes              |
| ---------- | ------------------ | ------------------ |
| 1          | ~100               |                    |
| 128        | ~7,000             | Peak in this sweep |


Continuous batching

---

## 3. Prefix caching (radix / APC)

Shared long prefix: APC reduces latency and raises throughput versus no cache.


| Scenario         | Latency (s) | Throughput (tok/s) |
| ---------------- | ----------- | ------------------ |
| No cache (run 1) | 1.774       | 676                |
| No cache (run 2) | 1.774       | 677                |
| APC (cold)       | 1.670       | 718                |
| APC (warm)       | 1.669       | 719                |


Prefix caching

---

## 4. Chunked prefill

For this workload, throughput and latency were nearly flat across no chunking and chunk sizes 512–4096.


| Config                | Throughput (tok/s) | Latency (s) |
| --------------------- | ------------------ | ----------- |
| No chunking (default) | 713                | 1.12        |
| Chunk 512–4096        | ~711–713           | ~1.12       |


Chunked prefill

---

## 5. CUDA graphs

Eager vs CUDA graph mode on the same prompts; graphs roughly double throughput in this configuration.


| Mode                   | Throughput (tok/s) |
| ---------------------- | ------------------ |
| Eager (no CUDA graphs) | 619                |
| CUDA graphs enabled    | 1,407              |


Speedup vs eager: **~2.27×**. Batch scaling with graphs reaches about **2,500 tok/s** at batch 32 in this run.

CUDA graphs

---

## 6. Speculative decoding

Draft-model speculation with tested *k* values; under this setup, higher *k* reduced throughput vs baseline.


| Setting  | Throughput (tok/s) | vs baseline |
| -------- | ------------------ | ----------- |
| Baseline | 729                | 1.00×       |
| k = 3    | 596                | ~0.82×      |
| k = 5    | 466                | ~0.64×      |


Speculative decoding

---

## 7. Quantization

BF16 baseline vs FP8 and 4-bit checkpoints (AWQ, GPTQ).


| Format          | Throughput (tok/s) | Weight size (GB) |
| --------------- | ------------------ | ---------------- |
| BF16 (baseline) | 1,405              | 16.1             |
| FP8             | 2,091              | 8.0              |
| AWQ (4-bit)     | 2,656              | 4.0              |
| GPTQ (4-bit)    | 2,380              | 4.0              |


Quantization

---

## 8. Multi-LoRA

LoRA infrastructure and adapter path showed small overhead vs base model alone.


| Scenario           | Throughput (tok/s) | Latency (s) |
| ------------------ | ------------------ | ----------- |
| Base (no LoRA)     | 718                | 1.426       |
| LoRA infra enabled | 715                | 1.432       |
| With LoRA adapter  | 714                | 1.433       |


Multi-LoRA