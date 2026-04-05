# vLLM features benchmarked

Each section matches one benchmark module under `benchmarks/`. Primary vLLM knobs called out below are the ones we varied in this project.

## 1. PagedAttention + block manager

PagedAttention stores KV cache in fixed-size blocks instead of one contiguous buffer per sequence, which cuts fragmentation and improves utilization on busy GPUs. vLLM exposes this via `block_size` and the block manager behind `max_num_seqs`. Our run swept concurrent sequence counts and plotted reserved GPU memory versus an estimated KV footprint alongside a simulated block-allocation view.

## 2. Continuous batching

Continuous batching keeps the GPU busy by admitting new sequences as others finish, rather than waiting for an entire static batch to complete. There is no single “off” switch in these scripts—we stress the scheduler by scaling concurrent sequences (`max_num_seqs` caps load) with `SamplingParams` tuned for mixed-length generations. The benchmark showed aggregate tokens/s rising with batch size while wall time increased sublinearly compared with naive batching intuition.

## 3. Prefix caching (radix / APC)

Prefix caching reuses KV blocks for shared prompt prefixes (system prompts, few-shot stems) so the long shared prefill is computed once. The controlling flag is `enable_prefix_caching`. We compared two cold runs without APC against cold and warm runs with APC on identical prefixed prompts; warm APC improved latency and throughput versus the no-cache second pass.

## 4. Chunked prefill

Chunked prefill splits long prefills so decode work can interleave, improving time-to-first-token for queued requests. vLLM uses `enable_chunked_prefill` and `max_num_batched_tokens` to bound chunk size. We benchmarked several `max_num_batched_tokens` values (and no chunking) on long shared prompts; the best throughput in our sweep became the summary bar for this feature.

## 5. CUDA graphs

CUDA graphs record a stable sequence of kernels and replay them with lower CPU launch overhead—especially helpful in the decode loop. `enforce_eager=True` disables graphs (eager mode); the default enables them. We measured the same prompt set eager vs graph mode and then scaled batch size under graphs; graph mode improved tokens/s versus eager on our A100 run.

## 6. Speculative decoding

Speculative decoding proposes multiple tokens with a small draft model or n-gram heuristic and verifies them in one target forward pass. vLLM 0.19 expects a `speculative_config` dict (not legacy kwargs). We tried `draft_model` with `meta-llama/Llama-3.2-1B-Instruct` and fell back to `ngram` if draft loading failed; summary throughput used the best successful configuration.

## 7. Quantization

Quantization trades numerical precision for memory and sometimes speed. We compared BF16 baseline (`MODEL_ID`), FP8 via `quantization="fp8"`, and pre-quantized AWQ/GPTQ checkpoints from Hugging Face, using **known parameter-count weight sizes** instead of reading vLLM internals (v1 engine does not expose `model_executor` the same way). The dashboard’s “best” quantization throughput is the maximum across successful loads.

## 8. Multi-LoRA serving

Multi-LoRA keeps one base model and swaps small adapter weights for different tasks. vLLM knobs include `enable_lora`, `max_loras`, and `max_lora_rank`. We benchmarked baseline, LoRA infrastructure enabled, and—when possible—a public adapter, to show throughput and latency overhead from the serving path.