# Troubleshooting

## `RuntimeError: Engine core initialization failed`

Usually means a previous engine did not shut down cleanly and worker processes are still holding GPU resources. **Restart the Python process / Colab runtime** after a crash. For normal teardown, use `cleanup_engine()` from `utils.py`, which terminates `multiprocessing.active_children()`—required for vLLM 0.19.x.

## `GatedRepoError` / HTTP 403 from Hugging Face

The default model (`meta-llama/Llama-3.1-8B-Instruct`) is gated. Accept the license on the model page, create a token, and export `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) before running. Do not commit tokens to git.

## GPU out-of-memory (OOM)

Lower `gpu_memory_utilization` and/or `max_model_len` in `config.py`, or run fewer concurrent sequences in the heavier benchmarks. Quantized checkpoints may help if compatible with your stack.

## Speculative decoding vocab mismatch

Draft and target models must share a tokenizer/vocab layout. Cross-family pairs (for example Llama + Qwen) often fail at load time. Prefer a same-family draft (we default to Llama-3.2-1B) or use the **ngram** speculative method as a fallback (see `b06_speculative_decoding.py`).

## `torch.cuda.memory_allocated()` shows 0

vLLM owns its allocator; PyTorch’s allocated counter may stay near zero. Prefer `torch.cuda.memory_reserved()` **after a warmup generation** while the engine is loaded, as in the PagedAttention benchmark.

## `model_executor` attribute errors

vLLM v1 engines may not expose historical internal attributes. Do not rely on `engine.llm_engine.model_executor` for parameter sizes; use documented configs or **known weight sizes** (see quantization benchmark).

## `speculative_model=` / unexpected keyword errors

vLLM 0.19 uses `speculative_config={...}` passed to `LLM()`. Older examples with `speculative_model=` will raise; update to the dict form with `method` and `num_speculative_tokens`.

## Dashboard fails with “Missing result JSON”

Run all benchmarks first (`python run_all.py`) or the individual modules so `results/*.json` exists, then `python dashboard.py`.
