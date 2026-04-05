# vLLM Inference Profiler

Modular benchmarks and plots for eight major [vLLM](https://github.com/vllm-project/vllm) features (PagedAttention, continuous batching, prefix caching, chunked prefill, CUDA graphs, speculative decoding, quantization, multi-LoRA). Logic is split from the original Colab workflow into `config.py`, `utils.py`, and `benchmarks/b01_*.py` … `b08_*.py`, with JSON under `results/` and PNGs under `plots/`.

**Requirements:** NVIDIA GPU (tested on A100 80GB), Python 3.10+, Hugging Face access for gated Llama weights.

## Authentication

```bash
export HF_TOKEN=your_token_here   # or HUGGING_FACE_HUB_TOKEN
```

Never commit tokens. The old Colab script embedded a token; this repo uses **environment variables only**.

## How to Run

### Option A: Command line

```bash
git clone https://github.com/YOUR_USERNAME/vllm-inference-profiler.git
cd vllm-inference-profiler
pip install -r requirements.txt

# Run all 8 benchmarks (long; loads multiple models)
python run_all.py

# Run a single feature (1–8)
python run_single.py --feature 3

# Regenerate plot from saved JSON only
python run_single.py --feature 3 --plot-only

# Combined LinkedIn-style dashboard (needs all eight result JSON files)
python dashboard.py
```

## Configuration

All parameters are centralized in `config.py`. Edit this file to customize benchmarks for your setup.

### Global Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_ID` | `meta-llama/Llama-3.1-8B-Instruct` | Base model to benchmark |
| `GPU_MEMORY_UTIL` | `0.85` | Fraction of GPU RAM for model + KV cache |
| `MAX_MODEL_LEN` | `8192` | Max total sequence length |
| `BENCH_RUNS` | `3` | Measurement iterations (median taken) |
| `WARMUP_RUNS` | `1` | Warmup iterations before measuring |
| `MAX_TOKENS` | `128` | Default max output tokens (PagedAttention, Continuous Batching) |

Additional sampling caps (defaults match the original notebook) live in `config.py`: `PREFIX_CACHE_MAX_TOKENS`, `CHUNKED_PREFILL_MAX_TOKENS`, `CUDA_GRAPHS_MAX_TOKENS`, `SPEC_DECODE_MAX_TOKENS`, `QUANTIZATION_MAX_TOKENS`, `LORA_MAX_TOKENS`. The CLI flag `--max-tokens` sets **all** of these, plus `MAX_TOKENS`, to the same value.

### Feature-Specific Settings

| Parameter | Default | Used By |
|-----------|---------|---------|
| `BATCH_SIZES` | `[1, 4, 8, 16, 32, 64]` | PagedAttention |
| `CB_BATCH_SIZES` | `[1, 2, 4, 8, 16, 32, 64, 128]` | Continuous Batching |
| `SHARED_PREFIX` | Long system + few-shot prefix | Prefix Caching |
| `CHUNK_SIZES` | `[512, 1024, 2048, 4096]` | Chunked Prefill |
| `CG_BATCH_SIZES` | `[1, 4, 8, 16, 32]` | CUDA Graphs |
| `DRAFT_MODEL` | `meta-llama/Llama-3.2-1B-Instruct` | Speculative Decoding |
| `SPEC_K_VALUES` | `[5, 3]` | Speculative Decoding |
| `QUANT_CONFIGS` | BF16, FP8, AWQ, GPTQ | Quantization |
| `MAX_LORAS` | `4` | Multi-LoRA |
| `LORA_RANK` | `64` | Multi-LoRA |

### Command Line Overrides

Override config without editing files:

```bash
python run_single.py --feature paged_attention --model Qwen/Qwen2.5-7B-Instruct --batch-sizes 1,8,32,128
python run_single.py --feature quantization --quant-methods fp8,awq --gpu-mem 0.70
python run_single.py --feature speculative_decoding --draft-model meta-llama/Llama-3.2-1B-Instruct --spec-tokens 3,5,7
python run_single.py --feature chunked_prefill --chunk-sizes 256,512,1024,2048,4096 --max-len 4096
python run_single.py --feature multi_lora --max-loras 8 --lora-rank 32
python run_single.py --feature cuda_graphs --max-tokens 256 --runs 5 --warmup 2
python run_all.py --model Qwen/Qwen2.5-7B-Instruct --gpu-mem 0.70
python run_all.py --skip speculative_decoding
python run_all.py --plot-only
```

`--feature` accepts a number `1`–`8` or a name such as `paged_attention`, `cuda_graphs`, `quantization`.

### Option B: Google Colab

1. Upload or clone this repository into the Colab runtime.
2. Set runtime to an A100 (or similar) GPU.
3. Open `notebooks/vLLM_Feature_Benchmark_Dashboard.ipynb` and run the cells (install → token → `run_all.py`).

## Layout

| Path | Role |
|------|------|
| `config.py` | Model IDs, GPU settings, dark/light plot themes, quantization table |
| `utils.py` | Engine cleanup, timing, `save_results` / `load_results` |
| `benchmarks/b0X_*.py` | One feature each: `run_benchmark()` + `plot_results()` |
| `results/*.json` | Serialized metrics (gitignored except `.gitkeep`) |
| `plots/*.png` | Per-feature charts + `vllm_benchmark_linkedin.png` |
| `docs/FEATURES.md` | Short explanation of each benchmark |
| `docs/TROUBLESHOOTING.md` | Common vLLM 0.19.x issues and fixes |

## License

MIT — see `LICENSE`.
