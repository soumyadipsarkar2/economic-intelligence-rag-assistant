# -*- coding: utf-8 -*-
"""Global configuration, plotting themes, and Hugging Face auth for benchmarks."""
import os
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# Hugging Face — use environment variables (never commit tokens)
# ============================================================
def ensure_hf_login():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        print("Authenticated with HuggingFace")
    else:
        print(
            "Warning: No HF token in environment. "
            "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN for gated models."
        )


# ============================================================
# USER INPUTS — Change these to customize your benchmark
# ============================================================

# Model
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DTYPE = "auto"

# GPU
GPU_MEMORY_UTIL = 0.85
MAX_MODEL_LEN = 8192

# Benchmark control
WARMUP_RUNS = 1
BENCH_RUNS = 3
MAX_TOKENS = 128

# Feature 1 & 2: PagedAttention, Continuous Batching
BATCH_SIZES = [1, 4, 8, 16, 32, 64]
CB_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

# Feature 3: Prefix Caching
SHARED_PREFIX = (
    "You are an expert AI assistant specializing in science and technology. "
    "You provide detailed, accurate, and well-structured answers. "
    "Always cite your reasoning and explain complex concepts clearly.\n\n"
    "Example 1: Q: What is quantum entanglement?\n"
    "A: Quantum entanglement is a phenomenon where two particles become "
    "interconnected so that the quantum state of one instantly influences the "
    "other, regardless of distance. This was famously called 'spooky action "
    "at a distance' by Einstein.\n\n"
    "Example 2: Q: How does CRISPR work?\n"
    "A: CRISPR-Cas9 uses a guide RNA to direct the Cas9 enzyme to a specific "
    "DNA location, where it makes a precise cut. The cell then repairs the "
    "break, allowing researchers to edit genes with unprecedented accuracy.\n\n"
    "Example 3: Q: Explain the Higgs boson.\n"
    "A: The Higgs boson is a fundamental particle associated with the Higgs "
    "field, which gives mass to other particles. Discovered at CERN in 2012, "
    "it confirmed a key prediction of the Standard Model.\n\n"
    "Now answer the following question:\n"
)

# Feature 4: Chunked Prefill
CHUNK_SIZES = [512, 1024, 2048, 4096]

# Feature 5: CUDA Graphs
CG_BATCH_SIZES = [1, 4, 8, 16, 32]

# Feature 6: Speculative Decoding
DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
SPEC_K_VALUES = [5, 3]

# Feature 7: Quantization
def build_quant_configs():
    """Rows that use the base model track the current MODEL_ID."""
    return [
        ("BF16\n(baseline)", MODEL_ID, {}),
        ("FP8", MODEL_ID, {"quantization": "fp8"}),
        ("AWQ\n(4-bit)", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", {}),
        ("GPTQ\n(4-bit)", "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4", {}),
    ]


QUANT_CONFIGS = build_quant_configs()

KNOWN_WEIGHT_SIZES = {
    "BF16\n(baseline)": 16.1,
    "FP8": 8.0,
    "AWQ\n(4-bit)": 4.0,
    "GPTQ\n(4-bit)": 4.0,
}

# Feature 8: Multi-LoRA
MAX_LORAS = 4
LORA_RANK = 64

# Other sampling caps (original notebook defaults; override with CLI --max-tokens)
PREFIX_CACHE_MAX_TOKENS = 150
CHUNKED_PREFILL_MAX_TOKENS = 100
CUDA_GRAPHS_MAX_TOKENS = 200
SPEC_DECODE_MAX_TOKENS = 200
QUANTIZATION_MAX_TOKENS = 150
LORA_MAX_TOKENS = 128

# ============================================================
# Paths
# ============================================================
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# ============================================================
# Dark theme (per-benchmark plots)
# ============================================================
COLORS = {
    "primary": "#6C63FF",
    "secondary": "#FF6584",
    "accent1": "#43E97B",
    "accent2": "#F9D423",
    "accent3": "#00D2FF",
    "accent4": "#FF9A9E",
    "accent5": "#A18CD1",
    "accent6": "#FBC2EB",
    "bg_dark": "#0D1117",
    "bg_card": "#161B22",
    "text": "#C9D1D9",
    "text_light": "#8B949E",
    "grid": "#21262D",
}
PALETTE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["accent1"],
    COLORS["accent2"],
    COLORS["accent3"],
    COLORS["accent4"],
    COLORS["accent5"],
    COLORS["accent6"],
]


def setup_plot_style():
    """Apply consistent dark professional styling."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg_dark"],
            "axes.facecolor": COLORS["bg_card"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["text_light"],
            "ytick.color": COLORS["text_light"],
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.6,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "figure.titlesize": 16,
            "figure.titleweight": "bold",
            "legend.facecolor": COLORS["bg_card"],
            "legend.edgecolor": COLORS["grid"],
            "legend.fontsize": 10,
        }
    )


# ============================================================
# White theme (LinkedIn / combined dashboard)
# ============================================================
COLORS_WHITE = {
    "bg": "#FFFFFF",
    "card": "#F8F9FA",
    "text": "#1A1A2E",
    "text_light": "#6C757D",
    "grid": "#E9ECEF",
    "blue": "#4361EE",
    "red": "#E63946",
    "green": "#2EC4B6",
    "yellow": "#F4A261",
    "cyan": "#00B4D8",
    "purple": "#7209B7",
    "pink": "#FF6B6B",
    "orange": "#FB8500",
}
PALETTE_WHITE = [
    COLORS_WHITE["blue"],
    COLORS_WHITE["red"],
    COLORS_WHITE["green"],
    COLORS_WHITE["yellow"],
    COLORS_WHITE["cyan"],
    COLORS_WHITE["purple"],
    COLORS_WHITE["pink"],
    COLORS_WHITE["orange"],
]


def setup_plot_style_white():
    """Light theme for social / presentation exports."""
    W = COLORS_WHITE
    plt.rcParams.update(
        {
            "figure.facecolor": W["bg"],
            "axes.facecolor": W["card"],
            "axes.edgecolor": W["grid"],
            "axes.labelcolor": W["text"],
            "text.color": W["text"],
            "xtick.color": W["text_light"],
            "ytick.color": W["text_light"],
            "grid.color": W["grid"],
            "grid.alpha": 0.8,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.facecolor": W["bg"],
            "legend.edgecolor": W["grid"],
        }
    )


def refresh_quant_configs_after_model_change():
    """Call after CLI overrides to MODEL_ID so BF16/FP8 rows track the base model."""
    global QUANT_CONFIGS
    QUANT_CONFIGS = build_quant_configs()


def apply_max_tokens_override(n: int):
    """Set all per-benchmark output caps to the same value (CLI --max-tokens)."""
    global MAX_TOKENS, PREFIX_CACHE_MAX_TOKENS, CHUNKED_PREFILL_MAX_TOKENS
    global CUDA_GRAPHS_MAX_TOKENS, SPEC_DECODE_MAX_TOKENS, QUANTIZATION_MAX_TOKENS, LORA_MAX_TOKENS
    MAX_TOKENS = n
    PREFIX_CACHE_MAX_TOKENS = n
    CHUNKED_PREFILL_MAX_TOKENS = n
    CUDA_GRAPHS_MAX_TOKENS = n
    SPEC_DECODE_MAX_TOKENS = n
    QUANTIZATION_MAX_TOKENS = n
    LORA_MAX_TOKENS = n


def filter_quant_configs(methods_csv: str):
    """
    Restrict QUANT_CONFIGS to named methods (comma-separated).
    Names: bf16, fp8, awq, gptq (case-insensitive).
    """
    global QUANT_CONFIGS
    want = {x.strip().lower() for x in methods_csv.split(",") if x.strip()}
    if not want:
        return

    def tag(display_name: str):
        nl = display_name.replace("\n", " ").lower()
        if "bf16" in nl or "baseline" in nl:
            return "bf16"
        if "fp8" in nl:
            return "fp8"
        if "awq" in nl:
            return "awq"
        if "gptq" in nl:
            return "gptq"
        return None

    full = build_quant_configs()
    filtered = [row for row in full if tag(row[0]) in want]
    if filtered:
        QUANT_CONFIGS = filtered
    else:
        print("Warning: --quant-methods matched no configs; keeping full QUANT_CONFIGS.")
