"""Shared constants: models, paths, Modal settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

# E0 calibration: 32B decodes 10/12 instances (readout AUROC >= 0.97); 8B and 14B fall short.
RECEIVER_MODEL = "Qwen/Qwen3-32B"  # 64 layers, d_model 5120
FALLBACK_MODEL = "Qwen/Qwen3-14B"
LOCAL_SMOKE_MODEL = "Qwen/Qwen3-1.7B"

N_PROBLEMS = 300
SEED = 0

MONITOR_FORMAT = "v2_score"  # suspicion digit 0-9; best calibrated in E0
POSITIONS = ("last", "carrier_mean", "code_mean")
CACHE_LAYER_STRIDE = 2  # keep layers 1, 3, ..., 63 (~28 GB for the full E0 grid on 32B)
READERS = ("receiver", "naive", "benign")
PLACEMENTS = ("before", "after")

MODAL_VOLUME = "rtd-cache"
MODAL_GPU = "A100-80GB"
MODAL_HF_SECRET = "huggingface-secret"  # injects HF_TOKEN
MODAL_ANTHROPIC_SECRET = "anthropic-secret"  # injects ANTHROPIC_API_KEY
