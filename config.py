"""
config.py
Central configuration for the fine-tuning project.
Tweak these values to experiment — that's the point!
"""

# ── Model ──────────────────────────────────────────────────────────────────
# BASE_MODEL = "mistralai/Mistral-7B-v0.1"   # swap with any HuggingFace model
# For quick local testing, use: "gpt2" (very small, no GPU needed)
BASE_MODEL = "gpt2"

# ── LoRA Hyperparameters ────────────────────────────────────────────────────
LORA_R          = 16      # rank — higher = more parameters, more capacity
LORA_ALPHA      = 32      # scaling factor (usually 2x rank)
LORA_DROPOUT    = 0.05    # regularization
LORA_TARGET_MODULES = ["c_attn", "c_proj"]  # which layers to apply LoRA to

# ── Training Hyperparameters ────────────────────────────────────────────────
OUTPUT_DIR          = "./outputs"
NUM_TRAIN_EPOCHS    = 3
PER_DEVICE_BATCH    = 4        # reduce to 1 or 2 if GPU memory is limited
GRADIENT_ACCUM      = 4        # simulates larger batch size
LEARNING_RATE       = 2e-4
MAX_SEQ_LENGTH      = 512
WARMUP_RATIO        = 0.03
LR_SCHEDULER        = "cosine"
SAVE_STEPS          = 100
LOGGING_STEPS       = 10

# ── Quantization (QLoRA) ────────────────────────────────────────────────────
USE_4BIT            = False     # enables QLoRA (loads model in 4-bit)
BNB_COMPUTE_DTYPE   = "float16"

# ── Data ───────────────────────────────────────────────────────────────────
TRAIN_FILE          = "data/train.jsonl"
TEXT_COLUMN         = "text"
