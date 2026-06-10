

# ── Model
BASE_MODEL = "gpt2"

# ── LoRA Hyperparameters 
LORA_R          = 16      
LORA_ALPHA      = 32      
LORA_DROPOUT    = 0.05    
LORA_TARGET_MODULES = ["c_attn", "c_proj"] 


OUTPUT_DIR          = "./outputs"
NUM_TRAIN_EPOCHS    = 3
PER_DEVICE_BATCH    = 4        
GRADIENT_ACCUM      = 4        
LEARNING_RATE       = 2e-4
MAX_SEQ_LENGTH      = 512
WARMUP_RATIO        = 0.03
LR_SCHEDULER        = "cosine"
SAVE_STEPS          = 100
LOGGING_STEPS       = 10

# ── Quantization (QLoRA) 
USE_4BIT            = False     
BNB_COMPUTE_DTYPE   = "float16"

# ── Data 
TRAIN_FILE          = "data/train.jsonl"
TEXT_COLUMN         = "text"
