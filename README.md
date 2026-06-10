# LLM Fine-Tuning with QLoRA

Fine-tuning **Mistral-7B** using **QLoRA** (Quantized Low-Rank Adaptation) for instruction-following tasks.

## Tech Stack
- **Model**: Mistral-7B-v0.1 (via HuggingFace)
- **Method**: QLoRA — 4-bit quantization + LoRA adapters
- **Libraries**: `transformers`, `peft`, `trl`, `bitsandbytes`

## Project Structure
```
llm-finetune/
├── config.py              # All hyperparameters in one place
├── train.py               # Main training script
├── inference.py           # Run the fine-tuned model
├── evaluate.py            # ROUGE score evaluation
├── data/
│   ├── prepare_dataset.py # Generate/format training data
│   └── train.jsonl        # Training examples (auto-generated)
├── outputs/               # Saved model checkpoints
└── requirements.txt
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset
python data/prepare_dataset.py

# 3. Train
python train.py

# 4. Run inference
python inference.py

# 5. Evaluate
pip install rouge-score
python evaluate.py
```

## Key Concepts Used
| Concept | What it does |
|---|---|
| **QLoRA** | Loads model in 4-bit to save GPU memory (~5GB vs ~14GB) |
| **LoRA** | Trains only small adapter matrices, not the full model |
| **SFTTrainer** | Supervised Fine-Tuning trainer from `trl` |
| **Alpaca format** | Standard instruction/input/response prompt format |
| **ROUGE** | Metric to evaluate generated text quality |

## GPU Requirements
- Minimum: 8GB VRAM (with 4-bit QLoRA)
- Recommended: 16–24GB VRAM
- CPU-only: Switch `BASE_MODEL = "gpt2"` in `config.py` for testing

## To use your own data
Replace the `SAMPLE_DATA` list in `data/prepare_dataset.py` with your own Q&A pairs in this format:
```python
{"instruction": "...", "input": "", "output": "..."}
```
