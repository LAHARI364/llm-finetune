import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

import config


# Load tokenizer
def load_tokenizer():
    print(f"Loading tokenizer: {config.BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.padding_side = "right"
    return tokenizer


# Load model in 4-bit (QLoRA)
def load_model():
    print(f"Loading model in 4-bit: {config.BASE_MODEL}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.USE_4BIT,
        bnb_4bit_quant_type="nf4",           
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,      
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",                   
        trust_remote_code=True,
    )

    
    model = prepare_model_for_kbit_training(model)
    return model


# Attach LoRA adapter
def apply_lora(model):
    print(f"Applying LoRA adapters (r={config.LORA_R}, alpha={config.LORA_ALPHA})")

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# Load dataset
def load_data():
    print(f"Loading dataset from {config.TRAIN_FILE}")
    dataset = load_dataset("json", data_files=config.TRAIN_FILE, split="train")
    print(f"  → {len(dataset)} training examples loaded")
    return dataset


#Train
def train(model, tokenizer, dataset):
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_BATCH,
        gradient_accumulation_steps=config.GRADIENT_ACCUM,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type=config.LR_SCHEDULER,
        save_steps=config.SAVE_STEPS,
        logging_steps=config.LOGGING_STEPS,
        fp16=True,                    
        optim="adamw_torch",     
        report_to="none",            
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )
    print("\n🚀 Starting training...")
    trainer.train()
    print("✅ Training complete!")
    return trainer


#save adapter weights
def save_model(trainer, tokenizer):
    save_path = f"{config.OUTPUT_DIR}/final_adapter"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"💾 Adapter saved to {save_path}")


# Main
if __name__ == "__main__":
    tokenizer = load_tokenizer()
    model     = load_model()
    model     = apply_lora(model)
    dataset   = load_data()
    trainer   = train(model, tokenizer, dataset)
    save_model(trainer, tokenizer)
