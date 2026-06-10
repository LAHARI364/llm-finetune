"""
inference.py
Loads the fine-tuned LoRA adapter and runs inference.
Use this to test your model after training.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

import config

ADAPTER_PATH = f"{config.OUTPUT_DIR}/final_adapter"


def load_finetuned_model():
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, user_input: str = ""):
    """Format prompt and generate a response."""
    if user_input:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{user_input}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


if __name__ == "__main__":
    model, tokenizer = load_finetuned_model()

    # Test cases
    test_cases = [
        {"instruction": "What is machine learning?"},
        {"instruction": "Explain overfitting in simple terms."},
        {"instruction": "Write a Python function to add two numbers."},
        {
            "instruction": "Summarize the following text.",
            "input": "Transformers use self-attention to process sequential data in parallel, revolutionizing NLP."
        },
    ]

    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)

    for i, test in enumerate(test_cases, 1):
        instruction = test["instruction"]
        user_input  = test.get("input", "")

        print(f"\n[Test {i}]")
        print(f"Instruction: {instruction}")
        if user_input:
            print(f"Input: {user_input}")

        response = generate_response(model, tokenizer, instruction, user_input)
        print(f"Response: {response}")
        print("-" * 40)
