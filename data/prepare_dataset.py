"""
data/prepare_dataset.py
Generates a sample instruction-following dataset (Q&A format).
Replace this with your own data in the same format for real use.
"""

import json
import os

SAMPLE_DATA = [
    {
        "instruction": "What is machine learning?",
        "input": "",
        "output": "Machine learning is a branch of AI where models learn patterns from data to make predictions or decisions without being explicitly programmed."
    },
    {
        "instruction": "Explain overfitting in simple terms.",
        "input": "",
        "output": "Overfitting is when a model memorizes training data too well and fails to generalize to new data. It performs great on training but poorly on unseen examples."
    },
    {
        "instruction": "What is the difference between supervised and unsupervised learning?",
        "input": "",
        "output": "Supervised learning uses labeled data (input-output pairs). Unsupervised learning finds patterns in data without labels, like clustering similar items together."
    },
    {
        "instruction": "Summarize the following text.",
        "input": "Deep learning uses neural networks with many layers to learn hierarchical representations of data. It has shown great success in vision, NLP, and speech.",
        "output": "Deep learning leverages multi-layer neural networks to learn data representations, excelling in vision, language, and speech tasks."
    },
    {
        "instruction": "Write a Python function to reverse a string.",
        "input": "",
        "output": "def reverse_string(s):\n    return s[::-1]"
    },
    {
        "instruction": "What is gradient descent?",
        "input": "",
        "output": "Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction that reduces the loss function, helping the model learn."
    },
    {
        "instruction": "What does fine-tuning mean in LLMs?",
        "input": "",
        "output": "Fine-tuning means taking a pre-trained LLM and training it further on a specific dataset so it becomes better at a particular task or domain."
    },
    {
        "instruction": "Translate to formal English.",
        "input": "Hey can u explain what neural nets do??",
        "output": "Could you please explain the functionality of neural networks?"
    },
]

def format_prompt(example):
    """Alpaca-style prompt format used by most instruction-tuned models."""
    if example["input"]:
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )

def prepare_and_save():
    os.makedirs("data", exist_ok=True)

    formatted = [{"text": format_prompt(ex)} for ex in SAMPLE_DATA]

    # Save as JSONL (one JSON object per line — standard for LLM training)
    with open("data/train.jsonl", "w") as f:
        for item in formatted:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Saved {len(formatted)} training examples to data/train.jsonl")
    print("\nSample entry:")
    print(formatted[0]["text"])

if __name__ == "__main__":
    prepare_and_save()
