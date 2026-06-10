"""
evaluate.py
Computes ROUGE scores to evaluate how well the fine-tuned model performs.
ROUGE is a standard metric for text generation tasks.
"""

from rouge_score import rouge_scorer
from inference import load_finetuned_model, generate_response

# Ground-truth examples to evaluate against
EVAL_DATA = [
    {
        "instruction": "What is machine learning?",
        "reference": "Machine learning is a branch of AI where models learn patterns from data to make predictions or decisions without being explicitly programmed."
    },
    {
        "instruction": "Explain overfitting in simple terms.",
        "reference": "Overfitting is when a model memorizes training data and fails to generalize to new unseen data."
    },
    {
        "instruction": "What is gradient descent?",
        "reference": "Gradient descent is an optimization algorithm that adjusts model parameters to minimize the loss function."
    },
]


def evaluate():
    model, tokenizer = load_finetuned_model()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    total_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}

    print("\nEvaluation Results:")
    print("=" * 60)

    for item in EVAL_DATA:
        prediction = generate_response(model, tokenizer, item["instruction"])
        scores = scorer.score(item["reference"], prediction)

        print(f"\nInstruction: {item['instruction']}")
        print(f"Predicted : {prediction[:100]}...")
        print(f"ROUGE-1: {scores['rouge1'].fmeasure:.3f} | "
              f"ROUGE-2: {scores['rouge2'].fmeasure:.3f} | "
              f"ROUGE-L: {scores['rougeL'].fmeasure:.3f}")

        for key in total_scores:
            total_scores[key] += scores[key].fmeasure

    n = len(EVAL_DATA)
    print("\n" + "=" * 60)
    print("Average Scores:")
    for key, val in total_scores.items():
        print(f"  {key}: {val/n:.3f}")


if __name__ == "__main__":
    evaluate()
