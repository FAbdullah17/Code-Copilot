import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate

OUTPUT_DIR = "./output"
model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model.eval()

examples = [
    {
        "input": "Write a function to calculate factorial of a number.",
        "expected": "def factorial(n):\n    return 1 if n == 0 else n * factorial(n - 1)"
    },
    {
        "input": "Create a Python list with the first 5 even numbers.",
        "expected": "even_numbers = [2, 4, 6, 8, 10]"
    },
]

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

predictions = []
references = []

for example in examples:
    inputs = tokenizer.encode(example["input"], return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=128)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nInput: {example['input']}")
    print(f"Output: {decoded_output}")
    print(f"Expected: {example['expected']}")

    predictions.append(decoded_output)
    references.append(example["expected"])

bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
rouge_score = rouge.compute(predictions=predictions, references=references)

print("\nEvaluation Results:")
print(f"BLEU Score: {bleu_score['bleu']:.4f}")
print(f"ROUGE-L Score: {rouge_score['rougeL']:.4f}")