import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

test_prompts = [
    "Write a Python function to check if a number is prime.",
    "Fix the bug in this code:\ndef add(a, b):\n  return a - b",
    "Explain what this code does:\n\nfor i in range(5): print(i ** 2)"
]

def generate_output(prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("===== Model Test Outputs =====\n")
    for i, prompt in enumerate(test_prompts):
        print(f"Prompt {i+1}: {prompt}\n")
        response = generate_output(prompt)
        print("Generated Output:\n", response)
        print("=" * 50, "\n")
