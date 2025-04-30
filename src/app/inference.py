import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "./output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

def infer(prompt: str, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """
    Generate model output for a given prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(">>> Code Copilot Inference (Type 'exit' to quit)")
    while True:
        user_input = input("Prompt: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = infer(user_input)
        print("\nGenerated Code:\n")
        print(response)
        print("=" * 50)
