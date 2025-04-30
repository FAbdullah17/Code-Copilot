from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn

MODEL_DIR = "output/checkpoint-750"  

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading model from {MODEL_DIR}: {e}")

app = FastAPI(title="Code Copilot API", description="Generate code using your fine-tuned LLM", version="1.0")

class CodeRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

class CodeResponse(BaseModel):
    generated_code: str

def generate_code(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.post("/generate", response_model=CodeResponse)
def generate_code_endpoint(request: CodeRequest):
    try:
        result = generate_code(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        return CodeResponse(generated_code=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
