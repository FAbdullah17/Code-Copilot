from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "deepseek-coder-1.3b-base"
CHECKPOINT_PATH = "output/checkpoint-750"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
model.eval()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return FileResponse("static/index.html", "static/style.css")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_code(request: PromptRequest):
    input_prompt = request.prompt
    inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, top_k=50, top_p=0.95, temperature=0.8)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    generated_code = generated_text[len(input_prompt):].strip()

    return JSONResponse(content={"generated_code": generated_code})