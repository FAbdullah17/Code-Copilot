# Code Copilot Project

**Code Copilot** is a comprehensive code-generation assistant built upon a fine-tuned DeepSeek Coder (1.3B) model. It can **generate**, **correct**, and **explain** code across multiple languages, and is served via a FastAPI backend with a simple web UI.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Data Pipeline](#data-pipeline)
4. [Model Fine‑Tuning](#model-fine-tuning)
5. [Evaluation & Testing](#evaluation--testing)
6. [Inference & API](#inference--api)
7. [Setup & Usage](#setup--usage)
8. [Future Work](#future-work)
9. [Acknowledgements](#acknowledgements)

---

## Project Overview

Code Copilot delivers a versatile code assistant that:

- **Generates code** from natural-language prompts.
- **Fixes bugs** based on diff descriptions.
- **Explains code** in a human‑readable way.

We curated and processed three hero datasets—MultiPL‑E, InstructCoder, and CodeEditSearch—covering general generation, instruction following, and code editing. The DeepSeek Coder base model is quantized with 4‑bit QLoRA and trained with adapter layers for resource‑efficient fine‑tuning. The final system is deployed via FastAPI and a sleek single‑page web UI.

---

## Repository Structure

```text
CODE‑COPILOT/
├── app/                      # FastAPI application
│   ├── __init__.py
│   ├── main.py               # API server and routes
│   └── inference.py          # helper for standalone inference
├── data/                     # Data artifacts
│   ├── raw/                  # Raw JSONL dataset downloads
│   ├── processed/            # Cleaned JSONL
│   └── tokenized/            # HuggingFace Dataset + tokenizer files
├── notebooks/                # EDA notebooks
│   ├── EDA1.ipynb            # Processed data analysis
│   └── EDA2.ipynb            # Tokenized data analysis
├── output/                   # Merged model checkpoints
│   └── checkpoint-750/       # Best QLoRA adapter files + config
├── src/                      # Data pipeline and training scripts
│   ├── preprocessing/        # download, preprocess code
│   ├── tokenization/         # tokenization scripts
│   └── train.py              # fine-tuning script
├── static/                   # Frontend UI files
│   ├── index.html
│   └── style.css
├── weights/                  # (optional) external weight files
├── evaluate.py               # BLEU/ROUGE evaluation
├── test.py                   # CLI testing script
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── LICENSE
```

---

## Data Pipeline

1. **Raw Download** (`src/preprocessing/download.py`): Pulls MultiPL‑E, InstructCoder, CodeEditSearch via `datasets` API into `data/raw/`.
2. **Cleaning** (`src/preprocessing/data_preprocessing.py`): Filters languages, formats to `{instruction, input, output}`, writes JSONL to `data/processed/`.
3. **Tokenization** (`src/tokenization/tokenize.py`): Loads processed JSONL, splits 80/20, tokenizes with DeepSeek tokenizer, saves to `data/tokenized/`.

---

## Model Fine‑Tuning

- **Quantization**: 4‑bit NF4 with `bitsandbytes` for memory savings.
- **Adapters (QLoRA)**: LoRA rank=4, α=16, targeting `q_proj` & `v_proj` layers via `peft`.
- **Trainer**: HuggingFace `Trainer` with gradient accumulation, early stopping, best‑model checkpoint.
- **Script**: `src/train.py` or accompanying notebook for Kaggle/Colab.

---

## Evaluation & Testing

- **evaluate.py**: Computes BLEU and ROUGE‑L using `evaluate` library on sample or full test sets.
- **test.py**: Interactive CLI for quick sanity checks of generation, correction, and explanation tasks.

---

## Inference & API

1. **inference.py**: Simple script to load model and generate code from prompts.
2. **FastAPI** (`app/main.py`): REST endpoint `/generate` accepting JSON `{prompt, max_new_tokens, temperature, top_p}` and returning generated code.
3. **UI** (`static/index.html` + `static/style.css`): Chat‑style webpage for interactive code generation.

---

## Setup & Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare Data**:
   ```bash
   python src/preprocessing/data_main.py
   python src/tokenization/tokenize.py
   ```
3. **Fine‑Tune Model**:
   ```bash
   python src/train.py
   ```
4. **Evaluate**:
   ```bash
   python evaluate.py
   ```
5. **Test CLI**:
   ```bash
   python test.py
   ```
6. **Run API + UI**:
   ```bash
   uvicorn app.main:app --reload
   # Open http://localhost:8000 in your browser
   ```

---

## Future Work

- Scale to full datasets.  
- Add more languages and edge‑case handling.  
- Websocket streaming for live code completion.  
- Docker/Kubernetes deployment.

---

## Acknowledgements

- Hugging Face `transformers`, `datasets`, `peft`, `bitsandbytes`.  
- Creators of MultiPL‑E, InstructCoder, CodeEditSearch.  
