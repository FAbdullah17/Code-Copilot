import os
import json
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm

# === Config ===
DATA_DIR = "./data/processed"
TOKENIZED_SAVE_PATH = "./data/tokenized"      # Tokenized dataset will be saved here
TOKENIZER_SAVE_PATH = "./data/tokenized/tokenizer"               # Tokenizer will be saved here
TOKENIZER_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
TRAIN_SPLIT = 0.8
SEED = 42

# === Setup ===
os.makedirs(TOKENIZED_SAVE_PATH, exist_ok=True)
os.makedirs(TOKENIZER_SAVE_PATH, exist_ok=True)
random.seed(SEED)

# === Helpers ===
def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# === Load and Merge All Dataset Files ===
all_data = []
for fname in ["multipl_e.jsonl", "instructcoder.jsonl", "codeeditsearch.jsonl"]:
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        all_data.extend(load_jsonl(fpath))
    else:
        print(f"⚠️ Skipping {fname} (file not found)")

# === Shuffle and Split ===
random.shuffle(all_data)
split_idx = int(len(all_data) * TRAIN_SPLIT)
train_data = all_data[:split_idx]
valid_data = all_data[split_idx:]

raw_datasets = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(valid_data)
})

print(f"🔢 Total Samples: {len(all_data)}")
print(f"📊 Train: {len(train_data)} | Validation: {len(valid_data)}")

# === Load and Save Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)
print(f"💾 Tokenizer files saved to: {TOKENIZER_SAVE_PATH}")

# === Tokenization Function (Batch Mode) ===
def format_and_tokenize(batch):
    prompts = []

    # We assume that 'instruction', 'input', and 'output' are all keys in the batch dict
    for instruction, input_text, output in zip(batch.get("instruction", []), batch.get("input", []), batch.get("output", [])):
        prompt = f"### Instruction:\n{instruction}\n"
        if input_text:
            prompt += f"### Input:\n{input_text}\n"
        prompt += "### Response:\n"

        # Normalize output
        if isinstance(output, list):
            output = "\n".join(str(line) for line in output)
        elif not isinstance(output, str):
            output = str(output)

        full_text = prompt + output
        prompts.append(full_text)

    return tokenizer(prompts, truncation=True, max_length=2048, padding="max_length")


# === Apply Tokenization ===
tokenized_datasets = raw_datasets.map(
    format_and_tokenize,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

# === Save Tokenized Dataset ===
tokenized_datasets.save_to_disk(TOKENIZED_SAVE_PATH)
print(f"\n✅ Tokenized dataset saved to: {TOKENIZED_SAVE_PATH}")
