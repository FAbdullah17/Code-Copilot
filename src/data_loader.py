import os
from datasets import load_from_disk
from transformers import AutoTokenizer

# === Paths ===
TOKENIZED_DATA_PATH = "./data/tokenized"
TOKENIZER_CONFIG_PATH = "./data/tokenized/tokenizer"

# === Validation ===
if not os.path.exists(TOKENIZED_DATA_PATH):
    raise FileNotFoundError(f"❌ Tokenized dataset not found at {TOKENIZED_DATA_PATH}")
if not os.path.exists(TOKENIZER_CONFIG_PATH):
    raise FileNotFoundError(f"❌ Tokenizer config not found at {TOKENIZER_CONFIG_PATH}")

# === Load Tokenized Dataset ===
print("📥 Loading tokenized dataset from disk...")
tokenized_datasets = load_from_disk(TOKENIZED_DATA_PATH)
print(f"✅ Loaded dataset with splits: {list(tokenized_datasets.keys())}")
print(f"🔢 Train samples: {len(tokenized_datasets['train'])}, Validation samples: {len(tokenized_datasets['validation'])}")

# === Load Tokenizer ===
print("📥 Loading tokenizer config from disk...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CONFIG_PATH, trust_remote_code=True)
print(f"✅ Tokenizer loaded: {tokenizer.name_or_path}")

# === Preview (Optional Debug Info) ===
print("\n🔍 Searching for a meaningful decoded sample...")

for sample in tokenized_datasets["train"]:
    if "input_ids" in sample and isinstance(sample["input_ids"], list) and sample["input_ids"]:
        decoded = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        if decoded.strip() and "Instruction" in decoded and "Response" in decoded:
            print("\n🔍 Sample Decoded Input:\n", decoded)
            break
else:
    print("⚠️ No valid decoded sample found.")
