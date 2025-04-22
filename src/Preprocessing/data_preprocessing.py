# src/preprocess/preprocess_datasets.py
import os
import json
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from tqdm import tqdm
from typing import List

SAVE_DIR = "./data/processed"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_jsonl(data: List[dict], filename: str, folder: str):
    with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

def process_multipl_e():
    print("\U0001F680 Phase 2A: Preprocessing MultiPL-E dataset...")
    configs = get_dataset_config_names("nuprl/MultiPL-E")
    all_data = []
    for config in tqdm(configs, desc="Processing MultiPL-E configs"):
        dataset = load_dataset("nuprl/MultiPL-E", name=config, split="test")
        for row in dataset:
            if "prompt" in row and "canonical_solution" in row:
                all_data.append({
                    "instruction": f"Write a function to solve: {row['prompt']}",
                    "input": "",
                    "output": row["canonical_solution"]
                })
    save_jsonl(all_data, "multipl_e.jsonl", SAVE_DIR)
    print(f"MultiPL-E: {len(all_data)} samples saved.")

def process_instructcoder():
    print("Preprocessing InstructCoder dataset...")
    dataset = load_dataset("likaixin/InstructCoder", split="train")
    cleaned = []
    for row in tqdm(dataset, desc="Processing InstructCoder"):
        cleaned.append({
            "instruction": row.get("instruction", ""),
            "input": row.get("input", ""),
            "output": row.get("output", "")
        })
    save_jsonl(cleaned, "instructcoder.jsonl", SAVE_DIR)
    print(f"InstructCoder: {len(cleaned)} samples saved.")

def process_codeedit():
    print("Preprocessing CodeEditSearch dataset...")
    all_configs = get_dataset_config_names("cassanof/CodeEditSearch")
    dataset = concatenate_datasets([
        load_dataset("cassanof/CodeEditSearch", name=lang, split="train") for lang in all_configs
    ])
    cleaned = []
    for row in tqdm(dataset, desc="Processing CodeEditSearch"):
        instruction = "Fix the following code bug based on the diff description."
        input_code = row.get("src_code", "")
        output_code = row.get("tgt_code", "")
        desc = row.get("edit_text", "")
        cleaned.append({
            "instruction": instruction + f"\nEdit: {desc}",
            "input": input_code,
            "output": output_code
        })
    save_jsonl(cleaned, "codeeditsearch.jsonl", SAVE_DIR)
    print(f"CodeEditSearch: {len(cleaned)} samples saved.")
