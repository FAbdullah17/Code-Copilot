import os
import json
from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm

RAW_DIR = "./data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

def download_multipl_e_raw():
    print("\U0001F680 Phase 1A: Downloading raw MultiPL-E datasets...")
    configs = get_dataset_config_names("nuprl/MultiPL-E")
    for config in tqdm(configs, desc="Downloading MultiPL-E configs"):
        dataset = load_dataset("nuprl/MultiPL-E", name=config, split="test")
        path = os.path.join(RAW_DIR, f"{config}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in dataset:
                json.dump(row, f)
                f.write("\n")
    print("All datasets are downloaded and saved to ./data/raw")
