import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk

DATASET_PATH = "/content/tokenized"
TOKENIZER_PATH = "/content/tokenized/tokenizer"
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
OUTPUT_DIR = "/content/output"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fp16 = torch.cuda.is_available()

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
dataset = load_from_disk(DATASET_PATH)
print(f"Dataset splits: {list(dataset.keys())}")
print(f"Train: {len(dataset['train'])}, Validation: {len(dataset['validation'])}")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if fp16 else torch.float32, trust_remote_code=True)
model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=fp16,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()