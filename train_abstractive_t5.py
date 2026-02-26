"""
train_abstractive_t5.py
Fine-tune T5-small on the XSum news summarization dataset.

Dataset: xsum (Hugging Face)
Model output: models/t5-xsum
"""

import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

BASE_MODEL = "t5-small"
DATASET_NAME = "xsum"
OUTPUT_DIR = "models/t5-xsum"


class XSumDataset(Dataset):
    """Wrapper to use XSum with T5 for summarization."""

    def __init__(self, hf_split, tokenizer, max_input_len=512, max_target_len=64):
        self.split = hf_split
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        article = str(self.split[idx]["document"])
        summary = str(self.split[idx]["summary"])

        source = "summarize: " + article

        model_inputs = self.tokenizer(
            source,
            max_length=self.max_input_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = self.tokenizer(
            text_target=summary,
            max_length=self.max_target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def main():
    print("=" * 70)
    print("Fine-tuning T5-small on XSum summarization dataset")
    print("=" * 70)

    # Control size (XSum has ~204k train examples)
    max_train_examples = 5000   # safer & faster than 20000
    max_val_examples = 500

    print(f"Base model:      {BASE_MODEL}")
    print(f"Dataset:         {DATASET_NAME}")
    print(f"Train examples:  up to {max_train_examples}")
    print(f"Val examples:    up to {max_val_examples}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load dataset
    print("\n📥 Loading dataset from Hugging Face...")
    raw_datasets = load_dataset(DATASET_NAME)  # no version arg

    train_split = raw_datasets["train"]
    val_split = raw_datasets["validation"]

    if max_train_examples is not None:
        train_split = train_split.select(
            range(min(max_train_examples, len(train_split)))
        )
    if max_val_examples is not None:
        val_split = val_split.select(
            range(min(max_val_examples, len(val_split)))
        )

    print(f"Loaded {len(train_split)} training and {len(val_split)} validation examples.")

    # Tokenizer & model
    print("\n🔧 Loading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)
    print("Model and tokenizer loaded.")

    train_dataset = XSumDataset(train_split, tokenizer)
    val_dataset = XSumDataset(val_split, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Safer training hyperparameters
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,              # was 3
        per_device_train_batch_size=8,   # lower to 4 if GPU OOM
        per_device_eval_batch_size=8,
        learning_rate=2e-5,              # was 3e-4
        weight_decay=0.01,
        logging_steps=100,
        save_steps=2000,
        save_total_limit=1,
        logging_dir=f"{OUTPUT_DIR}/logs",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to=[],    # disable logging integrations
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\n🚀 Starting training...")
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Training complete. Model saved to {OUTPUT_DIR}")

    # Demo
    print("\n🧪 Quick demo on a validation example...")
    model.eval()
    sample = val_split[0]
    article = sample["document"]
    gold_summary = sample["summary"]

    device = model.device

    inputs = tokenizer(
        "summarize: " + article,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            num_beams=4,
            early_stopping=True,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n--- Article (first 400 chars) ---")
    print(article[:400], "...")
    print("\n--- Reference summary ---")
    print(gold_summary)
    print("\n--- Generated summary ---")
    print(generated)
    print("\nDone.")


if __name__ == "__main__":
    main()
