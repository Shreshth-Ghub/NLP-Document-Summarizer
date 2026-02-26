import argparse
import os

from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--dataset_name", type=str, default="allenai/multi_lexsum")
    parser.add_argument("--dataset_config", type=str, default="v20230518")
    parser.add_argument("--text_field", type=str, default="sources")
    parser.add_argument("--summary_field", type=str, default="summary/long")
    parser.add_argument("--output_dir", type=str, default="./bart_multilexsum_model")
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num_train_samples", type=int, default=-1)
    parser.add_argument("--num_eval_samples", type=int, default=1000)
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer and model...")
    tokenizer = BartTokenizerFast.from_pretrained(args.base_model)
    model = BartForConditionalGeneration.from_pretrained(args.base_model)

    print("Loading dataset...")
    raw_datasets = load_dataset(args.dataset_name, name=args.dataset_config)

    # For Multi-LexSum, "sources" is a list of documents per case.
    # Concatenate them into a single string.
    def concat_sources(example):
        if isinstance(example[args.text_field], list):
            example["text"] = "\n\n".join(example[args.text_field])
        else:
            example["text"] = example[args.text_field]
        example["summary"] = example[args.summary_field]
        return example

    raw_datasets = raw_datasets.map(concat_sources)

    if args.num_train_samples > 0:
        raw_datasets["train"] = raw_datasets["train"].select(range(min(args.num_train_samples, len(raw_datasets["train"]))))

    if args.num_eval_samples > 0 and "validation" in raw_datasets:
        raw_datasets["validation"] = raw_datasets["validation"].select(
            range(min(args.num_eval_samples, len(raw_datasets["validation"])))
        )

    def preprocess_function(examples):
        inputs = examples["text"]
        targets = examples["summary"]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            truncation=True,
            padding="max_length",
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=args.max_target_length,
                truncation=True,
                padding="max_length",
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        fp16=args.fp16,
        warmup_ratio=args.warmup_ratio,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation", None),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    print("Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
