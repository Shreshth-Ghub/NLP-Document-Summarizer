import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# use original HF model instead of your corrupted checkpoint
MODEL_NAME = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()


def summarize(text, max_input_len=512, max_output_len=512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        max_length=max_input_len,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_output_len,
            do_sample=True,
            top_k=50, 
            top_p=0.9,
            temperature=0.9,
            no_repeat_ngram_size=3,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    article = (
        "Galgotias University faced criticism after showing a Chinese robot dog "
        "as their own invention at an AI summit in Delhi."
    )
    print("Summary:\n", summarize(article))
