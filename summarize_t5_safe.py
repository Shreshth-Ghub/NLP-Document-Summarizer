import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_DIR = r"C:\Users\Shreshth Gupt\Downloads\t5_xsum_model_safe\models\t5-xsum"

tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
model.eval()

def summarize(text, max_input_len=512, max_output_len=80):
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
            min_length=20,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    text = """
    Bill Gates will not deliver his keynote address at the India AI Impact Summit in Delhi, his philanthropic organisation said hours before the Microsoft co-founder was due to speak.
The Gates Foundation said the decision was made after "careful consideration" and "to ensure the focus remains on the [summit's] key priorities", but did not elaborate.
Gates's withdrawal comes amid a controversy over his ties to the late sex offender Jeffrey Epstein after he was named in new files released by the US Department of Justice in January.
Gates's spokesperson has called the claims in the files "absolutely absurd and completely false", and the billionaire has said he regretted spending time with Epstein.
Gates has not been accused of wrongdoing by any of Epstein's victims and the appearance of his name in the files does not imply criminal activity of any kind.
The Gates Foundation said Ankur Vora, president of its Africa and India offices, would speak at the summit instead of Gates.
The organisation added that it remained "fully committed" to its work in India to advance "shared health and development goals".
Gates's decision to not speak to the summit came after days of uncertainty over whether he would attend.
He is currently in India and had visited the southern state of Andhra Pradesh on Monday, where he reportedly discussed initiatives for boosting health, agriculture, education and technology.
    """
    print("Summary:\n", summarize(text))
