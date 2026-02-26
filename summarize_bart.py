from transformers import BartTokenizerFast, BartForConditionalGeneration
import torch

MODEL_DIR = "bart-summarizer-final"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BartTokenizerFast.from_pretrained(MODEL_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()


def summarize(text, max_input_len=512, max_summary_len=220, min_summary_len=120):
    inputs = tokenizer(
        text,
        max_length=max_input_len,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    summary_ids = model.generate(
        **inputs,
        max_length=max_summary_len,
        min_length=min_summary_len,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    sample = """" US Secretary of Defense Pete Hegseth vowed to remove Anthropic from his agency's supply chain if the company declined to allow its artificial intelligence (AI) technology to be used across military applications.

The threat was issued on Tuesday at a Pentagon meeting that Hegseth had demanded with Anthropic boss Dario Amodei, a source familiar with discussions told the BBC.

"We continued good-faith conversations about our usage policy to ensure Anthropic can continue to support the government's national security mission in line with what our models can reliably and responsibly do," Anthropic said in a statement.

A senior Pentagon official said Anthropic had until Friday evening to comply.

A source told the BBC the tone of the discussion between Hegseth and Amodei was cordial, but Amodei laid out what Anthropic considers to be its red lines.

These include involvement in autonomous kinetic operations in which AI tools make final military targeting decisions without human intervention.

The use of Anthropic tools for mass domestic surveillance constitutes another red line, the source said.

But the Pentagon official told the BBC the current conflict between the agency and Anthropic is unrelated to the use of autonomous weapons or mass surveillance.

The official said that if Anthropic did not comply, Hegseth would ensure the Defense Production Act was invoked on the company.

That measure could compel Anthropic executives to allow unrestricted use by the Pentagon on national security grounds.

The official added that the Pentagon would simultaneously label Anthropic as a supply chain risk.

An Anthropic spokesperson said Amodei "expressed appreciation for the Department's work and thanked the Secretary for his service" during the meeting with Hegseth.

Anthropic is the maker of the AI chatbot Claude and was one of four AI companies to be awarded contracts with the Pentagon last summer.

Google, ChatGPT-maker OpenAI and Elon Musk's xAI which makes the AI chatbot Grok were also awarded contracts of up to $200m (£148m) each.

Defence department official Emil Michael has previously said the agency wants OpenAI, Google, xAI, and Anthropic to allow the Pentagon to "be able to use any model for all lawful use cases."

Anthropic has consistently aimed to position itself as a more safety-orientated approach to AI research as compared to rivals.

It regularly shares safety reports on its own products with the public.

One such report from last year acknowledged its AI technology had been "weaponised" by hackers who used it to conduct sophisticated cyber-attacks.

The company's image was challenged after reports that the US military used its AI model Claude during the operation that led to the capture former Venezuelan President Nicolás Maduro in January.

Anthropic was the first tech company approved to work in the Pentagon's classified military networks and has partnerships with companies including Palantir.

Sources tell the BBC that the Claude model was used in the Maduro operation through a contract with Palantir.

The Pentagon's position is that Anthropic should have no say in how the Pentagon uses its products.

Observers say the current spat between Anthropic and the Pentagon has resulted from a breach of trust between the two sides.

"They need to get to a resolution," according to Emelia Probasco, Senior Fellow at Georgetown University's Center for Security and Emerging Technology.

"In my opinion, we should be giving the people we ask to serve every possible advantage. We owe it to them to figure this out," Probasco said."""
    print(summarize(sample))
