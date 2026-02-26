# Document Summarizer – NLP Project

A full stack **document summarization web app** built with Flask and modern NLP. It supports extractive methods, a custom fine-tuned **BART abstractive summarizer**, multi-document upload, keyword extraction, and named-entity highlighting.

## 🔍 Features

- **Multiple summarization methods**
  - Frequency-based extractive
  - TextRank extractive
  - Custom **BART** abstractive model (fine-tuned on news-style data)
- **Smart upload flow**
  - Upload one or many files at once
  - Supported formats: `.txt`, `.pdf`, `.docx`
  - Multi-document stats (total words, sentences, avg words per doc)
- **Rich analysis view**
  - Generated summary with compression stats
  - Keyword cloud with scores
  - Named entities grouped by People / Organizations / Locations
  - Clean, responsive UI with print-friendly layout

## 🧱 Architecture

High-level pipeline:

```text
Upload documents
    ↓
Parsing (.txt / .pdf / .docx)
    ↓
Preprocessing (cleaning, tokenization, stopwords)
    ↓
Summarization
    - frequency / textrank (extractive)
    - BART (abstractive, Hugging Face Transformers)
    ↓
Post-processing
    - keyword extraction
    - NER
    ↓
Flask UI rendering (summary + stats + entities)
```

Key modules:

```text
app.py                  # Flask app + routing
preprocess.py           # TextPreprocessor
model.py                # Extractive + ML summarizers
summarize_bart.py       # Wrapper around fine-tuned BART model
document_parser.py      # TXT / PDF / DOCX parsing
textrank.py             # TextRank summarizer
keywords.py             # KeywordExtractor
ner.py                  # Named Entity Recognition
templates/index.html    # Upload & options UI
templates/result.html   # Summary + analysis view
bart-summarizer-final/  # (local fine-tuned BART weights, not checked in)
```

## 🚀 Getting Started

### Requirements

- Python 3.9+  
- `pip`  

### Setup

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd doc-summarizer
```

2. **Create and activate virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download NLTK data (first run only)**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

5. **Add your BART model (optional)**

If you have a fine-tuned BART model, place it in the project root as:

```text
bart-summarizer-final/
    config.json
    model.safetensors (or pytorch_model.bin)
    tokenizer.json
    tokenizer_config.json
```

If you don't have a custom model, the abstractive method will use a pretrained BART model from Hugging Face.

6. **Run the application**

```bash
python app.py
```

7. **Open in browser**

```text
http://localhost:5000
```

## 💡 Usage

### Via Web Interface

1. Go to `http://localhost:5000`
2. Upload one or more `.txt`, `.pdf`, or `.docx` files
3. Choose:
   - Summary method: **Frequency**, **TextRank**, or **Abstractive**
   - Length (fixed sentences or percentage)
4. Click **Summarize**
5. View results:
   - Generated summary
   - Compression and sentence stats
   - Keywords and NER tags
   - Multi-document statistics (if multiple files uploaded)

### Programmatic Usage

```python
from preprocess import TextPreprocessor
from model import ExtractiveSummarizer
from summarize_bart import summarize as bart_summarize

# Initialize
preprocessor = TextPreprocessor()
summarizer = ExtractiveSummarizer()

text = "Your long document here..."

# Extractive summarization
summary_data = summarizer.summarize_text(
    text=text,
    num_sentences=5,
    method="frequency",  # or "tfidf"
    preprocessor=preprocessor
)
print(summary_data["summary"])

# Abstractive summarization (BART)
bart_summary = bart_summarize(text)
print(bart_summary)
```

## 🛠️ Tech Stack

- **Flask** – Web framework
- **NLTK** – Tokenization, stopwords
- **spaCy** – Named Entity Recognition
- **Hugging Face Transformers** – BART model for abstractive summarization
- **scikit-learn** – TF-IDF vectorization
- **PyPDF2 / python-docx** – Document parsing

## 📊 How It Works

### Extractive Methods

**Frequency-based:**
1. Calculate word frequencies (normalized)
2. Score each sentence = sum of word frequencies / sentence length
3. Rank and select top N sentences

**TextRank:**
1. Build graph where sentences are nodes
2. Connect sentences by similarity
3. Run PageRank algorithm
4. Select top-ranked sentences

### Abstractive Method (BART)

1. Fine-tuned BART model on news summarization dataset
2. Takes full document as input
3. Generates new summary text (not just extraction)
4. Produces more human-like, concise summaries

## 🎯 Project Structure

```text
doc-summarizer/
├── app.py                      # Main Flask application
├── preprocess.py               # Text preprocessing
├── model.py                    # Extractive summarization algorithms
├── summarize_bart.py           # BART abstractive summarizer
├── document_parser.py          # File parsing (TXT/PDF/DOCX)
├── textrank.py                 # TextRank algorithm
├── keywords.py                 # Keyword extraction
├── ner.py                      # Named Entity Recognition
├── multilingual.py             # Language detection & support
├── multi_document.py           # Multi-document processing
├── templates/
│   ├── index.html             # Upload interface
│   └── result.html            # Results display
├── uploads/                    # Temporary file storage (gitignored)
├── bart-summarizer-final/     # Fine-tuned BART model (gitignored)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🧪 Testing

Test with a sample document:

```bash
# Create test file
cat > sample.txt << EOF
Artificial intelligence has revolutionized modern technology. Machine learning 
algorithms can now process vast amounts of data efficiently. Deep learning neural 
networks have achieved remarkable success in image recognition. Natural language 
processing enables computers to understand human language. These advancements are 
transforming industries worldwide. Healthcare, finance, and transportation are all 
benefiting from AI innovations. However, ethical considerations must guide AI 
development. Responsible AI practices ensure technology serves humanity positively.
EOF
```

Then upload via web interface and try different methods.

## 🚀 Future Enhancements

- [ ] Add API endpoint for programmatic access
- [ ] Support for more file formats (HTML, Markdown)
- [ ] Batch processing interface
- [ ] Summary comparison tool
- [ ] Export summaries as PDF
- [ ] Deployment guide (Docker, cloud platforms)

## 📄 License

Educational / internship project – use for learning and portfolio purposes.

## 👤 Author

**Shreshth Gupt**  
NLP Document Summarization Project

## 🙏 Acknowledgments

- NLTK documentation and community
- Hugging Face for Transformers library
- Flask documentation
- Various NLP research papers on extractive and abstractive summarization

---

**Built with 🧠 for my NLP project in my AI Internship**
