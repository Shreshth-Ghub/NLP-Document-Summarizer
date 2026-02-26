"""
create_summarization_dataset.py
Generate a small generic article summarization dataset
"""

import csv
import os

examples = [
    {
        "text": "Machine learning algorithms are increasingly being used in everyday applications such as recommendation systems, voice assistants, fraud detection, and medical diagnosis. These models learn patterns from historical data and then use those patterns to make predictions on new, unseen data. However, the performance of a model depends heavily on the quality and quantity of the data it is trained on. Poorly curated datasets can lead to biased or inaccurate models, which can have serious consequences when decisions affect finance, health, or safety.",
        "summary": "Machine learning is widely used in daily applications, but model quality strongly depends on the training data. Bad datasets can create biased or inaccurate systems with serious real-world impact."
    },
    {
        "text": "Natural language processing, or NLP, focuses on enabling computers to understand and generate human language. Common NLP tasks include tokenization, part-of-speech tagging, named entity recognition, sentiment analysis, and summarization. Recent advances in transformer-based architectures, such as BERT and T5, have dramatically improved performance on many of these tasks. Despite these advances, NLP systems still struggle with ambiguity, sarcasm, domain-specific jargon, and low-resource languages.",
        "summary": "NLP aims to help computers understand and generate human language using tasks like tagging, NER, and summarization. Transformer models such as BERT and T5 improved results, but challenges like ambiguity and low-resource languages remain."
    },
    {
        "text": "Text summarization techniques can be broadly divided into extractive and abstractive methods. Extractive methods select important sentences directly from the original document, while abstractive methods generate new sentences that capture the meaning of the text. Extractive approaches are often simpler and more stable because they reuse original phrasing, but they may produce choppy summaries. Abstractive models can create more fluent summaries but usually require more data and computational resources to train effectively.",
        "summary": "Text summarization can be extractive, selecting key sentences, or abstractive, generating new phrasing. Extractive methods are simpler but sometimes choppy, while abstractive approaches are more fluent yet harder to train."
    },
    {
        "text": "Evaluation of automatic summaries is commonly done using metrics such as ROUGE, which compare system-generated summaries to human-written reference summaries. ROUGE measures lexical overlap, counting matching unigrams, bigrams, or longest common subsequences. While ROUGE is easy to compute and widely adopted, it does not directly measure factual consistency, readability, or user satisfaction. As a result, many research works also include human evaluation to assess summary usefulness and quality.",
        "summary": "Automatic summaries are often evaluated with ROUGE, which measures word overlap with reference texts. ROUGE is convenient but does not fully capture factual accuracy or readability, so human evaluation is still important."
    },
    {
        "text": "Building a practical document summarization system for end users requires more than just a good model. The system must handle different file formats such as PDF, Word, and plain text, perform preprocessing like cleaning and sentence segmentation, and provide a clear interface for users to upload documents and view results. Additional features such as keyword extraction, language detection, multi-document support, and named entity recognition can significantly improve usefulness in real-world scenarios.",
        "summary": "A real-world summarization tool needs preprocessing, support for multiple file formats, and a user-friendly interface. Extra features like keyword extraction, language detection, and NER make the system more useful."
    },
    {
        "text": "When training text models, splitting data into training, validation, and test sets is essential for reliable evaluation. The training set is used to fit the model parameters, the validation set guides hyperparameter tuning and early stopping, and the test set provides an unbiased estimate of final performance. Data leakage, where information from the test set influences training decisions, can lead to overly optimistic results and should be carefully avoided.",
        "summary": "Text models should be trained with separate training, validation, and test sets to avoid data leakage. The validation set helps tune hyperparameters, and the test set gives an unbiased final performance estimate."
    },
    {
        "text": "Modern deep learning libraries such as PyTorch and TensorFlow have made it much easier to experiment with complex neural architectures. They provide automatic differentiation, GPU acceleration, and rich ecosystems of prebuilt models. However, they also introduce new engineering challenges, including reproducibility, dependency management, and hardware compatibility issues, especially on Windows where GPU drivers and DLL libraries can be fragile.",
        "summary": "Frameworks like PyTorch and TensorFlow simplify building deep learning models but add challenges in reproducibility, dependencies, and hardware compatibility, particularly with GPUs and DLLs on Windows."
    },
    {
        "text": "For small student projects, it is often more practical to fine-tune an existing pretrained model on a modest custom dataset than to train a large model from scratch. Fine-tuning reuses the knowledge stored in the pretrained weights and adapts it to the target domain. This approach typically requires less data and computing power while still demonstrating understanding of the end-to-end machine learning workflow.",
        "summary": "Student projects usually benefit from fine-tuning pretrained models on small custom datasets instead of training from scratch. Fine-tuning needs less data and compute but still shows full ML workflow understanding."
    }
]

def main():
    os.makedirs("data", exist_ok=True)
    path = "data/news_summaries.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "summary"])
        writer.writeheader()
        writer.writerows(examples)
    print(f"Dataset written to {path} with {len(examples)} examples.")

if __name__ == "__main__":
    main()
