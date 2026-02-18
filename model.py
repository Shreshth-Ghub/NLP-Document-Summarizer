"""
model.py - Phase 6 Complete with 5 Methods

Summarization algorithms:
1. Word Frequency (basic)
2. TF-IDF (statistical)
3. TextRank (graph-based)
4. BERT TextRank (semantic graph) - NEW
5. T5 Abstractive (generative) - NEW
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Phase 6 imports
try:
    from ml_textrank import MLTextRank
    ML_TEXTRANK_AVAILABLE = True
except ImportError:
    ML_TEXTRANK_AVAILABLE = False
    print("BERT TextRank not available. Install sentence-transformers.")

try:
    from abstractive import AbstractiveSummarizer
    ABSTRACTIVE_AVAILABLE = True
except ImportError:
    ABSTRACTIVE_AVAILABLE = False
    print("Abstractive summarization not available. Install transformers.")


class TextSummarizer:
    """Handles multiple summarization methods"""
    
    def __init__(self, language='english'):
        """
        Initialize summarizer
        
        Args:
            language: Language for stopwords
        """
        self.language = language
        
        # Initialize ML models lazily (only when needed)
        self._ml_textrank = None
        self._abstractive = None
    
    def summarize(self, text, method='frequency', num_sentences=5):
        """
        Generate summary using specified method
        
        Args:
            text: Input text
            method: Summarization method
                   'frequency' - Word frequency
                   'tfidf' - TF-IDF scoring
                   'textrank' - Graph-based PageRank
                   'bert_textrank' - BERT + PageRank (NEW)
                   'abstractive' - T5 generative (NEW)
            num_sentences: Number of sentences in summary
        
        Returns:
            Summary text
        """
        if method == 'frequency':
            return self.frequency_based_summary(text, num_sentences)
        elif method == 'tfidf':
            return self.tfidf_based_summary(text, num_sentences)
        elif method == 'textrank':
            return self.textrank_summary(text, num_sentences)
        elif method == 'bert_textrank':
            return self.bert_textrank_summary(text, num_sentences)
        elif method == 'abstractive':
            return self.abstractive_summary(text, num_sentences)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def frequency_based_summary(self, text, num_sentences=5):
        """Word frequency-based extractive summarization"""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Tokenize and remove stopwords
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words(self.language))
        words = [w for w in words if w.isalnum() and w not in stop_words]
        
        # Calculate word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Normalize frequencies
        max_freq = max(word_freq.values()) if word_freq else 1
        for word in word_freq:
            word_freq[word] /= max_freq
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words_in_sentence = word_tokenize(sentence.lower())
            score = sum(word_freq.get(w, 0) for w in words_in_sentence)
            sentence_scores[i] = score
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), 
                              key=lambda x: x[1], 
                              reverse=True)[:num_sentences]
        top_sentences = sorted(top_sentences, key=lambda x: x[0])
        
        summary = ' '.join([sentences[i] for i, _ in top_sentences])
        return summary
    
    def tfidf_based_summary(self, text, num_sentences=5):
        """TF-IDF based extractive summarization"""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words=self.language)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores (sum of TF-IDF values)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Select top sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices = sorted(top_indices)
        
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    
    def textrank_summary(self, text, num_sentences=5):
        """TextRank graph-based summarization (traditional)"""
        try:
            from textrank import textrank_summarize
            return textrank_summarize(text, num_sentences)
        except ImportError:
            print("TextRank module not found, falling back to TF-IDF")
            return self.tfidf_based_summary(text, num_sentences)
    
    def bert_textrank_summary(self, text, num_sentences=5):
        """
        BERT-enhanced TextRank (NEW)
        Uses semantic embeddings instead of TF-IDF
        """
        if not ML_TEXTRANK_AVAILABLE:
            print("BERT TextRank not available, falling back to regular TextRank")
            return self.textrank_summary(text, num_sentences)
        
        try:
            # Lazy initialization
            if self._ml_textrank is None:
                self._ml_textrank = MLTextRank()
            
            return self._ml_textrank.summarize(text, num_sentences=num_sentences)
        
        except Exception as e:
            print(f"BERT TextRank error: {e}, falling back to TextRank")
            return self.textrank_summary(text, num_sentences)
    
    def abstractive_summary(self, text, num_sentences=5):
        """
        T5 Abstractive summarization (NEW)
        Generates new sentences instead of extracting
        """
        if not ABSTRACTIVE_AVAILABLE:
            print("Abstractive summarization not available, falling back to TextRank")
            return self.textrank_summary(text, num_sentences)
        
        try:
            # Lazy initialization
            if self._abstractive is None:
                self._abstractive = AbstractiveSummarizer(model_name='t5-small')
            
            return self._abstractive.summarize_by_sentences(text, num_sentences=num_sentences)
        
        except Exception as e:
            print(f"Abstractive summarization error: {e}, falling back to TextRank")
            return self.textrank_summary(text, num_sentences)


class ExtractiveSummarizer:
    """Legacy class for backward compatibility with existing app"""
    
    def __init__(self):
        self.summarizer = TextSummarizer()
    
    def summarize_text(self, text, num_sentences=5, method='frequency', preprocessor=None):
        """
        Generate summary with legacy interface
        
        Args:
            text: Input text
            num_sentences: Number of sentences
            method: Summarization method
            preprocessor: Ignored (for compatibility)
        
        Returns:
            dict: Summary data with all required fields
        """
        summary = self.summarizer.summarize(text, method, num_sentences)
        
        sentences = sent_tokenize(text)
        summary_sentences = sent_tokenize(summary)
        
        return {
            'summary': summary,
            'summary_sentences': summary_sentences,
            'num_sentences': len(summary_sentences),
            'original_sentences': len(sentences),
            'compression_ratio': len(summary) / len(text) if len(text) > 0 else 0,
            'method': method,
            'word_count_original': len(text.split()),
            'original_text_length': len(text),
            'summary_length': len(summary)
        }


# Convenience functions
def summarize_text(text, method='frequency', num_sentences=5, language='english'):
    """
    Quick function to summarize text
    
    Args:
        text: Input text
        method: Summarization method
        num_sentences: Number of sentences
        language: Language for stopwords
    
    Returns:
        Summary text
    """
    summarizer = TextSummarizer(language)
    return summarizer.summarize(text, method, num_sentences)


def get_available_methods():
    """
    Get list of available summarization methods
    
    Returns:
        dict: Method names and availability
    """
    methods = {
        'frequency': {'available': True, 'name': 'Word Frequency'},
        'tfidf': {'available': True, 'name': 'TF-IDF'},
        'textrank': {'available': True, 'name': 'TextRank'},
        'bert_textrank': {'available': ML_TEXTRANK_AVAILABLE, 'name': 'BERT TextRank'},
        'abstractive': {'available': ABSTRACTIVE_AVAILABLE, 'name': 'T5 Abstractive'}
    }
    return methods


# Testing
if __name__ == "__main__":
    test_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. Modern machine capabilities 
    generally classified as AI include successfully understanding human speech, 
    competing at the highest level in strategic game systems, autonomously 
    operating cars, and intelligent routing in content delivery networks.
    """
    
    print("=== Testing Updated Summarization Methods ===\n")
    
    summarizer = TextSummarizer()
    
    methods = ['frequency', 'tfidf', 'textrank', 'bert_textrank', 'abstractive']
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method.upper()}")
        print('='*60)
        
        try:
            summary = summarizer.summarize(test_text, method=method, num_sentences=3)
            print(f"Summary ({len(summary.split())} words):")
            print(summary)
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n\n=== Available Methods ===")
    available = get_available_methods()
    for method_id, info in available.items():
        status = "✓" if info['available'] else "✗"
        print(f"{status} {info['name']} ({method_id})")