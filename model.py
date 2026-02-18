"""
Extractive Summarization Model
Uses TF-IDF scoring to rank and select important sentences.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

class ExtractiveSummarizer:
    """TF-IDF based extractive text summarization."""
    
    def __init__(self):
        """Initialize the summarizer."""
        self.vectorizer = TfidfVectorizer()
    
    def calculate_word_frequency(self, words):
        """
        Calculate word frequency distribution.
        
        Args:
            words (list): List of words
            
        Returns:
            dict: Word frequency mapping
        """
        word_freq = Counter(words)
        max_freq = max(word_freq.values()) if word_freq else 1
        
        # Normalize frequencies
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        return word_freq
    
    def score_sentences_frequency(self, sentences, sentence_words, all_words):
        """
        Score sentences based on word frequency.
        
        Args:
            sentences (list): List of original sentences
            sentence_words (list): List of tokenized words per sentence
            all_words (list): All words from document
            
        Returns:
            dict: Sentence scores
        """
        # Calculate word frequencies
        word_freq = self.calculate_word_frequency(all_words)
        
        # Score each sentence
        sentence_scores = {}
        for i, words in enumerate(sentence_words):
            score = sum(word_freq.get(word, 0) for word in words)
            # Normalize by sentence length to avoid bias toward long sentences
            sentence_scores[i] = score / len(words) if words else 0
        
        return sentence_scores
    
    def score_sentences_tfidf(self, sentences):
        """
        Score sentences using TF-IDF vectorization.
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            dict: Sentence scores
        """
        if len(sentences) < 2:
            return {0: 1.0}
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores as sum of TF-IDF values
        sentence_scores = {}
        for i in range(len(sentences)):
            sentence_scores[i] = np.sum(tfidf_matrix[i].toarray())
        
        return sentence_scores
    
    def get_summary(self, preprocessed_data, num_sentences=5, method='frequency'):
        """
        Generate extractive summary by selecting top-scored sentences.
        
        Args:
            preprocessed_data (dict): Output from TextPreprocessor
            num_sentences (int): Number of sentences in summary
            method (str): Scoring method ('frequency' or 'tfidf')
            
        Returns:
            dict: Summary information
        """
        sentences = preprocessed_data['sentences']
        sentence_words = preprocessed_data['sentence_words']
        all_words = preprocessed_data['all_words']
        
        # Handle edge cases
        if len(sentences) == 0:
            return {
                'summary': '',
                'summary_sentences': [],
                'num_sentences': 0,
                'compression_ratio': 0
            }
        
        # Limit summary length to available sentences
        num_sentences = min(num_sentences, len(sentences))
        
        # Score sentences based on chosen method
        if method == 'tfidf':
            sentence_scores = self.score_sentences_tfidf(sentences)
        else:  # frequency
            sentence_scores = self.score_sentences_frequency(
                sentences, sentence_words, all_words
            )
        
        # Select top N sentences
        ranked_sentences = sorted(
            sentence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_sentences]
        
        # Sort selected sentences by original order for coherence
        selected_indices = sorted([idx for idx, score in ranked_sentences])
        summary_sentences = [sentences[i] for i in selected_indices]
        
        # Create summary text
        summary = ' '.join(summary_sentences)
        
        # Calculate compression ratio
        compression_ratio = len(summary_sentences) / len(sentences)
        
        return {
            'summary': summary,
            'summary_sentences': summary_sentences,
            'num_sentences': len(summary_sentences),
            'original_sentences': len(sentences),
            'compression_ratio': compression_ratio,
            'method': method
        }
    
    def summarize_text(self, text, num_sentences=5, method='frequency', preprocessor=None):
        """
        End-to-end summarization from raw text.
        
        Args:
            text (str): Raw input text
            num_sentences (int): Number of sentences in summary
            method (str): Scoring method ('frequency' or 'tfidf')
            preprocessor: TextPreprocessor instance (optional)
            
        Returns:
            dict: Summary with metadata
        """
        # Import here to avoid circular dependency
        if preprocessor is None:
            from preprocess import TextPreprocessor
            preprocessor = TextPreprocessor()
        
        # Preprocess document
        preprocessed_data = preprocessor.preprocess_document(text)
        
        # Generate summary
        summary_data = self.get_summary(
            preprocessed_data,
            num_sentences=num_sentences,
            method=method
        )
        
        # Add preprocessing metadata
        summary_data['original_text_length'] = len(text)
        summary_data['summary_length'] = len(summary_data['summary'])
        summary_data['word_count_original'] = preprocessed_data['num_words']
        
        return summary_data