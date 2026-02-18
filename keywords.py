"""
Keyword Extraction Module
Extracts important keywords from documents using multiple algorithms.
"""

import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class KeywordExtractor:
    """Extract keywords from text using various methods."""
    
    def __init__(self, language='english'):
        """
        Initialize keyword extractor.
        
        Args:
            language (str): Language for stopwords
        """
        self.language = language
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words(language))
    
    def extract_frequency_keywords(self, text, top_n=10):
        """
        Extract keywords using word frequency.
        
        Args:
            text (str): Input text
            top_n (int): Number of top keywords to return
            
        Returns:
            list: List of (keyword, score) tuples
        """
        # Tokenize and clean
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and w not in self.stop_words and len(w) > 3]
        
        # Count frequencies
        word_freq = Counter(words)
        
        # Normalize scores
        max_freq = max(word_freq.values()) if word_freq else 1
        normalized_freq = [(word, freq/max_freq) for word, freq in word_freq.items()]
        
        # Sort by frequency
        top_keywords = sorted(normalized_freq, key=lambda x: x[1], reverse=True)[:top_n]
        
        return top_keywords
    
    def extract_tfidf_keywords(self, text, top_n=10):
        """
        Extract keywords using TF-IDF scoring.
        
        Args:
            text (str): Input text
            top_n (int): Number of top keywords to return
            
        Returns:
            list: List of (keyword, score) tuples
        """
        # Split into sentences for TF-IDF
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            # Fall back to frequency if too few sentences
            return self.extract_frequency_keywords(text, top_n)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=100,
            ngram_range=(1, 2)  # Include single words and bigrams
        )
        
        try:
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate average TF-IDF score for each term
            avg_scores = tfidf_matrix.mean(axis=0).A1
            
            # Create keyword-score pairs
            keyword_scores = [(feature_names[i], avg_scores[i]) 
                             for i in range(len(feature_names))]
            
            # Sort by score
            top_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)[:top_n]
            
            return top_keywords
        
        except:
            # Fall back to frequency method if TF-IDF fails
            return self.extract_frequency_keywords(text, top_n)
    
    def extract_rake_keywords(self, text, top_n=10):
        """
        Extract keywords using RAKE-inspired algorithm.
        (Rapid Automatic Keyword Extraction)
        
        Args:
            text (str): Input text
            top_n (int): Number of top keywords to return
            
        Returns:
            list: List of (keyword, score) tuples
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text.lower())
        
        # Split sentences into phrase candidates (on stopwords and punctuation)
        phrase_candidates = []
        for sentence in sentences:
            # Split on stopwords
            words = word_tokenize(sentence)
            current_phrase = []
            
            for word in words:
                if word.isalpha() and word not in self.stop_words and len(word) > 3:
                    current_phrase.append(word)
                else:
                    if current_phrase:
                        phrase_candidates.append(' '.join(current_phrase))
                        current_phrase = []
            
            if current_phrase:
                phrase_candidates.append(' '.join(current_phrase))
        
        # Calculate word scores
        word_freq = Counter()
        word_degree = Counter()
        
        for phrase in phrase_candidates:
            words = phrase.split()
            degree = len(words) - 1
            
            for word in words:
                word_freq[word] += 1
                word_degree[word] += degree
        
        # Calculate word scores (degree / frequency)
        word_scores = {}
        for word in word_freq:
            word_scores[word] = word_degree[word] / word_freq[word] if word_freq[word] > 0 else 0
        
        # Calculate phrase scores
        phrase_scores = []
        for phrase in set(phrase_candidates):
            words = phrase.split()
            score = sum(word_scores.get(word, 0) for word in words)
            phrase_scores.append((phrase, score))
        
        # Sort and return top keywords
        top_keywords = sorted(phrase_scores, key=lambda x: x[1], reverse=True)[:top_n]
        
        # Normalize scores
        max_score = max([score for _, score in top_keywords]) if top_keywords else 1
        normalized_keywords = [(kw, score/max_score) for kw, score in top_keywords]
        
        return normalized_keywords
    
    def extract_keywords(self, text, method='tfidf', top_n=10):
        """
        Extract keywords using specified method.
        
        Args:
            text (str): Input text
            method (str): Method to use ('frequency', 'tfidf', 'rake')
            top_n (int): Number of keywords to extract
            
        Returns:
            dict: Keywords data with method info
        """
        if method == 'frequency':
            keywords = self.extract_frequency_keywords(text, top_n)
        elif method == 'rake':
            keywords = self.extract_rake_keywords(text, top_n)
        else:  # tfidf (default)
            keywords = self.extract_tfidf_keywords(text, top_n)
        
        return {
            'keywords': keywords,
            'method': method,
            'count': len(keywords)
        }