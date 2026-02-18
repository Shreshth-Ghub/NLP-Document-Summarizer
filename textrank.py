"""
Graph-based TextRank Summarization Module
Implements PageRank-style algorithm for sentence ranking.
"""

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re


class TextRankSummarizer:
    """TextRank algorithm for extractive summarization."""
    
    def __init__(self, damping_factor=0.85, threshold=0.0001, max_iterations=100):
        """
        Initialize TextRank summarizer.
        
        Args:
            damping_factor (float): PageRank damping factor (typically 0.85)
            threshold (float): Convergence threshold
            max_iterations (int): Maximum iterations for PageRank
        """
        self.damping_factor = damping_factor
        self.threshold = threshold
        self.max_iterations = max_iterations
    
    def _split_sentences(self, text):
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences
    
    def _build_similarity_matrix(self, sentences):
        """
        Build sentence similarity matrix using TF-IDF and cosine similarity.
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            numpy.ndarray: Similarity matrix
        """
        if len(sentences) < 2:
            return np.zeros((len(sentences), len(sentences)))
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            return similarity_matrix
        except Exception:
            # Fallback if TF-IDF fails
            return np.zeros((len(sentences), len(sentences)))
    
    def _pagerank(self, similarity_matrix):
        """
        Apply PageRank algorithm to similarity matrix.
        
        Args:
            similarity_matrix (numpy.ndarray): Sentence similarity matrix
            
        Returns:
            numpy.ndarray: PageRank scores for each sentence
        """
        n = similarity_matrix.shape[0]
        
        if n == 0:
            return np.array([])
        
        # Normalize similarity matrix to create transition matrix
        row_sums = similarity_matrix.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        transition_matrix = similarity_matrix / row_sums[:, np.newaxis]
        
        # Initialize PageRank scores
        scores = np.ones(n) / n
        
        # Iterate until convergence
        for _ in range(self.max_iterations):
            new_scores = (1 - self.damping_factor) / n + \
                         self.damping_factor * transition_matrix.T.dot(scores)
            
            # Check convergence
            if np.linalg.norm(new_scores - scores) < self.threshold:
                break
            
            scores = new_scores
        
        return scores
    
    def summarize(self, text, num_sentences=5):
        """
        Generate summary using TextRank.
        
        Args:
            text (str): Input text
            num_sentences (int): Number of sentences in summary
            
        Returns:
            dict: Summary data
        """
        sentences = self._split_sentences(text)
        
        if not sentences:
            return {
                'summary': '',
                'summary_sentences': [],
                'scores': [],
                'method': 'textrank',
                'original_sentences': 0,
                'compression_ratio': 0.0
            }
        
        # Limit number of sentences to available
        num_sentences = min(num_sentences, len(sentences))
        
        if len(sentences) == 1:
            return {
                'summary': sentences[0],
                'summary_sentences': [sentences[0]],
                'scores': [1.0],
                'method': 'textrank',
                'original_sentences': 1,
                'compression_ratio': 1.0
            }
        
        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(sentences)
        
        # Apply PageRank
        scores = self._pagerank(similarity_matrix)
        
        # Rank sentences by score
        ranked_indices = np.argsort(scores)[::-1]
        
        # Select top sentences
        top_indices = sorted(ranked_indices[:num_sentences])
        summary_sentences = [sentences[i] for i in top_indices]
        summary_scores = [scores[i] for i in top_indices]
        
        # Create summary text (preserve original order)
        summary = '. '.join(summary_sentences) + '.'
        
        return {
            'summary': summary,
            'summary_sentences': summary_sentences,
            'scores': summary_scores,
            'method': 'textrank',
            'original_sentences': len(sentences),
            'compression_ratio': len(summary_sentences) / len(sentences) if len(sentences) > 0 else 0
        }
