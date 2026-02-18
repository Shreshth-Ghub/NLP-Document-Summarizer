"""
ml_textrank.py
BERT-Enhanced TextRank Summarization

Combines TextRank graph algorithm with BERT semantic embeddings
for improved extractive summarization.
"""

import numpy as np
import networkx as nx
from bert_similarity import BERTSimilarity

class MLTextRank:
    """TextRank with BERT embeddings for semantic similarity"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize ML-TextRank
        
        Args:
            model_name: BERT model for embeddings
        """
        self.bert = BERTSimilarity(model_name)
    
    def summarize(self, text, num_sentences=5, damping=0.85, max_iter=100):
        """
        Generate summary using BERT-enhanced TextRank
        
        Args:
            text: Input text
            num_sentences: Number of sentences in summary
            damping: Damping factor for PageRank (default 0.85)
            max_iter: Maximum PageRank iterations
        
        Returns:
            Summary text
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Get BERT embeddings
        embeddings = self.bert.encode_sentences(sentences)
        
        # Build similarity matrix using BERT embeddings
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Build graph
        graph = nx.from_numpy_array(similarity_matrix)
        
        # Apply PageRank
        scores = nx.pagerank(
            graph,
            alpha=damping,
            max_iter=max_iter,
            tol=1e-6
        )
        
        # Rank sentences
        ranked_sentences = sorted(
            ((score, idx) for idx, score in scores.items()),
            reverse=True
        )
        
        # Select top sentences
        top_indices = sorted([idx for _, idx in ranked_sentences[:num_sentences]])
        
        # Generate summary in original order
        summary_sentences = [sentences[idx] for idx in top_indices]
        summary = ' '.join(summary_sentences)
        
        return summary
    
    def summarize_by_percentage(self, text, percentage=25):
        """
        Summarize to target percentage of original length
        
        Args:
            text: Input text
            percentage: Target percentage (10, 25, or 50)
        
        Returns:
            Summary text
        """
        sentences = self._split_sentences(text)
        num_sentences = max(1, int(len(sentences) * (percentage / 100)))
        return self.summarize(text, num_sentences=num_sentences)
    
    def _split_sentences(self, text):
        """Split text into sentences"""
        import re
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def get_sentence_scores(self, text):
        """
        Get PageRank scores for all sentences
        
        Args:
            text: Input text
        
        Returns:
            List of tuples (sentence, score)
        """
        sentences = self._split_sentences(text)
        
        if len(sentences) < 2:
            return [(sentences[0] if sentences else "", 1.0)]
        
        # Get BERT embeddings
        embeddings = self.bert.encode_sentences(sentences)
        
        # Build similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Build graph and calculate PageRank
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph, alpha=0.85)
        
        # Return sentence-score pairs
        sentence_scores = [(sentences[idx], scores[idx]) for idx in range(len(sentences))]
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        return sentence_scores


# Convenience function
def ml_textrank_summary(text, num_sentences=5):
    """
    Quick function for ML-TextRank summary
    
    Args:
        text: Input text
        num_sentences: Number of sentences
    
    Returns:
        Summary text
    """
    ml_tr = MLTextRank()
    return ml_tr.summarize(text, num_sentences=num_sentences)


# Testing
if __name__ == "__main__":
    # Test text
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. Colloquially, the term 
    "artificial intelligence" is often used to describe machines that mimic 
    "cognitive" functions that humans associate with the human mind, such as 
    "learning" and "problem solving". As machines become increasingly capable, 
    tasks considered to require "intelligence" are often removed from the 
    definition of AI, a phenomenon known as the AI effect. A quip in Tesler's 
    Theorem says "AI is whatever hasn't been done yet." For instance, optical 
    character recognition is frequently excluded from things considered to be AI, 
    having become a routine technology. Modern machine capabilities generally 
    classified as AI include successfully understanding human speech, competing 
    at the highest level in strategic game systems, autonomously operating cars, 
    intelligent routing in content delivery networks, and military simulations.
    """
    
    print("=== Testing BERT-Enhanced TextRank ===\n")
    print(f"Original text ({len(text.split())} words):")
    print(text[:200] + "...\n")
    
    # Initialize ML-TextRank
    ml_textrank = MLTextRank()
    
    # Generate summary
    print("\n=== Summary (5 sentences) ===")
    summary = ml_textrank.summarize(text, num_sentences=5)
    print(f"Summary ({len(summary.split())} words):")
    print(summary)
    
    # By percentage
    print("\n=== Summary (25% of original) ===")
    summary_25 = ml_textrank.summarize_by_percentage(text, percentage=25)
    print(f"Summary ({len(summary_25.split())} words):")
    print(summary_25)
    
    # Get sentence scores
    print("\n=== Top 3 Sentence Scores ===")
    sentence_scores = ml_textrank.get_sentence_scores(text)
    for i, (sentence, score) in enumerate(sentence_scores[:3], 1):
        print(f"{i}. Score: {score:.4f}")
        print(f"   {sentence[:100]}...")
        print()
