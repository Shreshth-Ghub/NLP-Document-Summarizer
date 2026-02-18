"""
bert_similarity.py
BERT Sentence Embeddings for Semantic Similarity

Uses sentence-transformers to generate BERT-based embeddings
for semantic sentence similarity calculation.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BERTSimilarity:
    """BERT-based semantic similarity calculator"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize BERT model
        
        Args:
            model_name: Pre-trained sentence transformer model
                       'all-MiniLM-L6-v2' (default): 80MB, fast, good quality
                       'all-mpnet-base-v2': 420MB, slower, better quality
        """
        print(f"Loading BERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("BERT model loaded successfully!")
    
    def encode_sentences(self, sentences, batch_size=32):
        """
        Encode sentences into BERT embeddings
        
        Args:
            sentences: List of sentences
            batch_size: Number of sentences to encode at once
        
        Returns:
            numpy array of shape (n_sentences, embedding_dim)
        """
        if not sentences:
            return np.array([])
        
        # Generate embeddings
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def calculate_similarity_matrix(self, sentences):
        """
        Calculate pairwise semantic similarity between sentences
        
        Args:
            sentences: List of sentences
        
        Returns:
            numpy array of shape (n_sentences, n_sentences)
            containing cosine similarity scores
        """
        if len(sentences) < 2:
            return np.array([[1.0]])
        
        # Get BERT embeddings
        embeddings = self.encode_sentences(sentences)
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    
    def get_most_similar(self, query_sentence, candidate_sentences, top_k=5):
        """
        Find most similar sentences to a query
        
        Args:
            query_sentence: The query sentence
            candidate_sentences: List of candidate sentences
            top_k: Number of top similar sentences to return
        
        Returns:
            List of tuples (sentence, similarity_score)
        """
        if not candidate_sentences:
            return []
        
        # Encode query and candidates
        all_sentences = [query_sentence] + candidate_sentences
        embeddings = self.encode_sentences(all_sentences)
        
        # Query embedding
        query_embedding = embeddings[0:1]
        
        # Candidate embeddings
        candidate_embeddings = embeddings[1:]
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (candidate_sentences[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results


# Convenience functions
def get_bert_embeddings(sentences, model_name='all-MiniLM-L6-v2'):
    """
    Quick function to get BERT embeddings
    
    Args:
        sentences: List of sentences
        model_name: Model name
    
    Returns:
        numpy array of embeddings
    """
    bert = BERTSimilarity(model_name)
    return bert.encode_sentences(sentences)


def calculate_bert_similarity(sentence1, sentence2, model_name='all-MiniLM-L6-v2'):
    """
    Calculate semantic similarity between two sentences
    
    Args:
        sentence1: First sentence
        sentence2: Second sentence
        model_name: Model name
    
    Returns:
        float: similarity score (0-1)
    """
    bert = BERTSimilarity(model_name)
    embeddings = bert.encode_sentences([sentence1, sentence2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)


# Testing
if __name__ == "__main__":
    # Test BERT similarity
    bert = BERTSimilarity()
    
    # Test sentences
    sentences = [
        "The cat sat on the mat",
        "A feline rested on a rug",
        "The dog played in the park",
        "Python is a programming language"
    ]
    
    print("\n=== Testing BERT Embeddings ===")
    embeddings = bert.encode_sentences(sentences)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    print("\n=== Similarity Matrix ===")
    sim_matrix = bert.calculate_similarity_matrix(sentences)
    print("Pairwise similarities:")
    for i, sent1 in enumerate(sentences):
        for j, sent2 in enumerate(sentences):
            if i < j:  # Only upper triangle
                print(f"\n'{sent1}'")
                print(f"'{sent2}'")
                print(f"Similarity: {sim_matrix[i][j]:.3f}")
    
    print("\n=== Most Similar Sentences ===")
    query = "A cat is sitting"
    similar = bert.get_most_similar(query, sentences, top_k=3)
    print(f"Query: '{query}'")
    print("\nMost similar:")
    for sent, score in similar:
        print(f"{score:.3f} - {sent}")
