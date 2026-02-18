"""
supervised_learning.py
Supervised Learning for Summarization

Train models on labeled summary data to learn what makes good summaries.
Includes BERT-based sentence ranking and regression models.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from bert_similarity_disabled import BERTSimilarity
import pickle
import os

class SupervisedSummarizer:
    """
    Supervised learning for extractive summarization.
    
    Trains a model to predict importance scores for sentences
    based on features like position, length, BERT embeddings, etc.
    """
    
    def __init__(self):
        """Initialize supervised summarizer"""
        self.model = None
        self.bert = None
        self.feature_names = []
    
    def extract_features(self, sentences, document_text):
        """
        Extract features for each sentence
        
        Args:
            sentences: List of sentences
            document_text: Full document text
        
        Returns:
            numpy array of features (n_sentences, n_features)
        """
        features = []
        total_sentences = len(sentences)
        
        # Get BERT embeddings if available
        if self.bert is None:
            try:
                self.bert = BERTSimilarity()
                embeddings = self.bert.encode_sentences(sentences)
            except Exception as e:
                print(f"BERT unavailable, using basic features: {e}")
                embeddings = None
        else:
            embeddings = self.bert.encode_sentences(sentences)
        
        for i, sentence in enumerate(sentences):
            sent_features = []
            
            # Positional features
            sent_features.append(i / total_sentences)  # Relative position
            sent_features.append(1 if i < 3 else 0)    # Is in first 3 sentences
            sent_features.append(1 if i >= total_sentences - 3 else 0)  # Is in last 3
            
            # Length features
            words = sentence.split()
            sent_features.append(len(words))  # Word count
            sent_features.append(len(sentence))  # Character count
            sent_features.append(len(words) / (document_text.count('.') + 1))  # Normalized length
            
            # Content features
            sent_features.append(sum(1 for w in words if w[0].isupper()))  # Proper nouns count
            sent_features.append(sum(1 for c in sentence if c.isdigit()))  # Digit count
            sent_features.append(1 if '?' in sentence or '!' in sentence else 0)  # Has question/exclamation
            
            # BERT embedding features (if available)
            if embeddings is not None:
                # Mean of embedding dimensions as features (reduced to 10 for efficiency)
                emb_features = embeddings[i][:10]
                sent_features.extend(emb_features.tolist())
            
            features.append(sent_features)
        
        return np.array(features)
    
    def prepare_training_data(self, documents_with_summaries):
        """
        Prepare training data from documents and their gold summaries
        
        Args:
            documents_with_summaries: List of dicts with keys:
                - 'document': Full document text
                - 'summary': Gold standard summary
                - 'sentences': List of sentences in document
        
        Returns:
            X: Feature matrix
            y: Target scores (1 if sentence in summary, 0 otherwise)
        """
        all_features = []
        all_labels = []
        
        for item in documents_with_summaries:
            sentences = item['sentences']
            document = item['document']
            gold_summary = item['summary']
            
            # Extract features
            features = self.extract_features(sentences, document)
            
            # Create labels: 1 if sentence appears in gold summary, 0 otherwise
            labels = []
            for sentence in sentences:
                # Simple matching: check if sentence (or most of it) appears in summary
                sentence_words = set(sentence.lower().split())
                summary_words = set(gold_summary.lower().split())
                
                # Jaccard similarity
                if len(sentence_words) > 0:
                    overlap = len(sentence_words & summary_words) / len(sentence_words)
                    labels.append(1 if overlap > 0.5 else 0)
                else:
                    labels.append(0)
            
            all_features.append(features)
            all_labels.extend(labels)
        
        X = np.vstack(all_features)
        y = np.array(all_labels)
        
        return X, y
    
    def train(self, documents_with_summaries, test_size=0.2):
        """
        Train the supervised model
        
        Args:
            documents_with_summaries: Training data
            test_size: Fraction for test set
        
        Returns:
            dict: Training results with metrics
        """
        print(f"Preparing training data from {len(documents_with_summaries)} documents...")
        X, y = self.prepare_training_data(documents_with_summaries)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Labels distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        print("Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        print(f"\nTraining Results:")
        print(f"Train MSE: {train_mse:.4f}")
        print(f"Test MSE:  {test_mse:.4f}")
        print(f"Test MAE:  {test_mae:.4f}")
        
        return results
    
    def summarize(self, text, num_sentences=5):
        """
        Generate summary using trained model
        
        Args:
            text: Input text
            num_sentences: Number of sentences in summary
        
        Returns:
            Summary text
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Split into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= num_sentences:
            return text
        
        # Extract features
        features = self.extract_features(sentences, text)
        
        # Predict importance scores
        scores = self.model.predict(features)
        
        # Select top sentences
        top_indices = np.argsort(scores)[::-1][:num_sentences]
        top_indices = sorted(top_indices)  # Maintain original order
        
        summary_sentences = [sentences[i] for i in top_indices]
        summary = ' '.join(summary_sentences)
        
        return summary
    
    def save_model(self, filepath='supervised_summarizer.pkl'):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='supervised_summarizer.pkl'):
        """Load trained model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Example training data format
    training_data = [
        {
            'document': """
                Artificial intelligence (AI) is transforming industries worldwide.
                Companies are investing heavily in AI research and development.
                Machine learning algorithms can now perform complex tasks.
                Deep learning has revolutionized computer vision and NLP.
                However, ethical concerns about AI remain significant.
                Bias in training data can lead to unfair outcomes.
                Transparency and explainability are crucial for AI systems.
                Governments are developing regulations for AI technologies.
            """,
            'summary': """
                Artificial intelligence is transforming industries with heavy investment.
                Deep learning has revolutionized computer vision and NLP.
                Ethical concerns and bias remain significant challenges.
            """,
            'sentences': [
                "Artificial intelligence (AI) is transforming industries worldwide.",
                "Companies are investing heavily in AI research and development.",
                "Machine learning algorithms can now perform complex tasks.",
                "Deep learning has revolutionized computer vision and NLP.",
                "However, ethical concerns about AI remain significant.",
                "Bias in training data can lead to unfair outcomes.",
                "Transparency and explainability are crucial for AI systems.",
                "Governments are developing regulations for AI technologies."
            ]
        },
        # Add more training examples here...
    ]
    
    print("=== Testing Supervised Learning Summarizer ===\n")
    
    # Note: This is a minimal example with 1 document
    # For real training, you need 100+ documents with gold summaries
    print("Note: This demo uses 1 document. For production, use 100+ labeled examples.\n")
    
    # Initialize and train
    summarizer = SupervisedSummarizer()
    
    # For demo purposes, duplicate the example to have enough samples
    print("Creating minimal training set (demo only)...")
    demo_data = training_data * 20  # Duplicate to have 20 samples
    
    try:
        results = summarizer.train(demo_data)
        
        # Test summarization
        print("\n=== Testing Summarization ===")
        test_text = """
            Climate change poses severe risks to global ecosystems.
            Rising temperatures are melting polar ice caps rapidly.
            Sea levels are projected to rise significantly by 2100.
            Extreme weather events are becoming more frequent.
            Renewable energy adoption is growing but not fast enough.
            International cooperation is essential for climate action.
            Individual actions also matter in reducing carbon footprints.
        """
        
        summary = summarizer.summarize(test_text, num_sentences=3)
        print(f"\nOriginal text ({len(test_text.split())} words):")
        print(test_text[:200] + "...")
        print(f"\nGenerated summary ({len(summary.split())} words):")
        print(summary)
        
        # Save model
        summarizer.save_model('demo_model.pkl')
        print("\nModel saved successfully!")
        
    except Exception as e:
        print(f"Training requires BERT. Error: {e}")
        print("\nTo use supervised learning:")
        print("1. Ensure bert_similarity.py is working")
        print("2. Prepare 100+ documents with gold summaries")
        print("3. Format as shown in training_data example")
        print("4. Run training with real dataset")
