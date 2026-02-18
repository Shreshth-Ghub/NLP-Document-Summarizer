"""
abstractive.py
Abstractive Summarization using T5/BART

Generates human-like summaries using state-of-the-art transformer models.
"""

from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class AbstractiveSummarizer:
    """Abstractive summarization using T5 or BART models"""
    
    def __init__(self, model_name='t5-small'):
        """
        Initialize abstractive summarization model
        
        Args:
            model_name: Model to use
                       't5-small' (default): 240MB, fast, good quality
                       't5-base': 850MB, slower, better quality
                       'facebook/bart-large-cnn': 1.6GB, best for news
                       'google/pegasus-xsum': 2.3GB, specialized for news
        """
        print(f"Loading abstractive model: {model_name}")
        print("(First time will download model - may take a few minutes)")
        
        self.model_name = model_name
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            device=-1  # CPU (use device=0 for GPU)
        )
        
        print("Abstractive model loaded successfully!")
    
    def summarize(self, text, max_length=150, min_length=30, length_penalty=2.0):
        """
        Generate abstractive summary
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            length_penalty: Control verbosity (higher = shorter summaries)
        
        Returns:
            Generated summary text
        """
        if not text or len(text.strip()) < 50:
            return text
        
        # Truncate if too long (model max input: ~1024 tokens ≈ 4000 chars)
        max_input_length = 4000
        if len(text) > max_input_length:
            text = text[:max_input_length]
        
        try:
            # Generate summary
            result = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                num_beams=4,  # Beam search width
                early_stopping=True
            )
            
            summary = result[0]['summary_text']
            return summary
        
        except Exception as e:
            print(f"Abstractive summarization error: {e}")
            # Fallback: return first few sentences
            sentences = text.split('.')[:3]
            return '. '.join(sentences) + '.'
    
    def summarize_by_percentage(self, text, percentage=25):
        """
        Generate summary of target percentage length
        
        Args:
            text: Input text
            percentage: Target percentage of original length
        
        Returns:
            Generated summary
        """
        # Estimate tokens (roughly 4 chars per token)
        estimated_tokens = len(text) // 4
        target_tokens = int(estimated_tokens * (percentage / 100))
        
        # Bounds
        max_length = max(50, min(target_tokens, 500))
        min_length = max(20, max_length // 3)
        
        return self.summarize(
            text,
            max_length=max_length,
            min_length=min_length
        )
    
    def summarize_by_sentences(self, text, num_sentences=5):
        """
        Generate summary targeting specific sentence count
        
        Args:
            text: Input text
            num_sentences: Target number of sentences
        
        Returns:
            Generated summary
        """
        # Rough estimate: 15-20 tokens per sentence
        tokens_per_sentence = 18
        target_tokens = num_sentences * tokens_per_sentence
        
        max_length = max(50, min(target_tokens + 20, 500))
        min_length = max(20, target_tokens - 20)
        
        return self.summarize(
            text,
            max_length=max_length,
            min_length=min_length
        )


# Convenience function
def generate_abstractive_summary(text, model_name='t5-small', max_length=150):
    """
    Quick function to generate abstractive summary
    
    Args:
        text: Input text
        model_name: Model to use
        max_length: Maximum summary length
    
    Returns:
        Generated summary
    """
    summarizer = AbstractiveSummarizer(model_name)
    return summarizer.summarize(text, max_length=max_length)


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
    having become a routine technology.
    """
    
    print("=== Testing Abstractive Summarization ===\n")
    print(f"Original text ({len(text.split())} words):")
    print(text[:200] + "...\n")
    
    # Initialize summarizer
    summarizer = AbstractiveSummarizer(model_name='t5-small')
    
    # Test different configurations
    print("\n=== Summary 1: Default (max_length=150) ===")
    summary1 = summarizer.summarize(text, max_length=150)
    print(f"Summary ({len(summary1.split())} words):")
    print(summary1)
    
    print("\n=== Summary 2: Short (max_length=80) ===")
    summary2 = summarizer.summarize(text, max_length=80, min_length=20)
    print(f"Summary ({len(summary2.split())} words):")
    print(summary2)
    
    print("\n=== Summary 3: By percentage (25%) ===")
    summary3 = summarizer.summarize_by_percentage(text, percentage=25)
    print(f"Summary ({len(summary3.split())} words):")
    print(summary3)
    
    print("\n=== Summary 4: By sentences (3 sentences target) ===")
    summary4 = summarizer.summarize_by_sentences(text, num_sentences=3)
    print(f"Summary ({len(summary4.split())} words):")
    print(summary4)
