"""
Multi-Document Summarization Module
Combines and summarizes multiple documents together.
"""

import re
from collections import Counter

class MultiDocSummarizer:
    """Summarize multiple documents together."""
    
    def __init__(self):
        """Initialize multi-document summarizer."""
        pass
    
    def combine_documents(self, documents):
        """
        Combine multiple document texts.
        """
        combined_text = ""
        doc_boundaries = []
        current_position = 0
        
        for doc in documents:
            text = doc['text']
            filename = doc['filename']
            
            # Add document separator
            if combined_text:
                combined_text += "\n\n"
                current_position += 2
            
            # Add text
            combined_text += text
            
            # Track document boundaries
            doc_boundaries.append({
                'filename': filename,
                'start': current_position,
                'end': current_position + len(text),
                'length': len(text)
            })
            
            current_position += len(text)
        
        return {
            'combined_text': combined_text,
            'doc_boundaries': doc_boundaries,
            'total_docs': len(documents),
            'total_length': len(combined_text)
        }
    
    def calculate_summary_length(self, text, percentage=None, num_sentences=None):
        """
        Calculate summary length from percentage or sentence count.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        total_sentences = len(sentences)
        
        if percentage:
            # Calculate from percentage
            target_sentences = max(1, int(total_sentences * (percentage / 100)))
            return min(target_sentences, total_sentences)
        elif num_sentences:
            # Use fixed count
            return min(num_sentences, total_sentences)
        else:
            # Default: 25% of document
            return max(1, int(total_sentences * 0.25))
    
    def get_document_stats(self, documents):
        """
        Get statistics about multiple documents.
        """
        total_words = 0
        total_sentences = 0
        total_chars = 0
        
        for doc in documents:
            text = doc['text']
            words = len(text.split())
            sentences = len(re.split(r'[.!?]+', text))
            chars = len(text)
            
            total_words += words
            total_sentences += sentences
            total_chars += chars
        
        return {
            'total_documents': len(documents),
            'total_words': total_words,
            'total_sentences': total_sentences,
            'total_characters': total_chars,
            'avg_words_per_doc': total_words // len(documents) if documents else 0,
            'avg_sentences_per_doc': total_sentences // len(documents) if documents else 0
        }