#Handles text cleaning, sentence tokenization, and word tokenization.
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data (run once)
def download_nltk_data():
    """Download required NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

class TextPreprocessor:
    """Handles text preprocessing for summarization."""
    
    def __init__(self, language='english'):
        """
        Initialize preprocessor.
        
        Args:
            language (str): Language for stopwords and tokenization
        """
        download_nltk_data()
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
    
    def clean_text(self, text):
        """
        Clean raw text by removing special characters and extra whitespace.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        # Remove special characters except sentence punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\?\!,]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize_sentences(self, text):
        """
        Split text into sentences.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        sentences = sent_tokenize(text, language=self.language)
        return sentences
    
    def tokenize_words(self, text, remove_stopwords=True, stem=False):
        """
        Split text into words and optionally remove stopwords and stem.
        
        Args:
            text (str): Input text
            remove_stopwords (bool): Whether to remove stopwords
            stem (bool): Whether to apply stemming
            
        Returns:
            list: List of processed words
        """
        # Tokenize
        words = word_tokenize(text.lower(), language=self.language)
        
        # Remove stopwords if requested
        if remove_stopwords:
            words = [w for w in words if w not in self.stop_words and w.isalpha()]
        else:
            words = [w for w in words if w.isalpha()]
        
        # Apply stemming if requested
        if stem:
            words = [self.stemmer.stem(w) for w in words]
        
        return words
    
    def preprocess_document(self, text):
        """
        Complete preprocessing pipeline for a document.
        
        Args:
            text (str): Raw document text
            
        Returns:
            dict: Preprocessed data with sentences and words
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize into sentences
        sentences = self.tokenize_sentences(cleaned_text)
        
        # Tokenize each sentence into words
        sentence_words = []
        for sentence in sentences:
            words = self.tokenize_words(sentence, remove_stopwords=True, stem=False)
            sentence_words.append(words)
        
        # Get all words from document (for frequency analysis)
        all_words = self.tokenize_words(cleaned_text, remove_stopwords=True, stem=False)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'sentence_words': sentence_words,
            'all_words': all_words,
            'num_sentences': len(sentences),
            'num_words': len(all_words)
        }