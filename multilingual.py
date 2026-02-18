"""
Multilingual Support Module
Provides language detection and multilingual text processing.
"""

from collections import Counter
import re

class MultilingualProcessor:
    """Handle multilingual text processing."""
    
    # Language-specific stopwords (basic sets)
    STOPWORDS = {
        'english': {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this', 'it', 'from', 'be', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'can'},
        'hindi': {'के', 'का', 'की', 'में', 'है', 'हैं', 'को', 'से', 'पर', 'ने', 'और', 'या', 'एक', 'यह', 'वह', 'इस', 'उस', 'तो', 'ही', 'थे', 'था', 'थी', 'हो', 'हूं', 'हूँ'},
        'spanish': {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber', 'por', 'con', 'su', 'para', 'como', 'estar', 'tener', 'le', 'lo', 'todo', 'pero', 'más', 'hacer', 'o', 'poder', 'decir'},
        'french': {'le', 'de', 'un', 'être', 'et', 'à', 'il', 'avoir', 'ne', 'je', 'son', 'que', 'se', 'qui', 'ce', 'dans', 'en', 'du', 'elle', 'au', 'pour', 'pas', 'que', 'vous', 'par', 'sur', 'faire', 'plus'},
        'german': {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als', 'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'dass'}
    }
    
    # Language names for UI
    LANGUAGE_NAMES = {
        'english': 'English',
        'hindi': 'हिंदी (Hindi)',
        'spanish': 'Español (Spanish)',
        'french': 'Français (French)',
        'german': 'Deutsch (German)'
    }
    
    def __init__(self, default_language='english'):
        """
        Initialize multilingual processor.
        
        Args:
            default_language (str): Default language code
        """
        self.default_language = default_language
    
    def get_stopwords(self, language='english'):
        """
        Get stopwords for a language.
        
        Args:
            language (str): Language code
            
        Returns:
            set: Stopwords for the language
        """
        return self.STOPWORDS.get(language.lower(), self.STOPWORDS['english'])
    
    def detect_language(self, text):
        """
        Simple language detection based on character patterns.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Detected language code
        """
        # Simple heuristic: check for Hindi Devanagari script
        hindi_chars = re.findall(r'[\u0900-\u097F]', text)
        if len(hindi_chars) > 20:
            return 'hindi'
        
        # Spanish indicators
        spanish_chars = re.findall(r'[áéíóúñü¿¡]', text.lower())
        if len(spanish_chars) > 5:
            return 'spanish'
        
        # French indicators
        french_chars = re.findall(r'[àâæçéèêëïîôùûüÿœ]', text.lower())
        if len(french_chars) > 5:
            return 'french'
        
        # German indicators
        german_chars = re.findall(r'[äöüß]', text.lower())
        if len(german_chars) > 3:
            return 'german'
        
        # Default to English
        return 'english'
    
    def tokenize(self, text, language='english'):
        """
        Tokenize text based on language.
        
        Args:
            text (str): Input text
            language (str): Language code
            
        Returns:
            list: List of tokens
        """
        # Basic tokenization (split on whitespace and punctuation)
        tokens = re.findall(r'\w+', text.lower())
        
        # Remove stopwords
        stopwords = self.get_stopwords(language)
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        
        return tokens
    
    def get_supported_languages(self):
        """
        Get list of supported languages.
        
        Returns:
            dict: Language codes and names
        """
        return self.LANGUAGE_NAMES