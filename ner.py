"""
Named Entity Recognition Module using spaCy
Extracts people, organizations, locations, and other entities.
"""

import spacy

class NERExtractor:
    """Extract named entities from text using spaCy."""
    
    def __init__(self, model='en_core_web_sm'):
        """
        Initialize NER extractor.
        
        Args:
            model (str): spaCy model name
        """
        try:
            self.nlp = spacy.load(model)
        except:
            # If model not found, download it
            import os
            os.system(f'python -m spacy download {model}')
            self.nlp = spacy.load(model)
    
    def extract_entities(self, text):
        """
        Extract named entities from text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Entities grouped by type
        """
        doc = self.nlp(text)
        
        # Group entities by type
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geo-Political Entity (countries, cities)
            'DATE': [],
            'MONEY': [],
            'PRODUCT': [],
            'EVENT': [],
            'OTHER': []
        }
        
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_
            }
            
            if ent.label_ in entities:
                entities[ent.label_].append(entity_info)
            else:
                entities['OTHER'].append(entity_info)
        
        # Remove duplicates
        for entity_type in entities:
            seen = set()
            unique_entities = []
            for ent in entities[entity_type]:
                if ent['text'].lower() not in seen:
                    seen.add(ent['text'].lower())
                    unique_entities.append(ent)
            entities[entity_type] = unique_entities
        
        # Count total entities
        total_count = sum(len(entities[key]) for key in entities)
        
        return {
            'entities': entities,
            'total_count': total_count,
            'entity_types': [key for key in entities if len(entities[key]) > 0]
        }
    
    def get_entity_summary(self, text):
        """
        Get a summary of entities with counts.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Entity counts by type
        """
        result = self.extract_entities(text)
        
        summary = {}
        for entity_type, entity_list in result['entities'].items():
            if len(entity_list) > 0:
                summary[entity_type] = {
                    'count': len(entity_list),
                    'examples': [e['text'] for e in entity_list[:5]]  # Top 5 examples
                }
        
        return summary