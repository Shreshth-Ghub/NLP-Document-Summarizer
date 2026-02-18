"""
Named Entity Recognition Module using spaCy
Extracts people, organizations, locations, and other entities.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["SPACY_FORCE_CPU"] = "true"

import spacy


class NERExtractor:
    """Extract named entities from text using spaCy."""
    
    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize NER extractor.
        
        Args:
            model (str): spaCy model name
        """
        try:
            self.nlp = spacy.load(model)
        except Exception:
            # If model not found, download it
            import os
            os.system(f"python -m spacy download {model}")
            self.nlp = spacy.load(model)
    
    def extract_entities(self, text: str):
        """
        Extract named entities from text.
        
        Returns:
            dict with:
              - total_entities: int
              - by_type: dict[label] = list[str]
        """
        doc = self.nlp(text)
        
        by_type = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "DATE": [],
            "MONEY": [],
            "PRODUCT": [],
            "EVENT": [],
            "OTHER": [],
        }
        
        for ent in doc.ents:
            label = ent.label_
            value = ent.text.strip()
            if not value:
                continue
            if label in by_type:
                by_type[label].append(value)
            else:
                by_type["OTHER"].append(value)
        
        # Remove duplicates and normalize to sorted unique strings
        for label, values in by_type.items():
            seen = set()
            unique = []
            for v in values:
                key = v.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(v)
            by_type[label] = unique
        
        total = sum(len(v) for v in by_type.values())
        
        return {
            "total_entities": total,
            "by_type": {k: v for k, v in by_type.items() if v},  # drop empty types
        }
    
    def get_entity_summary(self, text: str):
        """
        Get a summary of entities with counts and top examples.
        """
        result = self.extract_entities(text)
        summary = {}
        for label, values in result["by_type"].items():
            summary[label] = {
                "count": len(values),
                "examples": values[:5],
            }
        return summary
