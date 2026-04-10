import re
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a lightweight embedding model for the sarcasm contrast check
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def lexical_scan(text: str) -> float:
    """Returns basic sentiment polarity from -1.0 (negative) to 1.0 (positive)."""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def sarcasm_probe(text: str, lexical_score: float) -> tuple[bool, float]:
    """
    Detects sarcasm by checking for positive surface language paired with deflection triggers.
    """
    deflection_triggers = [r"it's fine", r"whatever", r"i'm great", r"doesn't matter"]
    
    # Check for exact trigger phrases
    has_trigger = any(re.search(pattern, text.lower()) for pattern in deflection_triggers)
    
    # Basic Contrastive Logic for Phase 1
    # If they use a deflection phrase but surface sentiment is neutral/positive, flag it.
    is_sarcastic = has_trigger and (lexical_score >= 0.0)
    
    confidence = 0.8 if is_sarcastic else 0.0
    return is_sarcastic, confidence

def implicit_distress_flag(text: str) -> tuple[bool, float]:
    """
    Flags behavioral patterns like skipping obligations or minimizing pain.
    """
    # Patterns indicating avoidance, past-tense joy, or minimization
    implicit_patterns = [
        r"used to love", r"stopped going", r"skip", r"stay in bed", 
        r"what's the point", r"too much effort", r"just going numb"
    ]
    
    has_implicit_cue = any(re.search(pattern, text.lower()) for pattern in implicit_patterns)
    
    confidence = 0.85 if has_implicit_cue else 0.0
    return has_implicit_cue, confidence