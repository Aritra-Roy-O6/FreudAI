import numpy as np
from textblob import TextBlob
from sentence_transformers import SentenceTransformer

# Load the shared lightweight embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# SEMANTIC CONCEPT ANCHORS
# ==========================================

# 1. Deflection / Masking Anchors
DEFLECTION_ANCHORS = [
    "It's whatever, I don't really care.",
    "I'm perfectly fine, everything is great.",
    "It doesn't matter anyway.",
    "I'm just laughing it off, it's fine."
]

# 2. Avoidance / Anhedonia Anchors
AVOIDANCE_ANCHORS = [
    "I just lie in bed all day and do nothing.",
    "I stopped doing the things I used to love.",
    "I can't be bothered to try anymore.",
    "I don't see the point in putting in effort."
]

# 3. Imposter Syndrome / Social Comparison Anchors
COMPARISON_ANCHORS = [
    "Everyone else seems to have their life figured out.",
    "I feel like I don't belong here compared to them.",
    "They are so much better than me at everything.",
    "I'm falling behind and everyone else is succeeding."
]

# Pre-compute embeddings for speed
deflection_embeddings = embedder.encode(DEFLECTION_ANCHORS)
avoidance_embeddings = embedder.encode(AVOIDANCE_ANCHORS)
comparison_embeddings = embedder.encode(COMPARISON_ANCHORS)

def cosine_similarity(vec1, vec2):
    """Calculates mathematical distance between two meaning vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_max_similarity(text_embedding, anchor_embeddings):
    """Returns the highest similarity score against a list of anchors."""
    similarities = [cosine_similarity(text_embedding, anchor) for anchor in anchor_embeddings]
    return max(similarities)

# ==========================================
# THE DETECTORS
# ==========================================

def lexical_scan(text: str) -> float:
    """Returns basic sentiment polarity from -1.0 (negative) to 1.0 (positive)."""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def sarcasm_probe(text: str, lexical_score: float) -> tuple[bool, float]:
    """
    Detects sarcasm by checking if the user is semantically deflecting, 
    but the basic text analyzer thinks they are being positive.
    """
    text_embedding = embedder.encode([text.lower()])[0]
    
    # Check if the text means "I am deflecting/masking"
    deflection_score = get_max_similarity(text_embedding, deflection_embeddings)
    
    is_sarcastic = False
    # If they are semantically deflecting (> 0.55) AND using neutral/positive words
    if deflection_score > 0.55 and lexical_score >= 0.0:
        is_sarcastic = True
        
    return is_sarcastic, deflection_score

def implicit_distress_flag(text: str) -> tuple[bool, float]:
    """
    Flags behavioral patterns (avoidance or imposter syndrome) using semantic math.
    """
    text_embedding = embedder.encode([text.lower()])[0]
    
    avoidance_score = get_max_similarity(text_embedding, avoidance_embeddings)
    comparison_score = get_max_similarity(text_embedding, comparison_embeddings)
    
    # If either concept hits the threshold
    max_implicit_score = max(avoidance_score, comparison_score)
    
    has_implicit_cue = max_implicit_score > 0.55
    
    return has_implicit_cue, max_implicit_score