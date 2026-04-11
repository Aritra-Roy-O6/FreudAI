import numpy as np
from sentence_transformers import SentenceTransformer
import re

# Load the same lightweight embedding model used in our RAG engine
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# CONCEPT ANCHORS (Semantic Definitions)
# ==========================================
CRISIS_ANCHORS = [
    "I want to end my life",
    "I can't take this pain anymore and want to die",
    "I'm thinking about hurting myself",
    "There is no point in living anymore"
]

OVERLOAD_ANCHORS = [
    "I am completely drowning in responsibilities",
    "Everything is happening at once and it's too much",
    "Between work, school, and family, I am breaking down",
    "I have way too much on my plate right now"
]

NUMBING_ANCHORS = [
    "I just feel completely numb",
    "I don't care about anything anymore, I feel empty",
    "Everything is just blank and flat"
]

# Pre-compute anchor embeddings to save time on every turn
crisis_embeddings = embedder.encode(CRISIS_ANCHORS)
overload_embeddings = embedder.encode(OVERLOAD_ANCHORS)
numbing_embeddings = embedder.encode(NUMBING_ANCHORS)

def cosine_similarity(vec1, vec2):
    """Calculates the mathematical distance between two meaning vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_max_similarity(text_embedding, anchor_embeddings):
    """Compares the user's text against all anchors and returns the highest match."""
    similarities = [cosine_similarity(text_embedding, anchor) for anchor in anchor_embeddings]
    return max(similarities)

# ==========================================
# THE MASTER ROUTER
# ==========================================
def priority_router(text: str, lexical_score: float, is_sarcastic: bool, is_implicit: bool) -> str:
    text_lower = text.lower()
    text_embedding = embedder.encode([text_lower])[0]

    # 1. SEMANTIC CRISIS SIGNAL (Strict Threshold: 0.70)
    if get_max_similarity(text_embedding, crisis_embeddings) > 0.70:
        return "[CRISIS_SIGNAL_ESCALATE]"
        
    # 2. SARCASM / DEFLECTION
    if is_sarcastic:
        return "[SARCASM_DEFLECTION]"
        
    # 3. IMPLICIT DISTRESS
    if is_implicit:
        return "[IMPLICIT_DISTRESS]"
        
    # 4. SEMANTIC COGNITIVE OVERLOAD (Moderate Threshold: 0.60)
    if get_max_similarity(text_embedding, overload_embeddings) > 0.60:
        return "[COGNITIVE_OVERLOAD]"

    # 5. EXPLICIT DISTRESS
    if lexical_score < -0.5:
        return "[EXPLICIT_DISTRESS]"
        
    # 6. SEMANTIC EMOTIONAL NUMBING (Moderate Threshold: 0.60)
    if get_max_similarity(text_embedding, numbing_embeddings) > 0.60:
        return "[EMOTIONAL_NUMBING]"
        
    # Default Fallback
    return "[NEUTRAL_CONVERSATIONAL]"