import os
import numpy as np
from textblob import TextBlob
import google.generativeai as genai

# ==========================================
# 1. SETUP GEMINI EMBEDDINGS (Cloud-Friendly)
# ==========================================
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    print("CRITICAL: API Key missing for detectors!")

EMBEDDING_MODEL = 'models/text-embedding-004'

def get_embeddings(text_input):
    """Fetches embeddings via Gemini API instead of heavy local models."""
    if not api_key:
        # Fallback to zero-vectors if the key fails so the server doesn't crash
        return [np.zeros(768)] if isinstance(text_input, list) else np.zeros(768)
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL, content=text_input)
        return result['embedding']
    except Exception as e:
        print(f"⚠ Gemini Detector Error: {e}")
        return [np.zeros(768)] if isinstance(text_input, list) else np.zeros(768)

# ==========================================
# 2. SEMANTIC CONCEPT ANCHORS
# ==========================================
DEFLECTION_ANCHORS = [
    "It's whatever, I don't really care.",
    "I'm perfectly fine, everything is great.",
    "It doesn't matter anyway.",
    "I'm just laughing it off, it's fine."
]

AVOIDANCE_ANCHORS = [
    "I just lie in bed all day and do nothing.",
    "I stopped doing the things I used to love.",
    "I can't be bothered to try anymore.",
    "I don't see the point in putting in effort."
]

COMPARISON_ANCHORS = [
    "Everyone else seems to have their life figured out.",
    "I feel like I don't belong here compared to them.",
    "They are so much better than me at everything.",
    "I'm falling behind and everyone else is succeeding."
]

CRISIS_ANCHORS = [
    "I want to kill myself.",
    "I'm going to kill my self.",
    "I wanna kill myself.",
    "kms",
    "I want to end my own life.",
    "I cannot go on living like this and want to die.",
    "planning to commit suicide.",
    "I'm going to hurt myself.",
    "I have no reason to live anymore.",
    "nobody would care if I was dead."
]

# Pre-compute embeddings on startup
print("Initializing detector anchors with Gemini...")
deflection_embeddings = get_embeddings(DEFLECTION_ANCHORS)
avoidance_embeddings = get_embeddings(AVOIDANCE_ANCHORS)
comparison_embeddings = get_embeddings(COMPARISON_ANCHORS)
crisis_embeddings = get_embeddings(CRISIS_ANCHORS)
print("✓ Detector anchors loaded.")

def cosine_similarity(vec1, vec2):
    """Calculates mathematical distance between two meaning vectors."""
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def get_max_similarity(text_embedding, anchor_embeddings):
    """Returns the highest similarity score against a list of anchors."""
    similarities = [cosine_similarity(text_embedding, anchor) for anchor in anchor_embeddings]
    return max(similarities) if similarities else 0.0

# ==========================================
# 3. THE DETECTORS
# ==========================================
# Gemini embeddings score higher generally, so we bump the threshold to 0.65
THRESHOLD = 0.65

def lexical_scan(text: str) -> float:
    """Returns basic sentiment polarity from -1.0 (negative) to 1.0 (positive)."""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def sarcasm_probe(text: str, lexical_score: float) -> tuple[bool, float]:
    text_embedding = get_embeddings(text.lower())
    deflection_score = get_max_similarity(text_embedding, deflection_embeddings)
    
    is_sarcastic = False
    if deflection_score > THRESHOLD and lexical_score >= 0.0:
        is_sarcastic = True
        
    return is_sarcastic, deflection_score

def implicit_distress_flag(text: str) -> tuple[bool, float]:
    text_embedding = get_embeddings(text.lower())
    
    avoidance_score = get_max_similarity(text_embedding, avoidance_embeddings)
    comparison_score = get_max_similarity(text_embedding, comparison_embeddings)
    
    max_implicit_score = max(avoidance_score, comparison_score)
    has_implicit_cue = max_implicit_score > THRESHOLD
    
    return has_implicit_cue, max_implicit_score

def crisis_flag_semantic(text: str) -> tuple[bool, float]:
    text_embedding = get_embeddings(text.lower())
    crisis_score = get_max_similarity(text_embedding, crisis_embeddings)
    
    # Crisis gets a slightly stricter threshold to avoid false alarms
    is_crisis = crisis_score > (THRESHOLD + 0.05)
    return is_crisis, crisis_score