from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer('all-MiniLM-L6-v2')

CRISIS_ANCHORS = [
    "I want to kill myself.",
    "I want to end my own life.",
    "I cannot go on living like this and want to die.",
    "planning to commit suicide.",
    "I'm going to hurt myself.",
    "I have no reason to live anymore.",
    "nobody would care if I was dead."
]

crisis_embeddings = embedder.encode(CRISIS_ANCHORS)
text = "i wanna kill my sself"
text_emb = embedder.encode([text])[0]

similarities = [np.dot(text_emb, anchor) / (np.linalg.norm(text_emb) * np.linalg.norm(anchor)) for anchor in crisis_embeddings]
print(f"Max similarity for '{text}': {max(similarities)}")
