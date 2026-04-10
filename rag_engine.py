import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ==========================================
# STEP 1: INITIALIZE MODELS & ENGINE
# ==========================================

# Load our lightweight embedding model for FAISS
embedder = SentenceTransformer('all-MiniLM-L6-v2')

class HybridRAGEngine:
    def __init__(self):
        self.corpus = []
        self.bm25_index = None
        self.faiss_index = None
        self.is_indexed = False

    def fit_corpus(self, memories: list[str]):
        """
        Takes a list of memory strings and builds BOTH the BM25 and FAISS indexes.
        """
        if not memories:
            print("Warning: Empty memory corpus provided to RAG Engine.")
            return

        self.corpus = memories

        # 1. Build BM25 Keyword Index
        # We tokenize the text by splitting it into lowercase words
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        # 2. Build FAISS Semantic Index
        embeddings = embedder.encode(self.corpus)
        dimension = embeddings.shape[1]  # Get the vector size (384 for MiniLM)
        
        # Initialize a flat (exact search) L2 distance FAISS index
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(np.array(embeddings))
        
        self.is_indexed = True
        print(f"Success: Hybrid RAG indexed {len(self.corpus)} memories.")

# ==========================================
# STEP 2: DUAL RETRIEVAL LOGIC
# ==========================================

    def search_bm25(self, query: str, top_k: int = 5) -> dict:
        """Exact keyword matching."""
        if not self.is_indexed: return {}
        
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get indices of the top_k highest scores
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return a dictionary of {doc_index: score}
        return {idx: scores[idx] for idx in top_n_indices if scores[idx] > 0}

    def search_faiss(self, query: str, top_k: int = 5) -> dict:
        """Semantic meaning matching."""
        if not self.is_indexed: return {}
        
        query_vector = embedder.encode([query])
        
        # FAISS returns distances (lower is better) and indices
        distances, indices = self.faiss_index.search(np.array(query_vector), top_k)
        
        results = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 means not enough results found
                # Invert distance to a similarity score so higher = better
                similarity_score = 1.0 / (1.0 + dist)
                results[idx] = similarity_score
                
        return results

# ==========================================
# STEP 3: RECIPROCAL RANK FUSION (RRF)
# ==========================================

    def reciprocal_rank_fusion(self, bm25_results: dict, faiss_results: dict, k: int = 60) -> list:
        """
        Merges results using their rank position rather than raw scores.
        Returns a list of tuples: [(doc_index, rrf_score), ...]
        """
        # Sort results by score descending to determine their rank (1st, 2nd, 3rd)
        bm25_sorted = sorted(bm25_results.keys(), key=lambda x: bm25_results[x], reverse=True)
        faiss_sorted = sorted(faiss_results.keys(), key=lambda x: faiss_results[x], reverse=True)

        rrf_scores = {}

        # Apply RRF formula: 1 / (k + rank) for BM25
        for rank, doc_idx in enumerate(bm25_sorted):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1))

        # Apply RRF formula for FAISS
        for rank, doc_idx in enumerate(faiss_sorted):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1))

        # Sort the final combined dictionary by the highest RRF score
        fused_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return fused_results

# ==========================================
# STEP 4: MASTER RETRIEVAL PIPELINE
# ==========================================

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """
        Executes the full pipeline: Dual Search -> RRF Fusion -> Returns Actual Text.
        """
        if not self.is_indexed:
            return []

        # 1. Search both indexes
        bm25_res = self.search_bm25(query, top_k=5)
        faiss_res = self.search_faiss(query, top_k=5)

        # 2. Fuse the rankings
        fused_ranked_items = self.reciprocal_rank_fusion(bm25_res, faiss_res)

        # 3. Extract the actual memory text based on the winning indices
        final_docs = []
        for doc_idx, rrf_score in fused_ranked_items[:top_k]:
            final_docs.append(self.corpus[int(doc_idx)])

        return final_docs

# --- Final Phase 3 Test Block ---
if __name__ == "__main__":
    print("\n--- Testing Hybrid Pipeline & RRF ---")
    
    mock_memories = [
        "User feels overwhelmed by university exams.",
        "User had a massive argument with their sister Sarah on Tuesday.",
        "User mentioned they used to love playing basketball but stopped.",
        "User feels like an imposter at their new job."
    ]
    
    engine = HybridRAGEngine()
    engine.fit_corpus(mock_memories)
    
    query = "fight with sister on Tuesday"
    
    print(f"\nQuery: '{query}'")
    print("Executing Hybrid Retrieval...\n")
    
    final_memories = engine.retrieve(query, top_k=2)
    
    for i, mem in enumerate(final_memories, 1):
        print(f"Rank {i}: {mem}")