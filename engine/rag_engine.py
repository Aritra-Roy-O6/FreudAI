import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Load our lightweight embedding model 
# (This stays local and fast)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

class HybridRAGEngine:
    def __init__(self):
        self.corpus = []
        self.bm25_index = None
        self.is_indexed = False

    def sync_with_db(self, memory_collection):
        """
        Fetches the latest memories from ChromaDB and rebuilds the BM25 index.
        This ensures keyword search is always up to date with the vector store.
        """
        if not memory_collection:
            return

        # 1. Pull all raw text from ChromaDB
        results = memory_collection.get()
        memories = results.get('documents', [])

        if not memories:
            self.is_indexed = False
            return

        self.corpus = memories

        # 2. Rebuild BM25 Keyword Index
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        self.is_indexed = True
        print(f"[RAG Engine] Synced {len(self.corpus)} memories for Hybrid Search.")

    def search_bm25(self, query: str, top_k: int = 5) -> dict:
        """Exact keyword matching (Great for names like 'Sarah' or 'Sophia')."""
        if not self.is_indexed or not self.bm25_index: 
            return {}
        
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        return {idx: scores[idx] for idx in top_n_indices if scores[idx] > 0}

    def search_semantic(self, query: str, memory_collection, top_k: int = 5) -> dict:
        """Semantic meaning matching using the existing ChromaDB collection."""
        if not memory_collection or memory_collection.count() == 0:
            return {}
        
        # Query ChromaDB directly
        results = memory_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        scores = {}
        if results and results['ids'] and results['ids'][0]:
            # ChromaDB doesn't always return standard similarity, 
            # so we normalize or use rank order.
            for i, doc_id in enumerate(results['ids'][0]):
                # We use a simple decay score for ranking
                scores[self.corpus.index(results['documents'][0][i])] = 1.0 / (i + 1)
                
        return scores

    def reciprocal_rank_fusion(self, bm25_results: dict, semantic_results: dict, k: int = 60) -> list:
        """
        Merges results using Reciprocal Rank Fusion (RRF).
        This boosts items that appear in BOTH keyword and semantic searches.
        """
        bm25_sorted = sorted(bm25_results.keys(), key=lambda x: bm25_results[x], reverse=True)
        semantic_sorted = sorted(semantic_results.keys(), key=lambda x: semantic_results[x], reverse=True)

        rrf_scores = {}

        for rank, doc_idx in enumerate(bm25_sorted):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1))

        for rank, doc_idx in enumerate(semantic_sorted):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1))

        return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    def retrieve(self, query: str, memory_collection, top_k: int = 3) -> list[str]:
        """
        The Master Pipeline: 
        1. Sync -> 2. Keyword Search -> 3. Semantic Search -> 4. RRF Fusion
        """
        # Always sync before retrieval to ensure we have the latest messages
        self.sync_with_db(memory_collection)

        if not self.is_indexed:
            return []

        # 1. Dual Search
        bm25_res = self.search_bm25(query, top_k=5)
        semantic_res = self.search_semantic(query, memory_collection, top_k=5)

        # 2. Fuse
        fused_ranked_items = self.reciprocal_rank_fusion(bm25_res, semantic_res)

        # 3. Return Text
        final_docs = []
        for doc_idx, rrf_score in fused_ranked_items[:top_k]:
            final_docs.append(self.corpus[int(doc_idx)])

        return final_docs