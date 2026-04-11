import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import our custom modules from Phases 1-4
from detectors import lexical_scan, sarcasm_probe, implicit_distress_flag
from router import priority_router
from llm_core import generate_response
from memory_manager import (
    store_in_long_term_memory,
    get_short_term_context,
    load_entities,
    memory_collection,
    wipe_all_memory
)
from rag_engine import HybridRAGEngine

app = FastAPI(title="FreudAI Backend")

class UserMessagePayload(BaseModel):
    message: str
    session_id: str = "default_user"

# Global state for short-term memory (in-memory list for MVP)
chat_history = []

@app.post("/chat")
async def chat_endpoint(payload: UserMessagePayload):
    user_text = payload.message

    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # ==========================================
    # 1. SIGNAL DETECTION & ROUTING
    # ==========================================
    lexical_score = lexical_scan(user_text)
    is_sarcastic, _ = sarcasm_probe(user_text, lexical_score)
    is_implicit, _ = implicit_distress_flag(user_text)

    # Determine the priority emotional state
    emotion_tag = priority_router(user_text, lexical_score, is_sarcastic, is_implicit)

    # ==========================================
    # 2. MEMORY & CONTEXT RETRIEVAL
    # ==========================================
    # A. Get Short-Term Context (Last 8 turns)
    short_term_context = "\n".join(get_short_term_context(chat_history, window_size=4))

    # B. Get Known Entities
    user_entities = load_entities()

    # C. Get Long-Term Context via Hybrid RAG
    retrieved_memories = []
    
    # Pull all past memories from ChromaDB to feed our dual-engine
    db_results = memory_collection.get()
    all_memories = db_results.get('documents', [])
    
    # If we have past memories, run the FAISS + BM25 fusion
    if all_memories:
        engine = HybridRAGEngine()
        engine.fit_corpus(all_memories)
        retrieved_memories = engine.retrieve(user_text, top_k=1)

    # ==========================================
    # 3. REASONING CORE (LLM GENERATION)
    # ==========================================
    # Inject all context into the 5-Layer Prompt
    bot_reply = generate_response(
        user_text=user_text,
        emotion_tag=emotion_tag,
        retrieved_memories=retrieved_memories,
        user_entities=user_entities,
        short_term_history=short_term_context
    )

    # ==========================================
    # 4. THE WRITE-BACK CYCLE
    # ==========================================
    # A. Update short-term history for the next turn
    chat_history.append(f"User: {user_text}")
    chat_history.append(f"AI: {bot_reply}")

    # B. Save the interaction to ChromaDB for long-term memory
   # Only save to vector database if the user typed a meaningful sentence (> 4 words)
    if len(user_text.split()) > 4:
        store_in_long_term_memory(user_text, bot_reply, emotion_tag, payload.session_id)

    # (Note: In a post-MVP production build, an LLM call here would extract new entities to update user_entities)

    return {
        "emotion_tag": emotion_tag,
        "response": bot_reply
    }

# ==========================================
@app.post("/reset")
async def reset_endpoint():
    """Wipes all short-term and long-term memory for a fresh start."""
    global chat_history
    # Clear the API's short-term list
    chat_history.clear()
    
    # Trigger the deep wipe
    wipe_all_memory()
    
    return {"status": "Memory fully wiped. Ready for a new session."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)