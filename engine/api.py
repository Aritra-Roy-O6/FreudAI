import sys
import os
import uvicorn
import json
import threading
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Ensure the engine directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from rag_engine import HybridRAGEngine
from detectors import lexical_scan, sarcasm_probe, implicit_distress_flag, crisis_flag_semantic
from router import priority_router
from llm_core import generate_response
from memory_manager import (
    store_in_long_term_memory,
    get_short_term_context,
    load_entities,
    save_entities,
    memory_collection,
    wipe_all_memory,
    summarize_session
)

app = FastAPI(title="FreudAI Backend")

# Enable CORS for local Electron communication
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

class UserMessagePayload(BaseModel):
    message: str
    session_id: str = "default_user"
    api_key: str = None  

class DeleteEntityPayload(BaseModel):
    category: str
    item: str

# Global state
chat_history = []
emotion_history = []
engine = HybridRAGEngine()

@app.post("/chat")
async def chat_endpoint(payload: UserMessagePayload):
    global chat_history, emotion_history
    user_text = payload.message
    
    # 1. EMOTION DETECTION
    lex_score = lexical_scan(user_text)
    is_sarcastic, _ = sarcasm_probe(user_text, lex_score)
    is_implicit, _ = implicit_distress_flag(user_text)
    is_crisis_semantic, _ = crisis_flag_semantic(user_text)
    emotion_tag = priority_router(user_text, lex_score, is_sarcastic, is_implicit, is_crisis_semantic)

    # 2. CONTEXT RETRIEVAL
    user_entities = load_entities() 
    retrieved_memories = engine.retrieve(user_text, memory_collection, top_k=3)
    short_term_context = "\n".join(get_short_term_context(chat_history, window_size=8))

    # 3. TRAJECTORY TRACKING
    emotion_history.append(emotion_tag)
    if len(emotion_history) > 3: emotion_history.pop(0)
    
    regression_note = ""
    if len(emotion_history) >= 2:
        prev, curr = emotion_history[-2], emotion_history[-1]
        if prev in ["[NEUTRAL_CONVERSATIONAL]", "[EMOTIONAL_NUMBING]"] and curr in ["[CRISIS_SIGNAL_ESCALATE]", "[EXPLICIT_DISTRESS]", "[COGNITIVE_OVERLOAD]"]:
            regression_note = f"\n[EMOTIONAL_REGRESSION: User shifted from {prev} to {curr}.]"

    # 4. GENERATION & ERROR INTERCEPTION (PHASE 4)
    try:
        raw_ai_output = generate_response(
            user_text=user_text,
            emotion_tag=emotion_tag,
            retrieved_memories=retrieved_memories,
            user_entities=user_entities,
            short_term_history=short_term_context,
            regression_note=regression_note,
            api_key=payload.api_key
        )

        # Entity Extraction Logic
        if "ENTITIES:" in raw_ai_output:
            split_idx = raw_ai_output.rfind("ENTITIES:")
            bot_reply = raw_ai_output[:split_idx].strip()
            # Clean JSON blocks
            raw_json = re.sub(r'^`{3}(?:json)?\s*', '', raw_ai_output[split_idx+9:].strip(), flags=re.IGNORECASE)
            raw_json = re.sub(r'\s*`{3}$', '', raw_json).strip()
            try:
                extracted = json.loads(raw_json)
                for cat, items in extracted.items():
                    if cat not in user_entities: user_entities[cat] = []
                    if isinstance(items, list):
                        for item in items:
                            if item.lower() not in [x.lower() for x in user_entities[cat]]:
                                user_entities[cat].append(item.strip())
                save_entities(user_entities)
            except: pass
        else:
            bot_reply = raw_ai_output

    except Exception as e:
        err = str(e).lower()
        # Diagnostic Interceptor for Phase 4
        if any(code in err for code in ["401", "403", "api_key_invalid"]):
            return {"error": True, "error_type": "invalid_key", "message": "Invalid API Key"}
        if any(code in err for code in ["429", "quota"]):
            return {"error": True, "error_type": "quota_exceeded", "message": "Quota Exceeded"}
        return {"error": True, "error_type": "general", "message": str(e)}

    # 5. POST-PROCESSING & BACKGROUND TASKS
    chat_history.append(f"User: {user_text}")
    chat_history.append(f"AI: {bot_reply}")
    
    # Store to Vector DB
    if len(user_text.split()) > 2:
        store_in_long_term_memory(user_text, bot_reply, emotion_tag, payload.session_id)
    
    # Trigger Summarizer (Phase 2)
    if len(chat_history) > 0 and len(chat_history) % 10 == 0:
        threading.Thread(target=summarize_session, args=(chat_history.copy(), payload.api_key)).start()

    return {
        "error": False,
        "response": bot_reply,
        "emotion_tag": emotion_tag,
        "entities": load_entities(),
        "emotion_arc": emotion_history
    }

@app.post("/forget-entity")
async def forget_entity(payload: DeleteEntityPayload):
    entities = load_entities()
    if payload.category in entities and payload.item in entities[payload.category]:
        entities[payload.category].remove(payload.item)
        save_entities(entities)
        return {"status": "ok", "updated_entities": entities}
    raise HTTPException(status_code=404, detail="Entity not found")

@app.post("/reset")
async def reset_endpoint():
    global chat_history, emotion_history
    chat_history, emotion_history = [], []
    wipe_all_memory()
    return {"status": "ok", "message": "Memory wiped."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)