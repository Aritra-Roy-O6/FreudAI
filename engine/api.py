import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re
from rag_engine import HybridRAGEngine
import json

# Import our custom modules from Phases 1-4
from detectors import lexical_scan, sarcasm_probe, implicit_distress_flag, crisis_flag_semantic
from router import priority_router
from llm_core import generate_response
from memory_manager import (
    store_in_long_term_memory,
    get_short_term_context,
    load_entities,
    save_entities,
    memory_collection,
    wipe_all_memory
)
from rag_engine import HybridRAGEngine

app = FastAPI(title="FreudAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows any frontend to connect (perfect for local testing)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserMessagePayload(BaseModel):
    message: str
    session_id: str = "default_user"

# Global state for short-term memory (in-memory list for MVP)
chat_history = []
emotion_history = []
engine = HybridRAGEngine()

import json
import re

@app.post("/chat")
async def chat_endpoint(payload: UserMessagePayload):
    global chat_history, emotion_history
    user_text = payload.message
    session_id = payload.session_id

    # 1. EMOTION DETECTION & ROUTING (The Brain)
    lex_score = lexical_scan(user_text)
    is_sarcastic, _ = sarcasm_probe(user_text, lex_score)
    is_implicit, _ = implicit_distress_flag(user_text)
    is_crisis_semantic, _ = crisis_flag_semantic(user_text)
    emotion_tag = priority_router(user_text, lex_score, is_sarcastic, is_implicit, is_crisis_semantic)

    # 2. RETRIEVE KNOWLEDGE & CONTEXT
    user_entities = load_entities() # Load existing JSON data
    retrieved_memories = engine.retrieve(user_text, top_k=3)
    short_term_context = "\n".join(get_short_term_context(chat_history, window_size=8))

    # 3. EMOTIONAL TRAJECTORY & REGRESSION NOTE
    emotion_history.append(emotion_tag)
    if len(emotion_history) > 3: emotion_history.pop(0)
    
    regression_note = ""
    if len(emotion_history) >= 2:
        prev, curr = emotion_history[-2], emotion_history[-1]
        if prev in ["[NEUTRAL_CONVERSATIONAL]", "[EMOTIONAL_NUMBING]"] and curr in ["[CRISIS_SIGNAL_ESCALATE]", "[EXPLICIT_DISTRESS]", "[COGNITIVE_OVERLOAD]"]:
            regression_note = f"\n[EMOTIONAL_REGRESSION: User shifted from {prev} to {curr}.]"

    # 4. COREFERENCE RESOLUTION (Resolving 'she' or 'it')
    def resolve_coreference(text, entities):
        pronoun_pattern = re.compile(r'\b(he|she|him|her|they|them|it)\b', re.IGNORECASE)
        if pronoun_pattern.search(text):
            # Flatten all entity lists into one searchable list
            all_known = [item for sublist in entities.values() for item in sublist]
            if all_known:
                return f"{text} [Context: Pronouns likely refer to '{all_known[-1]}']"
        return text

    resolved_user_text = resolve_coreference(user_text, user_entities)

    # 5. GENERATE RESPONSE & EXTRACT ENTITIES
    try:
        raw_ai_output = generate_response(
            user_text=resolved_user_text,
            emotion_tag=emotion_tag,
            retrieved_memories=retrieved_memories,
            user_entities=user_entities,
            short_term_history=short_term_context,
            regression_note=regression_note
        )

        # SEPARATE THE CHAT FROM THE DATA
        if "ENTITIES:" in raw_ai_output:
            # Split on the LAST occurrence so any mention of 'ENTITIES:' in
            # the chat text doesn't corrupt the JSON block.
            split_idx = raw_ai_output.rfind("ENTITIES:")
            bot_reply = raw_ai_output[:split_idx].strip()
            raw_json   = raw_ai_output[split_idx + len("ENTITIES:"):].strip()

            # Gemini often wraps JSON in markdown code fences — strip them
            raw_json = re.sub(r'^```(?:json)?\s*', '', raw_json, flags=re.IGNORECASE)
            raw_json = re.sub(r'\s*```$', '', raw_json)
            raw_json = raw_json.strip()

            try:
                extracted_data = json.loads(raw_json)
                for cat, items in extracted_data.items():
                    if cat not in user_entities:
                        user_entities[cat] = []
                    if not isinstance(user_entities[cat], list):
                        user_entities[cat] = []
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, str) and item.strip():
                                if item.lower() not in [x.lower() for x in user_entities[cat]]:
                                    user_entities[cat].append(item.strip())
                save_entities(user_entities)
                print(f"[Entities saved] {user_entities}")
            except Exception as parse_err:
                print(f"[Entity parse ERROR] {parse_err}  |  raw_json was: {raw_json!r}")
        else:
            bot_reply = raw_ai_output
            print("[WARN] LLM output had no ENTITIES: block — sidebar will not update.")

    except Exception as e:
        print(f"Error: {e}")
        bot_reply = "I'm listening. Tell me more about that?"

    # 6. UPDATE HISTORY & PERSISTENT MEMORY
    chat_history.append(f"User: {user_text}")
    chat_history.append(f"FreudAI: {bot_reply}")

    # Store updated entities and vectors
    if len(user_text.split()) > 2:
        store_in_long_term_memory(user_text, bot_reply, emotion_tag, session_id)

    # 7. RETURN TO FRONTEND
    # Reload from disk to guarantee the response carries the absolute freshest state
    fresh_entities = load_entities()
    print(f"[Entities → frontend] {fresh_entities}")
    return {
        "response": bot_reply,
        "emotion_tag": emotion_tag,
        "entities": fresh_entities,
        "emotion_arc": emotion_history
    }

@app.post("/reset")
async def reset_endpoint():
    """Wipes all server-side state: chat history, emotion arc, and persistent memory."""
    global chat_history, emotion_history
    chat_history    = []
    emotion_history = []
    wipe_all_memory()   # clears entity_store.json + ChromaDB vectors
    print("[Reset] All memory and history cleared.")
    return {"status": "ok", "message": "Memory wiped. Fresh start."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)