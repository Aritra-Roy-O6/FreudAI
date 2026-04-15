import os
import json
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import uuid

# ==========================================
# STEP 1: STORAGE ENVIRONMENT INITIALIZATION
# ==========================================

# Set the path to the root folder where memory_manager.py lives
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ENTITY_FILE = os.path.join(APP_DIR, "entity_store.json")
MEMORY_DIR = os.path.join(APP_DIR, 'chroma_db')

os.makedirs(MEMORY_DIR, exist_ok=True)

print("Initializing Storage Environment...")

# 1A. Setup the Embedding Function (Gemini API Strict)
# This enforces that Render MUST have GEMINI_API_KEY set
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    # This will print a bright error in your Render Logs so you know exactly what went wrong
    print("CRITICAL ERROR: GEMINI_API_KEY is missing from Render Environment Variables!")
    embedding_func = None
else:
    try:
        # Note: This specifically uses the legacy/standard google-generativeai package
        embedding_func = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key)
        print("✓ Using Gemini Embeddings API")
    except Exception as e:
        print(f"⚠ Gemini API init failed: {e}")
        embedding_func = None

# 1B. Initialize ChromaDB
# We use PersistentClient so the AI's memory survives even if you restart your computer or server.
try:
    chroma_client = chromadb.PersistentClient(path=MEMORY_DIR)
    
    # We MUST pass an embedding function. If Gemini failed, we pass None and Chroma will 
    # try its default (which will crash), but our error logs above will tell us why.
    if embedding_func:
        memory_collection = chroma_client.get_or_create_collection(
            name="freud_cloud_memory",
            embedding_function=embedding_func
        )
        print(f"✓ ChromaDB running at {MEMORY_DIR}")
        print(f"✓ Entity tracking will be saved to {ENTITY_FILE}")
    else:
        print("⚠ WARNING: Running without a valid embedding function. Memory will fail.")
        memory_collection = None
        
except Exception as e:
    print(f"[Memory Error] ChromaDB Init Failed: {e}")
    memory_collection = None

# ==========================================
# STEP 2: ENTITY LEDGER LOGIC (JSON)
# ==========================================

def load_entities() -> dict:
    """Loads the entity ledger from disk, or initializes a new one if it doesn't exist."""
    if not os.path.exists(ENTITY_FILE):
        # Initialize the blank memory schema — MUST be lists, not dicts
        return {
            "people": [],
            "incidents": [],
            "preferences": []
        }
    with open(ENTITY_FILE, "r") as f:
        return json.load(f)

def save_entities(entities_dict: dict):
    """Saves the fully updated entity ledger back to disk."""
    with open(ENTITY_FILE, "w") as f:
        json.dump(entities_dict, f, indent=4)

def update_entity(category: str, key: str, value: dict):
    """Helper function to safely update a specific entity and save the file."""
    ledger = load_entities()
    if category in ledger:
        ledger[category][key] = value
        save_entities(ledger)
        print(f"Success: Entity updated -> [{category}] {key}")
    else:
        print(f"Error: Category '{category}' not found in schema.")

# ==========================================
# STEP 3: CHROMADB READ/WRITE LOGIC
# ==========================================

def store_in_long_term_memory(user_text: str, bot_response: str, emotion_tag: str, session_id: str = "default_user"):
    """
    Saves the conversational turn into the vector database.
    Safely handles cases where memory_collection initialization failed.
    """
    if memory_collection is None:
        print("⚠ Warning: Memory collection not initialized, skipping storage")
        return
    
    # We combine both sides of the conversation so the AI remembers context and its own advice
    memory_chunk = f"User: {user_text}\nAI: {bot_response}"
    doc_id = str(uuid.uuid4())
    
    metadata = {
        "session_id": session_id,
        "emotion_tag": emotion_tag,
        "type": "conversation_turn"
    }
    
    try:
        memory_collection.add(
            documents=[memory_chunk],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(f"✓ Memory stored -> ID: {doc_id[:8]}... | Tag: {emotion_tag}")
    except Exception as e:
        print(f"⚠ Error storing memory: {e}")

def retrieve_relevant_memory(query_text: str, n_results: int = 1) -> list:
    """
    Searches the vector database for past context relevant to the current user input.
    Safely handles cases where memory_collection initialization failed.
    """
    if memory_collection is None:
        print("⚠ Warning: Memory collection not initialized, returning empty results")
        return []
    
    # If the database is empty, return nothing to prevent errors
    if memory_collection.count() == 0:
        return []

    try:
        # Query ChromaDB for the most semantically similar memories
        results = memory_collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        # Extract and return the actual text documents
        if results and 'documents' in results and results['documents'][0]:
            return results['documents'][0]
        return []
    except Exception as e:
        print(f"⚠ Error retrieving memory: {e}")
        return []

# ==========================================
# STEP 4: SHORT-TERM CONTEXT WINDOW
# ==========================================

def get_short_term_context(full_history: list, window_size: int = 8) -> list:
    """
    Slices the full chat history to return only the most recent N turns.
    Prevents the LLM prompt from overflowing while maintaining immediate continuity.
    """
    return full_history[-window_size:]


# --- Temporary Test Block for Phase 2 ---
if __name__ == "__main__":
    import time
    print("\n--- Testing Step 3 & 4: Vector Memory ---")
    
    # 1. Store a memory (Simulating a past conversation)
    store_in_long_term_memory(
        user_text="Everyone else here just seems to have it figured out. I don't even know why I'm at uni honestly.",
        bot_response="That feeling of not belonging, like everyone else got a manual you didn't. Is that just about uni?",
        emotion_tag="[IMPLICIT_DISTRESS]"
    )
    
    time.sleep(1) # Brief pause to ensure the database writes successfully
    
    # 2. Retrieve a memory (Simulating the user mentioning school later)
    print("\nSimulating user saying: 'School is just too much right now.'")
    print("Searching database for semantic matches...\n")
    
    recalled_memories = retrieve_relevant_memory("school")
    
    if recalled_memories:
        print("✅ MEMORY RECALLED SUCCESSFULLY:")
        for mem in recalled_memories:
            print(f" -> {mem}")
    else:
        print("❌ Retrieval failed.")

# ==========================================
# STEP 5: MEMORY WIPE (NEW CHAT)
# ==========================================
def wipe_all_memory():
    """Completely erases the JSON entity ledger and all ChromaDB vectors."""
    # 1. Erase the Entity Ledger
    if os.path.exists(ENTITY_FILE):
        os.remove(ENTITY_FILE)
        
    # 2. Erase the Vector Database
    if memory_collection is not None:
        try:
            all_docs = memory_collection.get()
            if all_docs and all_docs.get('ids'):
                memory_collection.delete(ids=all_docs['ids'])
                print("✓ All vector memories wiped")
        except Exception as e:
            print(f"⚠ Error wiping memory: {e}")
    
    print("✓ Success: Total memory wipe completed.")