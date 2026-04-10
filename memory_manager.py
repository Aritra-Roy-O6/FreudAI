import os
import json
import chromadb
from chromadb.utils import embedding_functions
import uuid

# ==========================================
# STEP 1: STORAGE ENVIRONMENT INITIALIZATION
# ==========================================

# Define where our memory will live on your local machine
ENTITY_FILE = "entity_store.json"
CHROMA_DB_PATH = "./chroma_db"

print("Initializing Storage Environment...")

# 1A. Initialize ChromaDB (Long-Term Vector Memory)
# We use PersistentClient so the AI's memory survives even if you restart your computer or server.
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# 1B. Setup the Embedding Function
# This automatically converts text into math (vectors) so the AI can search by "meaning" later.
embedding_func = embedding_functions.DefaultEmbeddingFunction()

# 1C. Create the Database Collection
# Think of a 'collection' like a specific folder or table dedicated to storing the user's conversational history.
memory_collection = chroma_client.get_or_create_collection(
    name="user_long_term_memory",
    embedding_function=embedding_func
)

print(f"Success: ChromaDB running at {CHROMA_DB_PATH}")
print(f"Success: Entity tracking will be saved to {ENTITY_FILE}")

# ==========================================
# STEP 2: ENTITY LEDGER LOGIC (JSON)
# ==========================================

def load_entities() -> dict:
    """Loads the entity ledger from disk, or initializes a new one if it doesn't exist."""
    if not os.path.exists(ENTITY_FILE):
        # Initialize the blank memory schema
        return {
            "people": {},     
            "incidents": {},  
            "preferences": {} 
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
    """
    # We combine both sides of the conversation so the AI remembers context and its own advice
    memory_chunk = f"User: {user_text}\nAI: {bot_response}"
    doc_id = str(uuid.uuid4())
    
    metadata = {
        "session_id": session_id,
        "emotion_tag": emotion_tag,
        "type": "conversation_turn"
    }
    
    memory_collection.add(
        documents=[memory_chunk],
        metadatas=[metadata],
        ids=[doc_id]
    )
    print(f"Success: Memory stored -> ID: {doc_id[:8]}... | Tag: {emotion_tag}")

def retrieve_relevant_memory(query_text: str, n_results: int = 1) -> list:
    """
    Searches the vector database for past context relevant to the current user input.
    """
    # If the database is empty, return nothing to prevent errors
    if memory_collection.count() == 0:
        return []

    # Query ChromaDB for the most semantically similar memories
    results = memory_collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    # Extract and return the actual text documents
    if results and 'documents' in results and results['documents'][0]:
        return results['documents'][0]
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