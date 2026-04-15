import os
import sys
import json
import chromadb
import uuid
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ==========================================
# STEP 4: ENCRYPTION LAYER (Privacy Vault)
# ==========================================
class PrivacyVault:
    def __init__(self):
        self.key = self._generate_machine_key()
        self.fernet = Fernet(self.key)

    def _generate_machine_key(self):
        """Generates a hardware-bound key using the machine's unique ID."""
        # node returns the hardware address (MAC) of the machine
        machine_id = str(uuid.getnode()).encode()
        
        # Use PBKDF2 to turn the machine ID into a valid 32-byte AES key
        salt = b'freud_salt_123' # Static salt is fine since ID is unique per machine
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(machine_id))
        return key

    def encrypt_data(self, data_str):
        return self.fernet.encrypt(data_str.encode()).decode()

    def decrypt_data(self, encrypted_str):
        try:
            return self.fernet.decrypt(encrypted_str.encode()).decode()
        except Exception:
            # If decryption fails (e.g., file tampered with), return empty object
            return "{}"

vault = PrivacyVault()

# ==========================================
# STEP 1: RELOCATE PERSISTENCE
# ==========================================
def get_app_data_dir():
    if os.name == 'nt':
        base_dir = os.environ.get('APPDATA')
    else:
        base_dir = os.path.expanduser('~/Library/Application Support' if sys.platform == 'darwin' else '~/.config')
    
    app_dir = os.path.join(base_dir, 'FreudAI')
    os.makedirs(app_dir, exist_ok=True)
    return app_dir

APP_DIR = get_app_data_dir()
MEMORY_DIR = os.path.join(APP_DIR, 'chroma_db')
ENTITY_FILE = os.path.join(APP_DIR, 'entity_store.json.enc') # Changed extension to .enc

# Init ChromaDB
try:
    chroma_client = chromadb.PersistentClient(path=MEMORY_DIR)
    memory_collection = chroma_client.get_or_create_collection(name="freud_memories")
except Exception as e:
    print(f"[Memory Error] ChromaDB Init Failed: {e}")
    memory_collection = None

# ==========================================
# CORE MEMORY FUNCTIONS (NOW WITH ENCRYPTION)
# ==========================================
def load_entities():
    if os.path.exists(ENTITY_FILE):
        try:
            with open(ENTITY_FILE, 'r', encoding='utf-8') as f:
                encrypted_blob = f.read()
                decrypted_json = vault.decrypt_data(encrypted_blob)
                return json.loads(decrypted_json)
        except Exception as e:
            print(f"[Vault Error] Failed to decrypt ledger: {e}")
    return {}

def save_entities(entities):
    try:
        json_str = json.dumps(entities, indent=4)
        encrypted_blob = vault.encrypt_data(json_str)
        with open(ENTITY_FILE, 'w', encoding='utf-8') as f:
            f.write(encrypted_blob)
    except Exception as e:
        print(f"[Vault Error] Encryption failed: {e}")

def store_in_long_term_memory(user_text, bot_reply, emotion_tag, session_id="default"):
    if not memory_collection: return
    doc_id = str(uuid.uuid4())
    memory_text = f"User: {user_text} | AI: {bot_reply}"
    memory_collection.add(
        documents=[memory_text],
        metadatas=[{"session_id": session_id, "emotion": emotion_tag}],
        ids=[doc_id]
    )

def get_short_term_context(chat_history, window_size=8):
    return chat_history[-window_size:]

def wipe_all_memory():
    if os.path.exists(ENTITY_FILE):
        os.remove(ENTITY_FILE)
    if chroma_client:
        try:
            chroma_client.delete_collection("freud_memories")
            global memory_collection
            memory_collection = chroma_client.get_or_create_collection("freud_memories")
        except: pass

# ==========================================
# STEP 2: SUMMARIZER
# ==========================================
def summarize_session(chat_history, api_key):
    if not chat_history or len(chat_history) < 6: return 
    history_text = "\n".join(chat_history)
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.1)
        prompt = PromptTemplate(
            input_variables=["history"],
            template="Extract permanent user traits, relationships, and stressors from this chat as a JSON object:\n{history}"
        )
        chain = prompt | llm
        response = chain.invoke({"history": history_text})
        raw_json = response.content.replace('`{3}json', '').replace('`{3}', '').strip()
        new_traits = json.loads(raw_json)
        
        existing_entities = load_entities()
        for cat, traits in new_traits.items():
            if cat not in existing_entities: existing_entities[cat] = []
            for t in traits:
                if t not in existing_entities[cat]: existing_entities[cat].append(t)
        save_entities(existing_entities)
    except Exception as e:
        print(f"[Summarizer Error] {e}")