import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.messages import SystemMessage, HumanMessage

# 1. Load API Key
load_dotenv()

# 2. Initialize the Gemini Engine (Removed max_tokens to prevent artificial choking)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4, 
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
)

def generate_response(
    user_text: str, 
    emotion_tag: str, 
    retrieved_memories: list[str], 
    user_entities: dict,
    short_term_history: str = ""
) -> str:
    
    # LAYER 1: PERSONA ANCHOR
    layer_1_persona = """
    You are an empathetic, highly perceptive, non-clinical listener. 
    Your tone must be warm, specific, and inquisitive. You do not fix problems; you untangle them.
    """
    
    # LAYER 2: USER PROFILE INJECTION
    entities_str = json.dumps(user_entities, indent=2) if user_entities else "No entities known yet."
    layer_2_entities = f"""
    KNOWN USER ENTITIES & RELATIONSHIPS:
    {entities_str}
    (Use these names and facts naturally. Do not ask for details you already know.)
    """
    
    # LAYER 3: RETRIEVED CONTEXT (RAG)
    memories_str = "\n".join([f"- {mem}" for mem in retrieved_memories]) if retrieved_memories else "No relevant past context retrieved."
    layer_3_rag = f"""
    RELEVANT PAST CONVERSATIONS:
    {memories_str}
    """
    
    # LAYER 4: EMOTIONAL STATE FRAME & ROUTING INSTRUCTIONS
    sub_prompts = {
        "[EXPLICIT_DISTRESS]": "Deep validation required. Ask an open question about the core of the pain.",
        "[IMPLICIT_DISTRESS]": "The user is masking their pain. Gently name the subtext without projecting, and probe the avoidance.",
        "[SARCASM_DEFLECTION]": "Name the deflection. Do not mirror false positivity. Contrast what they say with how they likely feel.",
        "[COGNITIVE_OVERLOAD]": "The user is tangled in multiple stressors. Help them untangle by prioritizing one specific thread.",
        "[EMOTIONAL_NUMBING]": "Acknowledge the flatness. Invite them to explore the numbness without pushing them to 'feel' immediately.",
        "[CRISIS_SIGNAL_ESCALATE]": "CRITICAL: Acknowledge their severe pain directly, validate their worth, and gently suggest professional support.",
        "[NEUTRAL_CONVERSATIONAL]": "Maintain a warm, casual, but attentive conversational flow."
    }
    routing_instruction = sub_prompts.get(emotion_tag, sub_prompts["[NEUTRAL_CONVERSATIONAL]"])
    
    layer_4_emotion = f"""
    CURRENT EMOTIONAL STATE: {emotion_tag}
    ROUTING INSTRUCTION: {routing_instruction}
    """
    
    # LAYER 5: ANTI-GENERIC CONSTRAINT
    layer_5_constraint = """
    HARD CONSTRAINT:
    - ZERO generic fallbacks. 
    - DO NOT use platitudes like "I'm sorry you're going through this" or "That sounds hard."
    - Every sentence must directly reference a specific word, entity, or concept the user just provided.
    """
    
    # ASSEMBLE THE MASTER PROMPT
    master_system_prompt = f"{layer_1_persona}\n{layer_2_entities}\n{layer_3_rag}\n{layer_4_emotion}\n{layer_5_constraint}"
    
    context_prefix = f"Recent Conversation Context:\n{short_term_history}\n\n" if short_term_history else ""
    final_user_prompt = f"{context_prefix}User's Latest Message: {user_text}"

    messages = [
        SystemMessage(content=master_system_prompt),
        HumanMessage(content=final_user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        
        # --- NEW DIAGNOSTIC PRINT ---
        print("\n--- GENERATION SUCCESS ---")
        print(f"Finish Reason: {response.response_metadata.get('finish_reason', 'UNKNOWN')}")
        print("--------------------------\n")
        
        return response.content
    except Exception as e:
        print(f"\n--- LLM GENERATION ERROR ---\n{e}\n----------------------------\n")
        return "I'm having trouble processing that right now, but I am listening. Could you tell me more?"