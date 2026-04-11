import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.messages import SystemMessage, HumanMessage

# 1. Load API Key
load_dotenv()

# 2. Initialize the Gemini Engine
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7, # Raised to prevent repetitive phrasing and semantic echo
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
    short_term_history: str = "",
    regression_note: str = ""
) -> str:
    
    # LAYER 1: PERSONA ANCHOR
    layer_1_persona = """
    You are an empathetic, highly perceptive, non-clinical listener named FreudAI. 
    Your tone must be warm, conversational, and inquisitive. You do not fix problems; you untangle them.
    """
    
    # LAYER 2: USER PROFILE INJECTION
    entities_str = json.dumps(user_entities, indent=2) if user_entities else "No entities known yet."
    layer_2_entities = f"""
    KNOWN USER ENTITIES & RELATIONSHIPS:
    {entities_str}
    (Use these details seamlessly. Do not ask for information you already have.)
    """
    
    # LAYER 3: RETRIEVED CONTEXT (RAG)
    memories_str = "\n".join([f"- {mem}" for mem in retrieved_memories]) if retrieved_memories else "No relevant past context retrieved."
    layer_3_rag = f"""
    RELEVANT PAST CONVERSATIONS:
    {memories_str}
    """
    
   # LAYER 4: EMOTIONAL STATE FRAME & ROUTING INSTRUCTIONS
    sub_prompts = {
        "[EXPLICIT_DISTRESS]": "Validate the pain deeply. Ask an open question about its core.",
        "[IMPLICIT_DISTRESS]": "The user is masking pain or comparing themselves. Gently name the subtext without projecting.",
        "[SARCASM_DEFLECTION]": "The user is using sarcasm or irony. Acknowledge the contrast between their positive words and the negative reality.",
        "[COGNITIVE_OVERLOAD]": "The user is overwhelmed by multiple stressors. Help them untangle by isolating one specific thread.",
        "[EMOTIONAL_NUMBING]": "Acknowledge the flatness. Invite them to explore the numbness without forcing them to 'feel'.",
        "[CRISIS_SIGNAL_ESCALATE]": "CRITICAL: Validate their worth directly, acknowledge severe pain, and gently suggest professional support.",
        "[NEUTRAL_CONVERSATIONAL]": "Maintain a warm, casual, but attentive conversational flow."
    }
    routing_instruction = sub_prompts.get(emotion_tag, sub_prompts["[NEUTRAL_CONVERSATIONAL]"])
    
    layer_4_emotion = f"""
    CURRENT EMOTIONAL STATE: {emotion_tag}
    ROUTING INSTRUCTION: {routing_instruction}
    {regression_note}  
    """
    
    # LAYER 5: ANTI-GENERIC CONSTRAINT & SYNTHESIS
    layer_5_constraint = """
    HARD CONSTRAINT:
    - ZERO generic fallbacks or platitudes (Never say "I'm sorry you're going through this" or "That sounds hard").
    - Ground your response in the specific reality the user just shared. 
    - SYNTHESIS OVER PARROTING: Never echo the user's exact phrasing verbatim. Instead, reframe, synthesize, and connect what they shared.
    - NATURAL CALLBACKS: You are encouraged to reference prior context to show you are tracking the conversation over time, but do it naturally.
    - AVOID clinical openers like "You mentioned..." or "It sounds like..."
    - INFER OBVIOUS CONTEXT: If the user uses a metaphor or vague reference (like "a piece of paper"), immediately infer what it means from the short-term history (e.g., the exams) instead of asking them to clarify obvious details.
    - NO SELF-REPETITION: You are strictly forbidden from repeating sentences or phrases you used in your previous turns. Always generate a fresh response.
    """
    
    # NEW: Extraction Instruction
    extraction_instruction = """
    [SYSTEM METADATA INSTRUCTION]
    At the very end of your response, you MUST provide a structured list of entities mentioned in the conversation so far. 
    Format it exactly like this:
    ENTITIES: {"people": ["Name"], "incidents": ["event"], "preferences": ["detail"]}

    Rules for extraction (MANDATORY — failure to follow these is an error):
    - CASE-INSENSITIVE NAMES: Always extract names regardless of how they are written. 
      If the user writes 'sophia', 'SOPHIA', or 'Sophia', you MUST include 'Sophia' (Title Case) in the output.
      Treat 'mom', 'dad', 'brother', 'sister' as named roles and include them as-is (e.g., 'Mom', 'Dad').
    - Identify key stressors, objects, or events (e.g., 'exams', 'the paper', 'parents fighting').
    - NO SELF-REPETITION: Do not re-use any sentence or phrase that appears earlier in your response.
    - Do not explain this block to the user. Simply append it silently at the very end.
    """

     # ==========================================

    # Combine everything
    master_system_prompt = f"{layer_1_persona}\n{layer_2_entities}\n{layer_3_rag}\n{layer_4_emotion}\n{layer_5_constraint}\n{extraction_instruction}"


    context_prefix = f"Recent Conversation Context:\n{short_term_history}\n\n" if short_term_history else ""
    final_user_prompt = f"{context_prefix}User's Latest Message: {user_text}"

    messages = [
        SystemMessage(content=master_system_prompt),
        HumanMessage(content=final_user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        print("\n--- GENERATION SUCCESS ---")
        print(f"Finish Reason: {response.response_metadata.get('finish_reason', 'UNKNOWN')}")
        print("--------------------------\n")
        return response.content
    except Exception as e:
        print(f"\n--- LLM GENERATION ERROR ---\n{e}\n----------------------------\n")
        return "I'm having trouble processing that right now, but I am listening. Could you tell me more?"