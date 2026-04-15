import os
import google.generativeai as genai

# ==========================================
# 1. SETUP GEMINI CLASSIFIER (Cloud-Friendly)
# ==========================================
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    print("CRITICAL: API Key missing for Semantic Router!")

# We use Flash because it is insanely fast for classification
MODEL_NAME = 'models/gemini-2.5-flash'

# ==========================================
# 2. EMOTION LABELS (Your exact definitions)
# ==========================================
EMOTION_LABELS: dict[str, str] = {
    "[EXPLICIT_DISTRESS]":
        "The person is expressing explicit emotional pain, grief, severe anxiety, "
        "intense sadness, feeling worthless, or distress caused by family conflict, "
        "arguments, or a toxic home environment.",

    "[IMPLICIT_DISTRESS]":
        "The person is masking or minimizing their pain, comparing themselves "
        "unfavorably to others, feeling disconnected, numb, or subtly uneasy "
        "without openly admitting they are struggling.",

    "[COGNITIVE_OVERLOAD]":
        "The person is overwhelmed because multiple serious stressors — such as "
        "exams, work, family problems, or responsibilities — are hitting them "
        "simultaneously and they cannot cope.",

    "[CRISIS_SIGNAL_ESCALATE]":
        "The person is expressing suicidal ideation, an active desire to die, "
        "intent to commit self-harm, or a concrete plan to end their own life. "
        "They may mention methods of self-harm, say they want to kill themselves, "
        "or express that they see no reason to continue living. This is an "
        "immediate life-threatening emergency.",

    "[EMOTIONAL_NUMBING]":
        "The person feels flat, empty, detached, or unable to feel any emotion "
        "at all, even when they objectively should.",

    "[SARCASM_DEFLECTION]":
        "The person is using sarcasm, dark humour, or irony to deflect from "
        "an underlying negative emotional state.",

    "[NEUTRAL_CONVERSATIONAL]":
        "The person is having a normal, calm conversation — sharing updates, "
        "asking general questions, or chatting without any significant "
        "emotional weight.",
}

# Build the prompt template once
_categories_text = "\n".join([f"{k}: {v}" for k, v in EMOTION_LABELS.items()])
_PROMPT_TEMPLATE = f"""You are a highly accurate psychological routing system. 
Classify the user's message into EXACTLY ONE of the following categories based on the descriptions provided.
Return ONLY the exact bracketed tag (e.g., [NEUTRAL_CONVERSATIONAL]) and absolutely nothing else.

Categories:
{_categories_text}

User Message: """

# ==========================================
# 3. THE ROUTING ENGINE
# ==========================================
def priority_router(user_text: str, lex_score: float, is_sarcastic: bool, is_implicit: bool, is_crisis_semantic: bool = False) -> str:
    """
    Hybrid routing: Semantic Anchor Safety Layer → Gemini ZS Classification → Fallback.
    """

    # ── 1. RED LINE: Semantic Crisis Anchors ──────────────────────
    if is_crisis_semantic:
        print(f"Router → [CRISIS_SIGNAL_ESCALATE]  (SEMANTIC CRISIS DETECTED)")
        return "[CRISIS_SIGNAL_ESCALATE]"

    # ── 2. Sarcasm Override ────────────────────────────────────────────
    if is_sarcastic:
        return "[SARCASM_DEFLECTION]"

    # ── 3. Graceful degradation if API fails ───────
    if not api_key:
        return "[IMPLICIT_DISTRESS]" if is_implicit else "[NEUTRAL_CONVERSATIONAL]"

    # ── 4. Zero-Shot Classification via Gemini (The LLM Math) ────────────────────
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        # We set temperature to 0.0 so the LLM acts purely analytically, not creatively
        response = model.generate_content(
            _PROMPT_TEMPLATE + f'"{user_text}"',
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=15
            )
        )
        
        best_emotion = response.text.strip()
        
        # Validation: Make sure the LLM actually picked a valid tag
        if best_emotion in EMOTION_LABELS:
            print(f"Router → {best_emotion}  (via Gemini ZS)")
            return best_emotion
        else:
            print(f"Router Warning: Invalid tag returned '{best_emotion}'. Falling back.")
            
    except Exception as e:
        print(f"Router ZS API Error: {e}")

    # ── 5. Low-confidence/Error fallback ────────────────────────────────────
    return "[IMPLICIT_DISTRESS]" if is_implicit else "[NEUTRAL_CONVERSATIONAL]"