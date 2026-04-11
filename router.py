from transformers import pipeline

# ==========================================
# 1. INITIALIZE ZERO-SHOT NLI CLASSIFIER
# ==========================================
# cross-encoder/nli-deberta-v3-small is ~180 MB, CPU-friendly, and far more
# generalizable than cosine-similarity over fixed anchor sentences.
# It treats each LABEL_HYPOTHESIS as a natural-language premise and scores
# whether the user's message "entails" it — no hardcoded examples needed.
try:
    print("Loading Zero-Shot Semantic Router (cross-encoder/nli-deberta-v3-small)...")
    _classifier = pipeline(
        "zero-shot-classification",
        model="cross-encoder/nli-deberta-v3-small",
        device=-1  # CPU; change to 0 if you have a CUDA GPU
    )
    print("Semantic Router Online.")
except Exception as e:
    print(f"Error loading ZS model: {e}")
    _classifier = None

# ==========================================
# 3. EMOTION LABELS  (plain English — edit freely)
# ==========================================
# These are NOT anchor sentences. They are hypothesis *descriptions* fed
# directly to the NLI model.  The model figures out on its own whether the
# user's text entails each one.  Add, rename, or rewrite any entry and the
# router automatically adapts — no retraining required.
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

# Pre-extract the ordered label list once so we don't rebuild it every call
_LABEL_KEYS   = list(EMOTION_LABELS.keys())
_LABEL_TEXTS  = [EMOTION_LABELS[k] for k in _LABEL_KEYS]

# ==========================================
# 4. THE ROUTING ENGINE
# ==========================================
def priority_router(user_text: str, lex_score: float, is_sarcastic: bool, is_implicit: bool, is_crisis_semantic: bool = False) -> str:
    """
    Hybrid routing: Semantic Anchor Safety Layer → NLI Semantic Math → Fallback.

    Order of evaluation (CRITICAL — do NOT rearrange):
      1. CRISIS semantic override — bypasses NLI math if crisis is high confidence
      2. Sarcasm override         — lexical detector confirmed deflection
      3. NLI Classification       — the general-case semantic math
      4. Low-confidence fallback  — implicit only if nothing else fired
    """

    # ── 1. RED LINE: Semantic Crisis Anchors ──────────────────────
    if is_crisis_semantic:
        print(f"Router → [CRISIS_SIGNAL_ESCALATE]  (SEMANTIC CRISIS DETECTED)")
        return "[CRISIS_SIGNAL_ESCALATE]"

    # ── 2. Sarcasm Override ────────────────────────────────────────────
    if is_sarcastic:
        return "[SARCASM_DEFLECTION]"

    # ── 3. Graceful degradation if the NLI model failed to load ───────
    if _classifier is None:
        return "[IMPLICIT_DISTRESS]" if is_implicit else "[NEUTRAL_CONVERSATIONAL]"

    # ── 4. Zero-Shot NLI Classification (The Math) ────────────────────
    # The model scores P(entailment | user_text, hypothesis) for every label.
    result = _classifier(
        user_text,
        candidate_labels=_LABEL_TEXTS,
        hypothesis_template="{}",   # hypothesis IS the label description
        multi_label=False           # pick the single most applicable label
    )

    # Map the winning label *text* back to its emotion tag
    winning_text  = result["labels"][0]
    winning_score = result["scores"][0]
    best_emotion  = _LABEL_KEYS[_LABEL_TEXTS.index(winning_text)]

    print(f"Router → {best_emotion}  (confidence: {winning_score:.3f})")

    # ── 5. Low-confidence fallback ────────────────────────────────────
    # If even the top label barely beats random (< 0.30 after softmax
    # over 7 classes), trust the lexical implicit-distress flag as a
    # safety net — but ONLY if the best emotion isn't already crisis or
    # explicit distress (never downgrade a severe signal).
    if winning_score < 0.30:
        if best_emotion in ("[CRISIS_SIGNAL_ESCALATE]", "[EXPLICIT_DISTRESS]"):
            return best_emotion          # never downgrade severity
        if is_implicit:
            return "[IMPLICIT_DISTRESS]"
        return "[NEUTRAL_CONVERSATIONAL]"

    return best_emotion