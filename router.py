# router.py
import re

def priority_router(text: str, lexical_score: float, is_sarcastic: bool, is_implicit: bool) -> str:
    """
    Scores input across the 6 Track B emotion axes and routes to the highest priority tag.
    Hierarchy: Crisis > Sarcasm > Implicit > Overload > Explicit > Numbing > Neutral
    """
    text_lower = text.lower()
    
    # 1. CRISIS SIGNAL (Highest Priority)
    crisis_keywords = ["kill", "end it", "don't want to be here", "harm", "can't take it anymore"]
    if any(word in text_lower for word in crisis_keywords):
        return "[CRISIS_SIGNAL_ESCALATE]"
        
    # 2. SARCASM / DEFLECTION
    if is_sarcastic:
        return "[SARCASM_DEFLECTION]"
        
    # 3. IMPLICIT DISTRESS (Masked pain, imposter syndrome)
    if is_implicit:
        return "[IMPLICIT_DISTRESS]"
        
    # 4. COGNITIVE OVERLOAD (Multiple stressors tangled together)
    overload_patterns = [r"and on top of", r"too much", r"everything is", r"also have to"]
    if any(re.search(p, text_lower) for p in overload_patterns) or len(text.split("and")) > 3:
        return "[COGNITIVE_OVERLOAD]"

    # 5. EXPLICIT DISTRESS (Direct venting / stated pain)
    if lexical_score < -0.5:
        return "[EXPLICIT_DISTRESS]"
        
    # 6. EMOTIONAL NUMBING (Flatness, dissociation)
    numbing_patterns = [r"numb", r"feel nothing", r"stopped caring", r"whatever happens"]
    if any(re.search(p, text_lower) for p in numbing_patterns):
        return "[EMOTIONAL_NUMBING]"
        
    # Default Fallback
    return "[NEUTRAL_CONVERSATIONAL]"