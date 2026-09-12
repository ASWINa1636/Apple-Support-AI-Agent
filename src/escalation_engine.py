"""
Escalation Engine: Decide whether a customer message should be auto-handled
or escalated to a human agent, with a stated reason.

Provides two approaches:
1. Rule-based heuristic (baseline)
2. LLM-based decision with confidence scoring (main system)
"""

import os
import re
from typing import Optional, Dict, Tuple


# Escalation rules
ESCALATION_KEYWORDS = {
    "legal_threat": [
        "lawyer", "attorney", "lawsuit", "sue", "legal action", "court",
        "consumer protection", "ftc", "bbb", "complaint", "regulatory",
    ],
    "safety_concern": [
        "fire", "burn", "smoke", "explode", "explosion", "injury", "hurt",
        "dangerous", "safety", "hazard", "overheating", "swelling",
    ],
    "explicit_escalation": [
        "supervisor", "manager", "escalate", "higher up", "someone else",
        "your boss", "complaint department", "formal complaint",
    ],
    "extreme_frustration": [
        "worst company", "never buying", "scam", "fraud", "stolen",
        "ridiculous", "unacceptable", "disgusted", "furious",
        "worst experience", "hate apple", "class action",
    ],
    "billing_dispute": [
        "unauthorized charge", "didn't authorize", "cancel subscription",
        "refund", "money back", "overcharged", "double charged",
        "charged twice", "billing error",
    ],
    "data_privacy": [
        "data breach", "hacked", "privacy", "personal data", "identity theft",
        "compromised", "unauthorized access", "leaked",
    ],
}

AUTO_HANDLE_PATTERNS = {
    "simple_question": [
        "how do i", "how to", "where can i", "what is", "can you help",
        "need help with", "having trouble", "not working",
    ],
    "positive_feedback": [
        "thank you", "thanks", "great", "awesome", "love", "appreciate",
        "helpful", "solved", "fixed", "working now",
    ],
    "status_check": [
        "update", "status", "progress", "when will", "how long",
    ],
}


class EscalationDecision:
    """Represents an escalation decision with reasoning."""
    
    def __init__(self, should_escalate: bool, reason: str, 
                 confidence: float = 0.0, category: str = ""):
        self.should_escalate = should_escalate
        self.reason = reason
        self.confidence = confidence
        self.category = category
    
    def to_dict(self) -> Dict:
        return {
            "should_escalate": self.should_escalate,
            "reason": self.reason,
            "confidence": self.confidence,
            "category": self.category,
        }
    
    def __repr__(self):
        action = "ESCALATE" if self.should_escalate else "AUTO-HANDLE"
        return f"[{action}] {self.reason} (confidence: {self.confidence:.2f})"


class RuleBasedEscalation:
    """Baseline: Rule-based escalation using keyword matching and heuristics."""
    
    def decide(self, customer_message: str, intent: str = "") -> EscalationDecision:
        """Make escalation decision based on rules."""
        text = customer_message.lower()
        
        # Check escalation keywords
        for category, keywords in ESCALATION_KEYWORDS.items():
            matched = [kw for kw in keywords if kw in text]
            if matched:
                reasons = {
                    "legal_threat": "Customer mentioned legal action or threats",
                    "safety_concern": "Message contains safety-related concerns",
                    "explicit_escalation": "Customer explicitly requested escalation",
                    "extreme_frustration": "Customer shows extreme frustration/anger",
                    "billing_dispute": "Billing dispute requiring account access",
                    "data_privacy": "Data privacy or security concern",
                }
                return EscalationDecision(
                    should_escalate=True,
                    reason=f"{reasons.get(category, 'Matched escalation pattern')}: "
                           f"detected [{', '.join(matched[:3])}]",
                    confidence=0.8,
                    category=category,
                )
        
        # Check intent-based escalation
        if intent in ["escalation_request"]:
            return EscalationDecision(
                should_escalate=True,
                reason="Intent classified as explicit escalation request",
                confidence=0.9,
                category="intent_based",
            )
        
        if intent in ["billing_charge"]:
            return EscalationDecision(
                should_escalate=True,
                reason="Billing issues require human agent with account access",
                confidence=0.7,
                category="intent_based",
            )
        
        # Check for multi-issue complexity
        issue_indicators = sum(1 for pattern_list in AUTO_HANDLE_PATTERNS.values()
                              for kw in pattern_list if kw in text)
        escalation_indicators = sum(1 for kw_list in ESCALATION_KEYWORDS.values()
                                   for kw in kw_list if kw in text)
        
        # Message length as a proxy for complexity
        if len(text.split()) > 50:
            return EscalationDecision(
                should_escalate=True,
                reason="Complex message (>50 words) likely needs human attention",
                confidence=0.6,
                category="complexity",
            )
        
        # ALL CAPS detection (anger signal)
        words = customer_message.split()
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)
        if caps_ratio > 0.4 and len(words) > 5:
            return EscalationDecision(
                should_escalate=True,
                reason=f"High proportion of ALL-CAPS words ({caps_ratio:.0%}) suggests frustration",
                confidence=0.65,
                category="sentiment",
            )
        
        # Default: auto-handle
        # Check if it matches auto-handle patterns
        for category, patterns in AUTO_HANDLE_PATTERNS.items():
            if any(p in text for p in patterns):
                return EscalationDecision(
                    should_escalate=False,
                    reason=f"Standard {category.replace('_', ' ')} — can be auto-handled "
                           f"with typical troubleshooting guidance",
                    confidence=0.75,
                    category=category,
                )
        
        # Fallback: auto-handle with lower confidence
        return EscalationDecision(
            should_escalate=False,
            reason="No escalation triggers detected; standard support query",
            confidence=0.5,
            category="default",
        )


class LLMEscalation:
    """Main System: LLM-based escalation decision via Groq."""
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: Optional[str] = None):
        from groq import Groq
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")
        self.client = Groq(api_key=self.api_key)
        self.model = model or os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
        # Also use rule-based as a fallback / signal
        self.rule_based = RuleBasedEscalation()
    
    def decide(self, customer_message: str, intent: str = "") -> EscalationDecision:
        """Make escalation decision using LLM."""
        # Get rule-based signal first
        rule_decision = self.rule_based.decide(customer_message, intent)
        
        prompt = f"""You are a customer support escalation system for Apple Support on Twitter.

Analyze this customer message and decide: should it be AUTO-HANDLED by an AI agent, or ESCALATED to a human agent?

Customer message: "{customer_message}"
Detected intent: {intent}
Rule-based signal: {"ESCALATE" if rule_decision.should_escalate else "AUTO-HANDLE"} ({rule_decision.reason})

Consider these escalation criteria:
- Safety concerns (device overheating, fire, injury)
- Legal threats or formal complaints
- Billing disputes requiring account access
- Data privacy/security concerns
- Extreme customer frustration or anger
- Complex multi-issue problems
- Customer explicitly requesting a human/supervisor
- Issues that cannot be resolved via Twitter DM

And these auto-handle criteria:
- Simple how-to questions
- Common troubleshooting (restart, update, reset settings)
- Status inquiries
- Positive feedback/thanks
- Standard technical issues with known solutions

Respond in this EXACT format (3 lines only):
DECISION: ESCALATE or AUTO-HANDLE
REASON: One sentence explaining why
CONFIDENCE: A number between 0.0 and 1.0"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an escalation decision system. Be concise and precise."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=150,
            )
            result = response.choices[0].message.content.strip()
            
            # Parse response
            lines = result.strip().split("\n")
            should_escalate = False
            reason = "LLM decision"
            confidence = 0.5
            
            for line in lines:
                line = line.strip()
                if line.upper().startswith("DECISION:"):
                    decision_text = line.split(":", 1)[1].strip().upper()
                    should_escalate = "ESCALATE" in decision_text and "AUTO" not in decision_text
                elif line.upper().startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()
                elif line.upper().startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                        confidence = max(0.0, min(1.0, confidence))
                    except ValueError:
                        confidence = 0.5
            
            return EscalationDecision(
                should_escalate=should_escalate,
                reason=reason,
                confidence=confidence,
                category="llm_decision",
            )
        
        except Exception as e:
            print(f"  LLM escalation error: {e}, falling back to rule-based")
            return rule_decision


if __name__ == "__main__":
    # Test examples
    engine = RuleBasedEscalation()
    
    test_cases = [
        ("My iPhone won't turn on", "device_not_working"),
        ("I'm going to sue Apple if this isn't fixed", "other"),
        ("I want to speak to your supervisor NOW", "escalation_request"),
        ("Thank you so much for the help!", "general_inquiry"),
        ("My phone caught fire while charging", "device_not_working"),
        ("I was charged $99 for something I never purchased", "billing_charge"),
        ("HOW DO I RESET MY PASSWORD THIS IS RIDICULOUS", "account_locked"),
    ]
    
    print("Escalation Engine Test:\n")
    for msg, intent in test_cases:
        decision = engine.decide(msg, intent)
        print(f"  Message: {msg}")
        print(f"  {decision}")
        print()
