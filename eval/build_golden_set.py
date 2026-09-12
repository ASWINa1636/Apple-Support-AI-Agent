"""
Golden Evaluation Set Builder: Create a hand-labelled evaluation set
of 200 examples from the Apple Support dataset.

Sampling strategy:
1. Load processed conversation pairs
2. Sample stratified across message types (different lengths, topics)
3. Use LLM for initial labelling, then provide for manual review
4. Output labelled JSON with intent, escalation, and quality annotations
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = Path(__file__).parent

from src.intent_classifier import INTENT_TAXONOMY, INTENT_LIST, clean_text


def load_pairs():
    """Load customer-response pairs."""
    pairs_path = DATA_DIR / "apple_pairs.json"
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    return pairs


def stratified_sample(pairs, target_size=200):
    """
    Sample examples ensuring diversity:
    - Different message lengths (short, medium, long)
    - Different apparent topics
    - Include edge cases
    """
    random.seed(42)  # Reproducibility
    
    # Clean and filter
    valid_pairs = []
    for p in pairs:
        cleaned = clean_text(p["customer_message"])
        if len(cleaned) > 10:  # Skip too-short messages
            p["cleaned_text"] = cleaned
            p["word_count"] = len(cleaned.split())
            valid_pairs.append(p)
    
    print(f"  Valid pairs after filtering: {len(valid_pairs)}")
    
    # Stratify by message length
    short = [p for p in valid_pairs if p["word_count"] <= 10]
    medium = [p for p in valid_pairs if 10 < p["word_count"] <= 25]
    long_msgs = [p for p in valid_pairs if p["word_count"] > 25]
    
    print(f"  Short (≤10 words): {len(short)}")
    print(f"  Medium (11-25 words): {len(medium)}")
    print(f"  Long (>25 words): {len(long_msgs)}")
    
    # Sample: 30% short, 50% medium, 20% long
    n_short = min(len(short), int(target_size * 0.3))
    n_medium = min(len(medium), int(target_size * 0.5))
    n_long = min(len(long_msgs), int(target_size * 0.2))
    
    # Adjust if we don't have enough
    remaining = target_size - n_short - n_medium - n_long
    if remaining > 0:
        n_medium = min(len(medium), n_medium + remaining)
    
    sampled = (
        random.sample(short, n_short) +
        random.sample(medium, n_medium) +
        random.sample(long_msgs, n_long)
    )
    
    random.shuffle(sampled)
    return sampled[:target_size]


def label_with_llm(samples, api_key=None):
    """Use LLM to generate initial labels (to be manually reviewed)."""
    from groq import Groq
    
    api_key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("  WARNING: No GROQ_API_KEY set. Using heuristic labelling.")
        return label_with_heuristics(samples)
    
    client = Groq(api_key=api_key)
    
    intent_desc = "\n".join(f"- {k}: {v}" for k, v in INTENT_TAXONOMY.items())
    
    labelled = []
    for i, sample in enumerate(samples):
        text = sample["cleaned_text"]
        
        prompt = f"""Analyze this Apple Support customer message and provide labels.

Customer message: "{text}"
Apple's actual response: "{sample.get('apple_response', 'N/A')}"

Available intents:
{intent_desc}

Respond in this EXACT JSON format (no markdown, just raw JSON):
{{
  "intent": "<one of the intent names above>",
  "should_escalate": <true or false>,
  "escalation_reason": "<why or why not escalate>",
  "difficulty": "<easy, medium, or hard>",
  "notes": "<any edge case notes>"
}}"""
        
        try:
            response = client.chat.completions.create(
                model="groq/compound",
                messages=[
                    {"role": "system", "content": "You are a data labelling assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=300,
            )
            result = response.choices[0].message.content.strip()
            
            # Parse JSON - handle potential markdown wrapping
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            result = result.strip()
            
            labels = json.loads(result)
            
            # Validate intent
            if labels.get("intent") not in INTENT_LIST:
                labels["intent"] = "other"
            
        except Exception as e:
            print(f"  Error labelling example {i}: {e}")
            labels = label_single_heuristic(sample)
        
        labelled_example = {
            "id": i,
            "customer_message": sample["customer_message"],
            "apple_response": sample.get("apple_response", ""),
            "thread_id": sample.get("thread_id", ""),
            "intent": labels.get("intent", "other"),
            "should_escalate": labels.get("should_escalate", False),
            "escalation_reason": labels.get("escalation_reason", ""),
            "difficulty": labels.get("difficulty", "medium"),
            "notes": labels.get("notes", ""),
            "labeller": "llm_initial",  # Will be updated to "human_reviewed" after manual check
        }
        labelled.append(labelled_example)
        
        if (i + 1) % 25 == 0:
            print(f"  Labelled {i+1}/{len(samples)}...")
        
        # Rate limiting
        time.sleep(0.2)
    
    return labelled


def label_single_heuristic(sample):
    """Heuristic labelling for a single sample."""
    text = sample.get("cleaned_text", clean_text(sample["customer_message"])).lower()
    
    # Simple keyword matching for intents
    if any(w in text for w in ["won't turn on", "frozen", "restart", "crash", "black screen", "stuck"]):
        intent = "device_not_working"
    elif any(w in text for w in ["app", "download", "install", "update app"]):
        intent = "app_issue"
    elif any(w in text for w in ["apple id", "password", "locked out", "sign in", "login"]):
        intent = "account_locked"
    elif any(w in text for w in ["charge", "bill", "refund", "subscription", "payment"]):
        intent = "billing_charge"
    elif any(w in text for w in ["battery", "charging", "drain"]):
        intent = "battery_drain"
    elif any(w in text for w in ["wifi", "bluetooth", "connection", "cellular", "signal"]):
        intent = "connectivity"
    elif any(w in text for w in ["update", "ios", "upgrade", "software"]):
        intent = "update_problem"
    elif any(w in text for w in ["photo", "backup", "restore", "lost", "data"]):
        intent = "data_loss"
    elif any(w in text for w in ["repair", "warranty", "genius", "service"]):
        intent = "repair_warranty"
    elif any(w in text for w in ["manager", "supervisor", "escalat", "complaint"]):
        intent = "escalation_request"
    elif any(w in text for w in ["how do", "how to", "what is", "thank", "help"]):
        intent = "general_inquiry"
    else:
        intent = "other"
    
    # Escalation heuristic
    should_escalate = any(w in text for w in [
        "lawyer", "sue", "manager", "supervisor", "unacceptable", 
        "fire", "smoke", "refund", "fraud", "scam"
    ])
    
    return {
        "intent": intent,
        "should_escalate": should_escalate,
        "escalation_reason": "Heuristic detection" if should_escalate else "Standard query",
        "difficulty": "medium",
        "notes": "Labelled by heuristic - needs manual review",
    }


def label_with_heuristics(samples):
    """Fallback: label all samples with heuristics."""
    labelled = []
    for i, sample in enumerate(samples):
        labels = label_single_heuristic(sample)
        labelled_example = {
            "id": i,
            "customer_message": sample["customer_message"],
            "apple_response": sample.get("apple_response", ""),
            "thread_id": sample.get("thread_id", ""),
            "intent": labels["intent"],
            "should_escalate": labels["should_escalate"],
            "escalation_reason": labels["escalation_reason"],
            "difficulty": labels["difficulty"],
            "notes": labels["notes"],
            "labeller": "heuristic",
        }
        labelled.append(labelled_example)
    return labelled


def build_golden_set(target_size=200):
    """Build the complete golden evaluation set."""
    print("=" * 60)
    print("Building Golden Evaluation Set")
    print("=" * 60)
    
    # Load pairs
    print("\n[1/3] Loading data...")
    pairs = load_pairs()
    print(f"  Total pairs: {len(pairs)}")
    
    # Sample
    print("\n[2/3] Stratified sampling...")
    samples = stratified_sample(pairs, target_size)
    print(f"  Sampled {len(samples)} examples")
    
    # Label
    print("\n[3/3] Labelling with LLM...")
    labelled = label_with_llm(samples)
    
    # Save
    output_path = EVAL_DIR / "golden_set.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labelled, f, indent=2, ensure_ascii=False)
    
    # Stats
    intent_counts = Counter(ex["intent"] for ex in labelled)
    escalation_count = sum(1 for ex in labelled if ex["should_escalate"])
    difficulty_counts = Counter(ex["difficulty"] for ex in labelled)
    
    print(f"\n  === Golden Set Stats ===")
    print(f"  Total examples: {len(labelled)}")
    print(f"  Intent distribution:")
    for intent, count in intent_counts.most_common():
        print(f"    {intent}: {count} ({count/len(labelled)*100:.1f}%)")
    print(f"  Escalation rate: {escalation_count}/{len(labelled)} "
          f"({escalation_count/len(labelled)*100:.1f}%)")
    print(f"  Difficulty: {dict(difficulty_counts)}")
    print(f"\n  Saved to {output_path}")
    
    return labelled


if __name__ == "__main__":
    build_golden_set(200)
