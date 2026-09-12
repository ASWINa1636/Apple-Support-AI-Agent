"""
Intent Classifier: Classify customer messages into support intent categories.

Provides three classifiers:
1. Trivial Baseline: Most-frequent-class
2. Simple Baseline: TF-IDF + Logistic Regression
3. Main System: LLM zero-shot classification via Groq
"""

import json
import os
import re
import random
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Intent taxonomy for Apple Support
INTENT_TAXONOMY = {
    "device_not_working": "Device won't turn on, frozen, unresponsive, or crashing",
    "app_issue": "App crashing, not downloading, not updating, or malfunctioning",
    "account_locked": "Apple ID locked, password reset, two-factor authentication issues",
    "billing_charge": "Unexpected charges, subscription issues, refund requests",
    "battery_drain": "Battery draining fast, not charging, battery health concerns",
    "connectivity": "WiFi, Bluetooth, cellular, or AirDrop connectivity problems",
    "update_problem": "iOS/macOS update failed, stuck, or causing issues after update",
    "data_loss": "Lost photos, contacts, messages, or backup/restore issues",
    "repair_warranty": "Repair status, warranty questions, service center inquiries",
    "general_inquiry": "General questions, how-to, feedback, or praise",
    "escalation_request": "Customer explicitly asking for supervisor or formal complaint",
    "other": "Doesn't fit any of the above categories",
}

INTENT_LIST = list(INTENT_TAXONOMY.keys())


def clean_text(text: str) -> str:
    """Clean tweet text for classification."""
    if not isinstance(text, str):
        return ""
    # Remove @mentions (anonymized user IDs)
    text = re.sub(r'@\S+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove __email__ and __phone__ masks
    text = re.sub(r'__\w+__', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class TrivialClassifier:
    """Baseline 1: Always predicts the most frequent class."""
    
    def __init__(self):
        self.most_frequent = None
    
    def fit(self, texts, labels):
        """Find the most frequent label."""
        from collections import Counter
        counts = Counter(labels)
        self.most_frequent = counts.most_common(1)[0][0]
        print(f"  TrivialClassifier: most frequent class = '{self.most_frequent}' "
              f"({counts[self.most_frequent]}/{len(labels)} = "
              f"{counts[self.most_frequent]/len(labels)*100:.1f}%)")
    
    def predict(self, texts):
        """Predict most frequent class for all inputs."""
        return [self.most_frequent] * len(texts)
    
    def predict_one(self, text):
        return self.most_frequent


class TfidfClassifier:
    """Baseline 2: TF-IDF + Logistic Regression."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
        )
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=42,
        )
        self.label_encoder = LabelEncoder()
    
    def fit(self, texts, labels):
        """Train TF-IDF + LogReg classifier."""
        cleaned = [clean_text(t) for t in texts]
        X = self.vectorizer.fit_transform(cleaned)
        y = self.label_encoder.fit_transform(labels)
        self.model.fit(X, y)
        print(f"  TfidfClassifier: trained on {len(texts)} examples, "
              f"{len(self.label_encoder.classes_)} classes")
    
    def predict(self, texts):
        """Predict intents for a batch of texts."""
        cleaned = [clean_text(t) for t in texts]
        X = self.vectorizer.transform(cleaned)
        y_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred).tolist()
    
    def predict_one(self, text):
        return self.predict([text])[0]
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "model": self.model, 
                         "label_encoder": self.label_encoder}, f)
    
    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.model = data["model"]
            self.label_encoder = data["label_encoder"]


class LLMClassifier:
    """Main System: LLM zero-shot classification via Groq API."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from groq import Groq
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set. Please set it as an environment variable.")
        self.client = Groq(api_key=self.api_key)
        self.model = model or os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
    
    def _build_prompt(self, text: str) -> str:
        """Build classification prompt."""
        intent_descriptions = "\n".join(
            f"- {name}: {desc}" for name, desc in INTENT_TAXONOMY.items()
        )
        return f"""You are an intent classifier for Apple customer support messages on Twitter.

Classify the following customer message into exactly ONE of these intent categories:

{intent_descriptions}

Customer message: "{text}"

Respond with ONLY the intent name (one of: {', '.join(INTENT_LIST)}). No explanation, no punctuation, just the intent name."""
    
    def predict_one(self, text: str) -> str:
        """Classify a single message."""
        cleaned = clean_text(text)
        if not cleaned:
            return "other"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an intent classifier. Respond with only the intent category name."},
                    {"role": "user", "content": self._build_prompt(cleaned)},
                ],
                temperature=0,
                max_tokens=50,
            )
            result = response.choices[0].message.content.strip().lower()
            # Extract the intent name
            for intent in INTENT_LIST:
                if intent in result:
                    return intent
            return "other"
        except Exception as e:
            print(f"  LLM classification error: {e}")
            return "other"
    
    def predict(self, texts: list, batch_delay: float = 0.1) -> list:
        """Classify a batch of messages with rate limiting."""
        import time
        results = []
        for i, text in enumerate(texts):
            result = self.predict_one(text)
            results.append(result)
            if batch_delay > 0 and i < len(texts) - 1:
                time.sleep(batch_delay)
            if (i + 1) % 25 == 0:
                print(f"  Classified {i+1}/{len(texts)}...")
        return results


def discover_intents_from_data(pairs_path: str, sample_size: int = 200):
    """
    Use a sample of customer messages to validate/refine our intent taxonomy.
    This is an exploratory step - the taxonomy above was designed from domain knowledge
    and validated against the data.
    """
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    # Sample customer messages
    sample = random.sample(pairs, min(sample_size, len(pairs)))
    messages = [clean_text(p["customer_message"]) for p in sample]
    messages = [m for m in messages if len(m) > 10]  # Filter very short
    
    print(f"\nIntent Discovery: Sampled {len(messages)} customer messages")
    print(f"Example messages:")
    for msg in messages[:10]:
        print(f"  - {msg[:100]}")
    
    return messages


if __name__ == "__main__":
    # Test with a few examples
    examples = [
        "My iPhone won't turn on after the latest update",
        "I was charged $9.99 for something I didn't buy",
        "How do I reset my Apple ID password?",
        "The battery on my MacBook drains in 2 hours",
        "WiFi keeps disconnecting on my iPad",
        "I want to speak to a manager about this issue",
    ]
    
    print("Testing intent classification on examples:\n")
    for ex in examples:
        print(f"  Message: {ex}")
        # Would test with LLM classifier here if API key is set
