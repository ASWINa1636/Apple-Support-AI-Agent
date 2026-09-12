"""
Pipeline: End-to-end AI Support Agent for Apple Support.

Takes a raw customer message and returns:
- Intent classification
- Drafted reply
- Escalation decision with reason

Can be used in CLI mode or imported as a module.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment early
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from src.intent_classifier import LLMClassifier, TfidfClassifier, TrivialClassifier, clean_text
from src.reply_generator import (
    RAGReplyGenerator, NearestNeighborReplyGenerator, TrivialReplyGenerator,
    EmbeddingStore,
)
from src.escalation_engine import LLMEscalation, RuleBasedEscalation

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


class AppleSupportAgent:
    """End-to-end AI Support Agent for Apple Support."""
    
    def __init__(self, mode: str = "main", api_key: str = ""):
        """
        Initialize the agent.
        
        Args:
            mode: "main" (LLM-based), "simple" (TF-IDF + NN), or "trivial" (baselines)
            api_key: Groq API key (or uses GROQ_API_KEY env var)
        """
        self.mode = mode
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        
        print(f"\n{'='*60}")
        print(f"Initializing Apple Support Agent (mode: {mode})")
        print(f"{'='*60}")
        
        # Initialize components based on mode
        self._init_intent_classifier(mode)
        self._init_reply_generator(mode)
        self._init_escalation_engine(mode)
        
        print(f"\n✅ Agent ready! (mode: {mode})")
    
    def _init_intent_classifier(self, mode: str):
        """Initialize the appropriate intent classifier."""
        if mode == "main":
            print("\n[Intent] Loading LLM classifier...")
            self.intent_classifier = LLMClassifier(api_key=self.api_key)
        elif mode == "simple":
            print("\n[Intent] Loading TF-IDF classifier...")
            self.intent_classifier = TfidfClassifier()
            # Try to load pre-trained model
            model_path = DATA_DIR / "tfidf_model.pkl"
            if model_path.exists():
                self.intent_classifier.load(str(model_path))
                print("  Loaded pre-trained model")
            else:
                print("  WARNING: No pre-trained model found. Run evaluation first to train.")
        else:
            print("\n[Intent] Loading trivial classifier...")
            self.intent_classifier = TrivialClassifier()
            self.intent_classifier.most_frequent = "device_not_working"
    
    def _init_reply_generator(self, mode: str):
        """Initialize the appropriate reply generator."""
        if mode in ["main", "simple"]:
            pairs_path = DATA_DIR / "apple_pairs.json"
            embeddings_path = DATA_DIR / "embeddings.npz"
            
            if pairs_path.exists():
                print(f"\n[Reply] Loading embedding store...")
                self.embedding_store = EmbeddingStore(str(pairs_path))
                self.embedding_store.build_embeddings(str(embeddings_path))
                
                if mode == "main":
                    self.reply_generator = RAGReplyGenerator(
                        self.embedding_store, api_key=self.api_key
                    )
                else:
                    self.reply_generator = NearestNeighborReplyGenerator(
                        self.embedding_store
                    )
            else:
                print(f"\n[Reply] WARNING: No pairs data at {pairs_path}")
                print("  Run data pipeline first: python data/download_data.py")
                self.embedding_store = None
                self.reply_generator = TrivialReplyGenerator()
        else:
            print(f"\n[Reply] Loading trivial reply generator...")
            self.embedding_store = None
            self.reply_generator = TrivialReplyGenerator()
    
    def _init_escalation_engine(self, mode: str):
        """Initialize the appropriate escalation engine."""
        if mode == "main":
            print("\n[Escalation] Loading LLM escalation engine...")
            self.escalation_engine = LLMEscalation(api_key=self.api_key)
        else:
            print("\n[Escalation] Loading rule-based escalation engine...")
            self.escalation_engine = RuleBasedEscalation()
    
    def process(self, customer_message: str) -> dict:
        """
        Process a single customer message through the full pipeline.
        
        Returns:
            dict with keys: intent, reply, escalation (with should_escalate, reason, confidence)
        """
        # Step 1: Classify intent
        intent = self.intent_classifier.predict_one(customer_message)
        
        # Step 2: Make escalation decision
        escalation = self.escalation_engine.decide(customer_message, intent)
        
        # Step 3: Generate reply
        if hasattr(self.reply_generator, 'generate'):
            reply = self.reply_generator.generate(customer_message, intent)
        else:
            reply = self.reply_generator.generate(customer_message)
        
        return {
            "customer_message": customer_message,
            "intent": intent,
            "reply": reply,
            "escalation": escalation.to_dict(),
        }
    
    def process_batch(self, messages: list, delay: float = 0.15) -> list:
        """Process a batch of customer messages."""
        results = []
        for i, msg in enumerate(messages):
            text = msg if isinstance(msg, str) else msg.get("customer_message", msg.get("text", ""))
            result = self.process(text)
            results.append(result)
            if delay > 0 and i < len(messages) - 1:
                time.sleep(delay)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(messages)}...")
        return results


def main():
    """CLI interface for the Apple Support Agent."""
    parser = argparse.ArgumentParser(description="Apple Support AI Agent")
    parser.add_argument("--message", "-m", type=str, help="Customer message to process")
    parser.add_argument("--mode", choices=["main", "simple", "trivial"], 
                       default="main", help="Agent mode")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive mode")
    parser.add_argument("--batch", "-b", type=str,
                       help="Path to JSON file with messages to process")
    
    args = parser.parse_args()
    
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    
    agent = AppleSupportAgent(mode=args.mode)
    
    if args.message:
        # Single message mode
        result = agent.process(args.message)
        print(f"\n{'='*60}")
        print(f"Customer: {result['customer_message']}")
        print(f"Intent:   {result['intent']}")
        print(f"Reply:    {result['reply']}")
        esc = result['escalation']
        action = "🚨 ESCALATE" if esc['should_escalate'] else "✅ AUTO-HANDLE"
        print(f"Action:   {action}")
        print(f"Reason:   {esc['reason']}")
        print(f"Confidence: {esc['confidence']:.2f}")
        print(f"{'='*60}")
    
    elif args.interactive:
        # Interactive mode
        print("\n🍎 Apple Support Agent (Interactive Mode)")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                message = input("Customer> ").strip()
                if message.lower() in ["quit", "exit", "q"]:
                    break
                if not message:
                    continue
                
                result = agent.process(message)
                print(f"\n  Intent: {result['intent']}")
                print(f"  Reply: {result['reply']}")
                esc = result['escalation']
                action = "🚨 ESCALATE" if esc['should_escalate'] else "✅ AUTO-HANDLE"
                print(f"  Action: {action} — {esc['reason']}")
                print()
            except KeyboardInterrupt:
                break
        print("\nGoodbye! 👋")
    
    elif args.batch:
        # Batch mode
        with open(args.batch, "r", encoding="utf-8") as f:
            messages = json.load(f)
        
        results = agent.process_batch(messages)
        
        # Save results
        RESULTS_DIR.mkdir(exist_ok=True)
        output_path = RESULTS_DIR / "batch_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")
    
    else:
        # Demo mode
        demo_messages = [
            "My iPhone won't turn on after the latest iOS update",
            "I was charged $14.99 for an app I never purchased! I want a refund",
            "How do I transfer my photos to my new iPhone?",
            "My MacBook battery only lasts 2 hours now. It used to last 8!",
            "I WANT TO SPEAK TO A MANAGER. THIS IS THE WORST SERVICE EVER!!!",
        ]
        
        print("\n🍎 Apple Support Agent — Demo\n")
        for msg in demo_messages:
            result = agent.process(msg)
            print(f"Customer: {msg}")
            print(f"  -> Intent: {result['intent']}")
            print(f"  -> Reply: {result['reply'][:150]}...")
            esc = result['escalation']
            action = "🚨 ESCALATE" if esc['should_escalate'] else "✅ AUTO-HANDLE"
            print(f"  -> {action}: {esc['reason']}")
            print()


if __name__ == "__main__":
    main()
