"""
Reply Generator: Draft replies grounded in Apple Support's historical responses.

Provides three approaches:
1. Trivial Baseline: Canned response
2. Simple Baseline: Nearest-neighbor retrieval (most similar historical reply)
3. Main System: RAG — retrieve similar threads + LLM generation
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np

# Load environment variables early if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Suppress unauthenticated requests warning and noisy progress bars from huggingface_hub by default
if "HF_HUB_VERBOSITY" not in os.environ:
    os.environ["HF_HUB_VERBOSITY"] = "error"
if "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

DATA_DIR = Path(__file__).parent.parent / "data"

# Canned response for trivial baseline
CANNED_RESPONSE = (
    "Thank you for reaching out to Apple Support! We're sorry to hear you're "
    "experiencing this issue. Could you please DM us your device model, iOS version, "
    "and a brief description of the problem so we can help you further? "
    "We're here to help! 🍎"
)


def clean_text_for_embedding(text: str) -> str:
    """Clean text for embedding similarity search."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'__\w+__', '[REDACTED]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class EmbeddingStore:
    """Manages embeddings for historical customer-response pairs."""
    
    def __init__(self, pairs_path: Optional[str] = None):
        self.pairs = []
        self.embeddings = None
        self.model = None
        
        if pairs_path:
            self.load_pairs(pairs_path)
    
    def load_pairs(self, pairs_path: str):
        """Load customer-response pairs."""
        with open(pairs_path, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)
        print(f"  Loaded {len(self.pairs)} customer->response pairs")
    
    def build_embeddings(self, cache_path: Optional[str] = None):
        """Build sentence embeddings for all customer messages."""
        if cache_path and os.path.exists(cache_path):
            print(f"  Loading cached embeddings from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            self.embeddings = data["embeddings"]
            return
        
        print("  Building embeddings with sentence-transformers...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        texts = [clean_text_for_embedding(p["customer_message"]) for p in self.pairs]
        self.embeddings = self.model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=128,
            normalize_embeddings=True,
        )
        
        if cache_path:
            np.savez_compressed(cache_path, embeddings=self.embeddings)
            print(f"  Cached embeddings to {cache_path}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find most similar historical conversations."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        query_clean = clean_text_for_embedding(query)
        query_emb = self.model.encode([query_clean], normalize_embeddings=True)
        
        # Cosine similarity (embeddings are normalized)
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "customer_message": self.pairs[idx]["customer_message"],
                "apple_response": self.pairs[idx]["apple_response"],
                "similarity": float(similarities[idx]),
            })
        return results


class TrivialReplyGenerator:
    """Baseline 1: Always returns a canned response."""
    
    def generate(self, customer_message: str, intent: str = "") -> str:
        return CANNED_RESPONSE


class NearestNeighborReplyGenerator:
    """Baseline 2: Returns the Apple response from the most similar historical thread."""
    
    def __init__(self, embedding_store: EmbeddingStore):
        self.store = embedding_store
    
    def generate(self, customer_message: str, intent: str = "") -> str:
        results = self.store.search(customer_message, top_k=1)
        if results:
            return results[0]["apple_response"]
        return CANNED_RESPONSE


class RAGReplyGenerator:
    """Main System: RAG-based reply generation using Groq LLM."""
    
    def __init__(self, embedding_store: EmbeddingStore, 
                 api_key: Optional[str] = None,
                 model: Optional[str] = None):
        from groq import Groq
        self.store = embedding_store
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")
        self.client = Groq(api_key=self.api_key)
        self.model = model or os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
    
    def _build_prompt(self, customer_message: str, intent: str, 
                      similar_threads: List[Dict]) -> str:
        """Build RAG prompt with retrieved context."""
        context = "\n\n".join([
            f"--- Historical Example {i+1} (similarity: {t['similarity']:.2f}) ---\n"
            f"Customer: {t['customer_message']}\n"
            f"Apple Support: {t['apple_response']}"
            for i, t in enumerate(similar_threads[:3])
        ])
        
        return f"""You are an Apple Support agent responding to a customer on Twitter.

The customer's issue has been classified as: {intent}

Here are similar historical conversations that Apple Support has handled:

{context}

---

Now draft a reply to this new customer message. Your reply should:
1. Be empathetic and professional, matching Apple's brand voice
2. Address the specific issue raised
3. Provide actionable next steps when possible
4. Be concise (Twitter-length, under 280 characters if possible, max 560)
5. Use Apple's typical support patterns (e.g., suggesting DMs for personal info, linking to support pages)
6. Do NOT make up specific troubleshooting steps that weren't in the historical examples
7. If the issue needs more info, ask for it politely

Customer message: "{customer_message}"

Draft your reply:"""
    
    def generate(self, customer_message: str, intent: str = "general_inquiry") -> str:
        """Generate a reply using RAG."""
        # Retrieve similar threads (top 3 provides high relevance without payload limits)
        similar = self.store.search(customer_message, top_k=3)
        
        prompt = self._build_prompt(customer_message, intent, similar)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "You are an Apple Support agent on Twitter. Be helpful, empathetic, "
                        "and concise. Mirror Apple's professional yet friendly tone."
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            reply = response.choices[0].message.content.strip()
            # Remove any quotation marks wrapping the reply
            reply = reply.strip('"\'')
            return reply
        except Exception as e:
            print(f"  Reply generation error: {e}")
            return CANNED_RESPONSE
    
    def generate_batch(self, messages: List[Dict], delay: float = 0.15) -> List[str]:
        """Generate replies for a batch of messages."""
        replies = []
        for i, msg in enumerate(messages):
            reply = self.generate(
                msg.get("customer_message", msg.get("text", "")),
                msg.get("intent", "general_inquiry"),
            )
            replies.append(reply)
            if delay > 0 and i < len(messages) - 1:
                time.sleep(delay)
            if (i + 1) % 25 == 0:
                print(f"  Generated {i+1}/{len(messages)} replies...")
        return replies


if __name__ == "__main__":
    print("Reply Generator Module")
    print(f"Canned response: {CANNED_RESPONSE}")
