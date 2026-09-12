"""
Data Pipeline: Download, filter, and process Apple Support conversations
from the Twitter Customer Support dataset (SunidhiSriram/twcs on HuggingFace).

This is the original twcs.csv format with tweet_id, author_id, inbound, etc.
We filter to AppleSupport brand conversations and reconstruct threads.
"""

import json
import os
import sys
import re
import random
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent
PROJECT_ROOT = DATA_DIR.parent


def download_dataset():
    """Download original format dataset from HuggingFace."""
    print("[1/5] Downloading dataset from HuggingFace (SunidhiSriram/twcs)...")
    
    from datasets import load_dataset
    ds = load_dataset("SunidhiSriram/twcs", split="train")
    df = ds.to_pandas()
    print(f"  Downloaded {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Sample author_ids: {df['author_id'].value_counts().head(10).to_dict()}")
    
    return df


def identify_apple_support(df):
    """Find the AppleSupport author_id."""
    print("\n[2/5] Identifying Apple Support account...")
    
    # Look for author_ids containing 'apple'
    all_authors = df["author_id"].unique()
    apple_candidates = [a for a in all_authors if "apple" in str(a).lower()]
    
    print(f"  Apple-related author_ids: {apple_candidates}")
    
    if apple_candidates:
        # Get tweet counts for each
        for author in apple_candidates:
            count = len(df[df["author_id"] == author])
            inbound_count = len(df[(df["author_id"] == author) & (df["inbound"] == False)])
            print(f"    {author}: {count} total, {inbound_count} outbound (responses)")
        
        # Use the primary Apple Support account
        apple_author = None
        for candidate in ["AppleSupport", "applesupport", "apple_support"]:
            if candidate in apple_candidates:
                apple_author = candidate
                break
        if not apple_author:
            apple_author = apple_candidates[0]
        
        print(f"  Using: {apple_author}")
        return {apple_author}
    
    # If no apple author found, show top outbound authors
    outbound = df[df["inbound"] == False]
    top_authors = outbound["author_id"].value_counts().head(20)
    print(f"  No Apple author found. Top 20 outbound authors:")
    print(f"  {top_authors.to_dict()}")
    
    # Search for Apple-related content in outbound messages
    for author in top_authors.head(30).index:
        author_texts = outbound[outbound["author_id"] == author]["text"].head(50)
        combined = " ".join(author_texts.astype(str)).lower()
        apple_score = sum(1 for kw in ["iphone", "ipad", "mac", "ios", "apple", "itunes", "icloud"]
                         if kw in combined)
        if apple_score >= 3:
            print(f"  Found Apple-related author by content: {author} (score: {apple_score})")
            return {author}
    
    raise ValueError("Could not find Apple Support author in dataset")


def filter_apple_conversations(df, apple_authors):
    """Filter to conversations involving Apple Support."""
    print("\n[3/5] Filtering Apple Support conversations...")
    
    apple_author = list(apple_authors)[0]
    
    # Get all Apple Support tweets (outbound responses)
    apple_responses = df[(df["author_id"] == apple_author) & (df["inbound"] == False)]
    print(f"  Apple Support responses: {len(apple_responses)}")
    
    # Get the customer tweets they responded to
    customer_tweet_ids = apple_responses["in_response_to_tweet_id"].dropna().astype(int).astype(str)
    df["tweet_id_str"] = df["tweet_id"].astype(str)
    
    customer_tweets = df[df["tweet_id_str"].isin(customer_tweet_ids)]
    print(f"  Matching customer tweets: {len(customer_tweets)}")
    
    # Build pairs: customer message → Apple response
    # Create a lookup from tweet_id to text
    tweet_text_map = dict(zip(df["tweet_id_str"], df["text"].astype(str)))
    
    pairs = []
    for _, apple_row in tqdm(apple_responses.iterrows(), 
                              total=len(apple_responses), 
                              desc="  Building pairs"):
        parent_id = apple_row.get("in_response_to_tweet_id")
        if pd.isna(parent_id):
            continue
        
        parent_id_str = str(int(parent_id))
        customer_text = tweet_text_map.get(parent_id_str, "")
        apple_text = str(apple_row["text"])
        
        if not customer_text or len(customer_text) < 10:
            continue
        if len(apple_text) < 10:
            continue
        
        pairs.append({
            "customer_message": customer_text,
            "apple_response": apple_text,
            "customer_tweet_id": parent_id_str,
            "apple_tweet_id": str(apple_row["tweet_id"]),
            "thread_id": f"thread_{parent_id_str}",
        })
    
    print(f"  Built {len(pairs)} customer->response pairs")
    return pairs


def clean_and_sample(pairs, max_pairs=2500):
    """Clean pairs and sample to manageable size."""
    print("\n[4/5] Cleaning and sampling...")
    
    cleaned = []
    for p in pairs:
        customer = p["customer_message"]
        apple = p["apple_response"]
        
        # Remove @mentions from beginning (but keep the message)
        customer_clean = re.sub(r'^(@\S+\s*)+', '', customer).strip()
        apple_clean = re.sub(r'^(@\S+\s*)+', '', apple).strip()
        
        # Skip if too short after cleaning
        if len(customer_clean) < 10 or len(apple_clean) < 10:
            continue
        
        p["customer_message_clean"] = customer_clean
        p["apple_response_clean"] = apple_clean
        cleaned.append(p)
    
    print(f"  After cleaning: {len(cleaned)} pairs")
    
    # Sample
    if len(cleaned) > max_pairs:
        random.seed(42)
        cleaned = random.sample(cleaned, max_pairs)
        print(f"  Sampled to {max_pairs} pairs")
    
    return cleaned


def save_data(pairs):
    """Save processed data."""
    print("\n[5/5] Saving processed data...")
    
    # Save pairs
    pairs_path = DATA_DIR / "apple_pairs.json"
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    
    # Save conversations format
    conversations = []
    for p in pairs:
        conversations.append({
            "thread_id": p["thread_id"],
            "messages": [
                {
                    "tweet_id": p["customer_tweet_id"],
                    "author_id": "customer",
                    "text": p["customer_message"],
                    "is_customer": True,
                    "is_apple": False,
                },
                {
                    "tweet_id": p["apple_tweet_id"],
                    "author_id": "AppleSupport",
                    "text": p["apple_response"],
                    "is_customer": False,
                    "is_apple": True,
                },
            ],
            "num_messages": 2,
        })
    
    conv_path = DATA_DIR / "apple_conversations.json"
    with open(conv_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved {len(pairs)} pairs to {pairs_path}")
    print(f"  Saved {len(conversations)} conversations to {conv_path}")
    
    # Stats
    avg_customer_len = sum(len(p["customer_message"].split()) for p in pairs) / max(len(pairs), 1)
    avg_apple_len = sum(len(p["apple_response"].split()) for p in pairs) / max(len(pairs), 1)
    
    print(f"\n  === Dataset Stats ===")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Avg customer message length: {avg_customer_len:.1f} words")
    print(f"  Avg Apple response length: {avg_apple_len:.1f} words")
    
    # Show some examples
    print(f"\n  === Sample Pairs ===")
    for p in pairs[:3]:
        customer = p.get("customer_message_clean", p["customer_message"])
        apple = p.get("apple_response_clean", p["apple_response"])
        print(f"  Customer: {customer[:100]}...")
        print(f"  Apple:    {apple[:100]}...")
        print()
    
    return pairs


def main():
    """Run the full data pipeline."""
    print("=" * 60)
    print("Apple Support Data Pipeline")
    print("=" * 60)
    
    # Step 1: Download
    df = download_dataset()
    
    # Step 2: Identify Apple Support
    apple_authors = identify_apple_support(df)
    
    # Step 3: Filter conversations
    pairs = filter_apple_conversations(df, apple_authors)
    
    # Step 4: Clean and sample
    pairs = clean_and_sample(pairs)
    
    # Step 5: Save
    save_data(pairs)
    
    print("\n✅ Data pipeline complete!")
    return pairs


if __name__ == "__main__":
    main()
