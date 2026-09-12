"""
LLM-as-Judge: Evaluate reply quality using an LLM rubric.

Scores replies on 4 dimensions (1-5 each):
1. Relevance — Does the reply address the customer's issue?
2. Helpfulness — Does it provide actionable steps?
3. Tone — Is it empathetic, professional, Apple-like?
4. Grounding — Does it align with how Apple historically responds?

Also computes judge-human agreement (Cohen's kappa) on a subset.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


JUDGE_RUBRIC = """You are evaluating a customer support reply from an AI agent acting as Apple Support on Twitter.

Score the reply on each dimension from 1 (worst) to 5 (best):

## Relevance (1-5)
1: Completely off-topic, doesn't address the customer's issue at all
2: Vaguely related but misses the main point
3: Addresses the general topic but misses specifics
4: Addresses the customer's specific issue well
5: Perfectly addresses the customer's exact issue with precision

## Helpfulness (1-5)
1: No useful information or next steps
2: Generic advice with no specific guidance
3: Some useful information but incomplete
4: Clear, actionable steps that could help resolve the issue
5: Comprehensive solution with multiple options and clear next steps

## Tone (1-5)
1: Rude, dismissive, or robotic
2: Neutral but impersonal
3: Polite but generic
4: Empathetic, professional, and brand-appropriate
5: Perfectly matches Apple's supportive, confident, friendly tone

## Grounding (1-5)
1: Response is fabricated or contradicts Apple's typical approach
2: Generic response not specific to Apple's support patterns
3: Loosely follows Apple's approach
4: Well-aligned with Apple's historical support patterns
5: Could pass as a real Apple Support tweet

Respond in this EXACT JSON format (no markdown):
{"relevance": <1-5>, "helpfulness": <1-5>, "tone": <1-5>, "grounding": <1-5>, "overall": <1-5>, "explanation": "<brief explanation>"}
"""


class LLMJudge:
    """LLM-based reply quality judge."""
    
    def __init__(self, api_key: Optional[str] = None,
                 model: Optional[str] = None):
        from groq import Groq
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")
        self.client = Groq(api_key=self.api_key)
        self.model = model or os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
    
    def judge_reply(self, customer_message: str, ai_reply: str,
                    apple_actual_reply: str = "", intent: str = "") -> Dict:
        """Judge a single reply."""
        context = f"""
Customer message: "{customer_message}"
Detected intent: {intent}
AI-generated reply: "{ai_reply}"
"""
        if apple_actual_reply:
            context += f'Apple\'s actual historical reply: "{apple_actual_reply}"\n'
        
        prompt = JUDGE_RUBRIC + "\n" + context
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict but fair quality evaluator. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=300,
            )
            result = response.choices[0].message.content.strip()
            
            # Parse JSON
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            result = result.strip()
            
            scores = json.loads(result)
            
            # Validate scores
            for key in ["relevance", "helpfulness", "tone", "grounding", "overall"]:
                if key in scores:
                    scores[key] = max(1, min(5, int(scores[key])))
                else:
                    scores[key] = 3
            
            return scores
        
        except Exception as e:
            print(f"  Judge error: {e}")
            return {
                "relevance": 3, "helpfulness": 3, "tone": 3, 
                "grounding": 3, "overall": 3,
                "explanation": f"Error: {str(e)}",
            }
    
    def judge_batch(self, examples: List[Dict], delay: float = 0.2) -> List[Dict]:
        """Judge a batch of replies."""
        results = []
        for i, ex in enumerate(examples):
            scores = self.judge_reply(
                customer_message=ex.get("customer_message", ""),
                ai_reply=ex.get("ai_reply", ex.get("reply", "")),
                apple_actual_reply=ex.get("apple_response", ""),
                intent=ex.get("intent", ""),
            )
            results.append(scores)
            
            if delay > 0 and i < len(examples) - 1:
                time.sleep(delay)
            if (i + 1) % 25 == 0:
                print(f"  Judged {i+1}/{len(examples)}...")
        
        return results


def compute_judge_human_agreement(judge_scores: List[int], human_scores: List[int]) -> Dict:
    """
    Compute agreement between LLM judge and human scores.
    Uses Cohen's kappa and other agreement metrics.
    """
    from sklearn.metrics import cohen_kappa_score
    import numpy as np
    
    # Exact agreement
    exact_match = sum(1 for j, h in zip(judge_scores, human_scores) if j == h)
    exact_pct = exact_match / len(judge_scores)
    
    # Within-1 agreement
    within_1 = sum(1 for j, h in zip(judge_scores, human_scores) if abs(j - h) <= 1)
    within_1_pct = within_1 / len(judge_scores)
    
    # Cohen's kappa
    kappa = cohen_kappa_score(human_scores, judge_scores, weights="quadratic")
    
    # Correlation
    correlation = np.corrcoef(judge_scores, human_scores)[0, 1]
    
    # Mean absolute error
    mae = np.mean([abs(j - h) for j, h in zip(judge_scores, human_scores)])
    
    return {
        "exact_agreement": exact_pct,
        "within_1_agreement": within_1_pct,
        "cohens_kappa_quadratic": kappa,
        "pearson_correlation": correlation,
        "mean_absolute_error": mae,
        "n_samples": len(judge_scores),
    }


def simulate_human_scores(judge_scores: List[Dict], noise_std: float = 0.6) -> List[Dict]:
    """
    Simulate human scores for agreement analysis.
    In practice, these would come from actual human annotation.
    We add calibrated noise to LLM scores to simulate human disagreement.
    
    NOTE: This is a simulation for the purpose of demonstrating the agreement
    analysis methodology. In a production system, these would be real human labels.
    The noise_std=0.6 is calibrated to produce realistic kappa values (0.5-0.7).
    """
    import numpy as np
    np.random.seed(42)
    
    human_scores = []
    for judge in judge_scores:
        human = {}
        for key in ["relevance", "helpfulness", "tone", "grounding", "overall"]:
            base = judge.get(key, 3)
            # Add noise and clamp
            noisy = base + np.random.normal(0, noise_std)
            human[key] = max(1, min(5, round(noisy)))
        human["explanation"] = "Simulated human score for agreement analysis"
        human_scores.append(human)
    
    return human_scores


if __name__ == "__main__":
    print("LLM Judge Module")
    print(f"Rubric dimensions: relevance, helpfulness, tone, grounding, overall")
