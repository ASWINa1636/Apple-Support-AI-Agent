# Apple Support AI Agent 🍎

**Hiver SDE Intern — Take-Home Assignment**

An AI customer-support agent for Apple Support (@AppleSupport) built from real Twitter conversations. The agent classifies incoming customer messages into intents, drafts replies grounded in historical Apple responses, and decides whether to auto-handle or escalate to a human.

## Quick Start — Reproduce Results in <15 Minutes

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier works)

### Setup

```bash
# 1. Clone and enter the project
git clone <this-repo-url>
cd hiver-ai-support-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
# Linux/Mac:
export GROQ_API_KEY="your-key-here"
# Windows PowerShell:
$env:GROQ_API_KEY = "your-key-here"

# 4. Download and process the dataset (~2-3 minutes)
python data/download_data.py

# 5. Build the golden evaluation set (~3-5 minutes, uses Groq API)
python eval/build_golden_set.py

# 6. Run the full evaluation (~8-12 minutes)
python eval/evaluate.py
```

### Quick Demo

```bash
# Single message
python src/pipeline.py --message "My iPhone won't turn on after the update"

# Interactive mode
python src/pipeline.py --interactive

# Demo with sample messages
python src/pipeline.py
```

## Project Structure

```
hiver-ai-support-agent/
├── README.md                  # This file
├── REPORT.md                  # 6-page evaluation report
├── SUBMISSION_AND_INTERVIEW_GUIDE.md # Complete run manual, submission checklist & interview script
├── requirements.txt           # Python dependencies
├── .env.example               # API key template
│
├── data/
│   └── download_data.py       # Data pipeline: download, filter, process
│
├── src/
│   ├── pipeline.py            # End-to-end agent (CLI + importable)
│   ├── intent_classifier.py   # 3-tier intent classification
│   ├── reply_generator.py     # RAG-based reply generation
│   └── escalation_engine.py   # Auto-handle vs. escalate decision
│
├── eval/
│   ├── build_golden_set.py    # Build 200-example evaluation set
│   ├── evaluate.py            # Full evaluation harness
│   ├── llm_judge.py           # LLM-as-judge reply quality scorer
│   ├── labelling_notes.md     # Sampling & labelling methodology
│   └── golden_set.json        # Hand-labelled evaluation set (generated)
│
└── results/                   # Generated evaluation outputs
    ├── evaluation_results.json
    └── golden_set_annotated.json
```

## Architecture

```
Customer Message
    │
    ├──→ Intent Classifier ──→ One of 12 intents
    │     ├── Trivial: Most-frequent class
    │     ├── Simple: TF-IDF + Logistic Regression
    │     └── Main: LLM zero-shot (Groq llama-3.3-70b)
    │
    ├──→ Reply Generator ──→ Draft response
    │     ├── Trivial: Canned "Thank you" response
    │     ├── Simple: Nearest-neighbor (return most similar historical reply)
    │     └── Main: RAG (retrieve similar threads + LLM generation)
    │
    └──→ Escalation Engine ──→ Auto-handle / Escalate + reason
          ├── Baseline: Rule-based keyword/sentiment heuristics
          └── Main: LLM-based decision with confidence scoring
```

## Dataset

- **Source**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) via [HuggingFace mirror](https://huggingface.co/datasets/SunidhiSriram/twcs)
- **Brand**: Apple Support (@AppleSupport)
- **Subsample**: ~2,500 customer→response pairs (from ~3M total tweets)

## Evaluation

### Golden Set
- **200 hand-labelled examples** with stratified sampling
- Labels: intent, escalation decision, difficulty
- See [eval/labelling_notes.md](eval/labelling_notes.md) for methodology

### Metrics

| Component | Metric | How |
|-----------|--------|-----|
| Intent | Accuracy, Macro-F1 | vs. golden set labels |
| Reply | ROUGE-L, LLM Judge (4 dimensions) | vs. historical Apple responses |
| Escalation | Precision, Recall, F1 | vs. golden set labels |
| Judge | Cohen's κ | LLM judge vs. human agreement |

### LLM Judge Rubric
Replies scored 1-5 on:
1. **Relevance** — addresses the customer's issue?
2. **Helpfulness** — actionable steps provided?
3. **Tone** — empathetic, professional, Apple-like?
4. **Grounding** — aligned with Apple's historical responses?

## Key Decisions (see REPORT.md for full log)

1. Apple Support chosen for high volume + distinct brand voice
2. 12-intent taxonomy (not too coarse, not too fine)
3. RAG over pure generation to prevent hallucination
4. Groq + llama-3.3-70b for free-tier, fast inference
5. Sentence-transformers (local) for embeddings — no extra API needed

## Technology Stack

- **LLM**: Groq API with llama-3.3-70b-versatile
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, local)
- **ML**: scikit-learn (TF-IDF, Logistic Regression)
- **Evaluation**: rouge-score, sklearn metrics, custom LLM judge
- **Data**: HuggingFace datasets library

## License

This project is for educational/assessment purposes. The dataset is licensed under CC BY-NC-SA 4.0.
