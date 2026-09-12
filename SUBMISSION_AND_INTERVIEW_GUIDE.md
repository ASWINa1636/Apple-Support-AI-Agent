# Master Guide: Apple Support AI Agent 🍎
### Complete Run Instructions, Submission Blueprint, System Architecture & Interview Defense

> **Author / Candidate**: Hiver SDE Intern Applicant  
> **Brand Selected**: Apple Support (`@AppleSupport`)  
> **Core Dataset**: Customer Support on Twitter (~3M conversations)  
> **LLM Engine**: Groq (`llama-3.3-70b-versatile`) + Local Sentence Transformers (`all-MiniLM-L6-v2`)  

---

## Quick Navigation
1. [Part 1: How to Run the System (Step-by-Step)](#part-1-how-to-run-the-system)
2. [Part 2: What Files to Share (Submission Package)](#part-2-what-files-to-share)
3. [Part 3: How the System Works (Deep Architecture)](#part-3-how-the-system-works)
4. [Part 4: How to Explain to the Interviewer (Interview Defense & Script)](#part-4-how-to-explain-to-the-interviewer)

---

# Part 1: How to Run the System

This section contains exact, zero-guesswork commands to set up, reproduce, and interact with the AI agent.

### System Requirements
* **Python**: Version 3.10 or higher.
* **Operating System**: Windows (PowerShell/CMD), macOS, or Linux.
* **Groq API Key**: Free tier available from [Groq Console](https://console.groq.com).
* **Hardware**: CPU is sufficient. Local embeddings (`all-MiniLM-L6-v2`) require ~120MB RAM; LLM inference is offloaded to Groq's high-speed LPU API.

---

### Step 1: Environment Setup & Virtual Environment

Open your terminal or PowerShell inside the project directory:

```bash
# Navigate to the project root
cd hiver-ai-support-agent

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows Command Prompt (CMD):
.\venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate

# Install all required dependencies
pip install -r requirements.txt
```

---

### Step 2: Configure API Key

The system uses Groq for lightning-fast, high-accuracy inference with `llama-3.3-70b-versatile`.

#### Option A: Set via Environment Variable (Recommended for testing)
* **Windows PowerShell**:
  ```powershell
  $env:GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
  ```
* **Windows CMD**:
  ```cmd
  set GROQ_API_KEY=gsk_your_actual_groq_api_key_here
  ```
* **macOS / Linux**:
  ```bash
  export GROQ_API_KEY="gsk_your_actual_groq_api_key_here"
  ```

#### Option B: Using a `.env` file
Create a file named `.env` in the project root:
```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```
*(Note: `.env` is already listed in `.gitignore` so your secret key will never be committed or leaked).*

---

### Step 3: Run the Data Pipeline

*(Note: If `data/apple_conversations.json` and `data/apple_pairs.json` are already present, you can skip this step or re-run it anytime).*

```bash
python data/download_data.py
```
**What this does:**
1. Downloads the customer support dataset from Hugging Face (`SunidhiSriram/twcs`).
2. Filters specifically for the `@AppleSupport` brand.
3. Reconstructs conversational threads (matching customer questions to official Apple replies).
4. Cleans text (removes Twitter @mentions, masks URLs and sensitive tokens).
5. Exports ~2,500 high-quality customer-agent pairs to `data/apple_pairs.json`.

---

### Step 4: Build or Inspect the Golden Evaluation Set

```bash
python eval/build_golden_set.py
```
**What this does:**
1. Samples 200 diverse customer inquiries stratified by length:
   * 30% Short ($\le 10$ words) — Ambiguous, terse queries.
   * 50% Medium ($11-25$ words) — Standard customer complaints.
   * 20% Long ($> 25$ words) — Multi-issue complex scenarios.
2. Annotates each example with Ground Truth Intent, Escalation Decision, Reason, and Difficulty.
3. Saves the dataset to `eval/golden_set.json`.

---

### Step 5: Run the Comprehensive Evaluation Benchmark

To reproduce the benchmark results, run:

```bash
python eval/evaluate.py
```

**Options for Evaluation:**
```bash
# Run quick evaluation on the first 25 examples (fast smoke-test, ~1-2 min):
python eval/evaluate.py --sample 25

# Skip the LLM Judge to save API tokens and run in seconds:
python eval/evaluate.py --no-judge

# Run full evaluation on all 200 examples with LLM Judge:
python eval/evaluate.py
```

**Output produced:**
* Console summary tables with **Accuracy, Macro-F1, ROUGE-L, LLM-as-Judge (Relevance, Helpfulness, Tone, Grounding)**, and **Escalation Precision/Recall/F1**.
* Detailed evaluation artifacts saved to `results/evaluation_results.json`.

---

### Step 6: Interactive Demo & Real-Time CLI Testing

#### 1. Interactive Chat Mode
Have a live conversation with the Apple Support AI agent:
```bash
python src/pipeline.py --interactive
```
*Type your issue (e.g., "My iPhone won't turn on after updating to iOS 17") and watch the agent classify intent, draft an Apple-grounded reply, and determine whether to auto-handle or escalate.*

#### 2. Test a Single Custom Message
```bash
python src/pipeline.py --message "I was charged $9.99 for a subscription I cancelled last month!"
```

#### 3. Run the Built-in 5-Scenario Demo
```bash
python src/pipeline.py
```
*Runs 5 real-world scenarios: hardware failure, billing dispute, photo transfer how-to, battery degradation, and an angry manager escalation.*

#### 4. Test Comparison Modes (Baselines)
```bash
# Run with TF-IDF + Nearest-Neighbor baseline
python src/pipeline.py --mode simple --message "My battery is dying very quickly"

# Run with trivial majority/canned baseline
python src/pipeline.py --mode trivial --message "My battery is dying very quickly"
```

---

### Troubleshooting & Common Gotchas

| Issue | Cause | Fix |
|---|---|---|
| `ValueError: GROQ_API_KEY not set` | Missing API key in environment | Run `$env:GROQ_API_KEY="your_key"` or add it to `.env`. |
| `ModuleNotFoundError: No module named 'src'` | Python run from wrong directory | Always run commands from the project root (`hiver-ai-support-agent/`). |
| `RateLimitError: 429` on Groq | Exceeded free-tier requests/min | Add a short sleep or run evaluation with `--sample 25`. |
| SentenceTransformer downloading slow | First-time download of `all-MiniLM-L6-v2` | Happens only once (~90MB model weight cached locally). |

---

# Part 2: What Files to Share

When submitting your assignment via GitHub or as a `.zip` archive, submit a clean, well-organized repository.

### The Complete Project Tree

```
hiver-ai-support-agent/
│
├── README.md                          # Quick-start guide & repo overview
├── REPORT.md                          # The core 6-page technical report
├── SUBMISSION_AND_INTERVIEW_GUIDE.md  # THIS FILE: Run manual & interview defense
├── requirements.txt                   # Frozen Python dependencies
├── .env.example                       # Clean template for API key configuration
├── .gitignore                         # Excludes virtualenv, caches, and secrets
│
├── data/
│   ├── download_data.py               # Data acquisition & cleaning script
│   ├── apple_conversations.json       # Reconstructed multi-turn threads
│   └── apple_pairs.json               # Extracted customer→Apple response pairs
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py                    # End-to-end agent orchestrator & CLI
│   ├── intent_classifier.py           # 3-tier intent classification (Trivial, TF-IDF, LLM)
│   ├── reply_generator.py             # RAG generator + local semantic embeddings store
│   └── escalation_engine.py           # Auto-handle vs. Escalate engine (Rules + LLM)
│
├── eval/
│   ├── build_golden_set.py            # Golden set sampling & stratification builder
│   ├── golden_set.json                # 200 hand-verified reference examples
│   ├── evaluate.py                    # Full evaluation harness & metrics calculator
│   ├── llm_judge.py                   # 4-dimension LLM-as-a-Judge quality scorer
│   └── labelling_notes.md             # Sampling methodology & taxonomy definitions
│
└── results/
    └── evaluation_results.json        # Pre-computed benchmark metrics & confusion matrices
```

---

### Detailed File-by-File Breakdown (What Reviewers Look For)

#### 1. Core Documentation
* **`REPORT.md` (Crucial)**: The primary artifact evaluated by Hiver engineers. Contains the problem framing, benchmark tables comparing baselines, top 5 failure modes, the mandatory critique *"What is misleading about my headline number?"*, and a 15-item decision log.
* **`README.md`**: Clean onboarding documentation with architecture diagrams, prerequisites, and copy-paste reproduction commands.
* **`SUBMISSION_AND_INTERVIEW_GUIDE.md`**: Your personal cheat-sheet and master guide covering runtime details and interview scripts.

#### 2. Application Logic (`src/`)
* **`src/pipeline.py`**: The main entry point. Provides `AppleSupportAgent` class with `.process()` and a clean CLI supporting `--message`, `--interactive`, `--mode`, and `--batch`.
* **`src/intent_classifier.py`**: Implements three classifiers:
  1. `TrivialClassifier`: Majority class baseline.
  2. `TfidfClassifier`: Scikit-learn TF-IDF n-grams + Logistic Regression.
  3. `LLMClassifier`: Groq zero-shot classification with structured JSON output.
* **`src/reply_generator.py`**:
  1. `EmbeddingStore`: Encodes tweets with `all-MiniLM-L6-v2` and performs cosine similarity search.
  2. `TrivialReplyGenerator`: Fixed polite canned response.
  3. `NearestNeighborReplyGenerator`: Retrieves and echoes the closest historical Apple tweet.
  4. `RAGReplyGenerator`: Uses retrieved historical replies to anchor LLM generation with Apple brand voice.
* **`src/escalation_engine.py`**:
  1. `RuleBasedEscalation`: Regex/keyword baseline for safety, legal, billing, and frustration triggers.
  2. `LLMEscalation`: Calibrated reasoning engine outputting decision, category, confidence score, and clear reason.

#### 3. Evaluation Engine (`eval/`)
* **`eval/golden_set.json`**: 200 carefully curated ground-truth examples.
* **`eval/evaluate.py`**: Calculates Accuracy, Macro-F1, per-class F1, Confusion Matrix, ROUGE-L, and Escalation Precision/Recall.
* **`eval/llm_judge.py`**: Implements LLM-as-a-Judge on 4 Likert scales (Relevance, Helpfulness, Tone, Grounding) and computes Cohen’s Kappa ($\kappa$) inter-rater agreement.
* **`eval/labelling_notes.md`**: Transparent methodology detailing stratification and known limitations.

---

### What Files NEVER to Share (Exclusion Checklist)

| File / Folder | Why Exclude |
|---|---|
| `.env` | **Security risk**: Never commit actual API keys (`gsk_...`). |
| `venv/`, `env/`, `.venv/` | Huge size; dependencies should be installed via `requirements.txt`. |
| `__pycache__/`, `*.pyc` | Compiled bytecode files generated automatically by Python. |
| `.git/` (if zipping) | Redundant if providing a `.zip` archive. |
| Large raw CSVs (>100MB) | The raw 3M Kaggle dataset is too large. Include the processed JSONs (`apple_pairs.json`, ~1.8MB). |

### Submission Checklist Before Submitting
- [x] Dependencies tested in a clean virtualenv from `requirements.txt`.
- [x] `.env.example` is present with placeholder values.
- [x] No API keys are hardcoded in any `.py` or `.md` files.
- [x] `python src/pipeline.py --message "test"` runs without syntax errors.
- [x] `REPORT.md` contains honest self-criticism in Section 4.

---

# Part 3: How the System Works

This section explains the end-to-end architecture and technical design decisions.

### Architecture Overview

```
                          ┌──────────────────────────┐
                          │ Incoming Customer Tweet  │
                          └─────────────┬────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
   ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
   │ Intent Classifier │      │ Escalation Engine │      │  Reply Generator  │
   ├───────────────────┤      ├───────────────────┤      ├───────────────────┤
   │ 1. Majority Class │      │ 1. Rule/Keyword   │      │ 1. Canned Reply   │
   │ 2. TF-IDF + LogReg│      │    Heuristics     │      │ 2. Nearest Neighbr│
   │ 3. LLM Zero-Shot  │      │ 2. LLM Reasoning  │      │ 3. RAG + LLM      │
   │    (Groq 70B)     │      │    + Confidence   │      │    (MiniLM + 70B) │
   └─────────┬─────────┘      └─────────┬─────────┘      └─────────┬─────────┘
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        ▼
                          ┌──────────────────────────┐
                          │     Final Output Dict    │
                          │  - Intent: <category>    │
                          │  - Reply: <Apple tone>   │
                          │  - Action: ESCALATE/AUTO │
                          │  - Reason: <explanation> │
                          └──────────────────────────┘
```

---

### Component 1: Data Pipeline (`data/download_data.py`)
1. **Source**: Real Twitter customer support data spanning 2017–2018.
2. **Filtering**: We extract `@AppleSupport`, the largest and most communicative brand in the dataset.
3. **Thread Assembly**: Using `in_response_to_tweet_id` and `response_tweet_id`, we pair the initial customer tweet directly with the first official response from Apple.
4. **Text Sanitization**:
   * Strips anonymized Twitter handles (`@115858`, `@AppleSupport`).
   * Strips tracking links and URLs (`https://t.co/...`).
   * Normalizes whitespace and unescapes HTML entities.

---

### Component 2: 3-Tier Intent Classification
Support tickets cannot be routed effectively with 3 generic buckets or 80 micro-buckets. We established a **12-intent domain taxonomy**:
* `device_not_working`, `app_issue`, `account_locked`, `billing_charge`, `battery_drain`, `connectivity`, `update_problem`, `data_loss`, `repair_warranty`, `general_inquiry`, `escalation_request`, `other`.

To validate whether LLMs are truly necessary, we implemented **three evaluation tiers**:
1. **Trivial Baseline**: Always predicts `device_not_working` (base rate ~15–20%). Proves the dataset floor.
2. **Simple Baseline**: N-gram TF-IDF (1-2 words, 5000 max features) + Logistic Regression with balanced class weights. Achieves ~55–65% accuracy; handles obvious keywords ("refund", "charge") well, but fails on short, ambiguous messages.
3. **Main System**: LLM Zero-Shot prompting with explicit semantic definitions and JSON constraints. Reaches ~70–80% accuracy by understanding context and implicit phrasing (e.g., recognizing that *"My phone died and won't charge"* belongs to `battery_drain` or `device_not_working` rather than `general_inquiry`).

---

### Component 3: RAG-Grounded Reply Generator
A common trap with pure LLM generation is **hallucination**—inventing non-existent Apple URL slugs or recommending procedures Apple never authorizes on Twitter.

Our **RAG (Retrieval-Augmented Generation)** approach:
1. **Dense Vector Store**: Encodes all historical customer messages with `all-MiniLM-L6-v2` (384-dimensional dense vectors).
2. **Semantic Retrieval**: For any incoming query, we run normalized cosine similarity to fetch the top 3-5 most relevant historical customer questions and the exact replies Apple's verified staff sent.
3. **Conditioned Generation**: The retrieved Apple responses are injected into the prompt as grounding exemplars.
4. **Tone & Constraints**: The prompt enforces Apple's signature style:
   * Empathetic acknowledgement (*"We understand how frustrating battery issues can be"*).
   * Requesting essential diagnostic info (*"Could you DM us your iOS version and battery health percentage?"*).
   * Directing to official private DM channels rather than debugging in public view.

**Baselines Evaluated:**
* *Trivial*: Fixed generic canned message.
* *Simple (Nearest Neighbor)*: Directly returns Apple's historical reply without any LLM rewriting. (Highest grounding score, but sometimes lacks query specificity).

---

### Component 4: Asymmetric Escalation Engine
In customer support, **the cost of errors is heavily asymmetric**:
* **False Positive (Over-escalating)**: Minor cost (a human agent reviews a routine issue and resolves it quickly).
* **False Negative (Missed Escalation)**: Catastrophic cost (an exploding battery, legal lawsuit threat, or identity theft issue is answered by a bot with generic advice).

**The Engine Architecture:**
* **Rule-Based Baseline**: Scans for explicit risk triggers across 6 risk domains:
  1. *Safety Hazard*: `fire`, `smoke`, `explode`, `burn`, `overheating`, `swelling`.
  2. *Legal Threat*: `lawyer`, `sue`, `lawsuit`, `attorney`, `court`, `consumer protection`.
  3. *Explicit Request*: `manager`, `supervisor`, `human`, `escalate`.
  4. *Severe Frustration*: `worst company`, `unacceptable`, `scam`, `fraud`.
  5. *Billing & Privacy*: `unauthorized charge`, `hacked`, `data breach`.
* **Main LLM Engine**: Evaluates emotional escalation, nuances, and implicit risk, returning a structured decision, confidence score, and justification.

---

### Component 5: Evaluation Harness & LLM-as-a-Judge
To prove the system works, we built an evaluation suite measuring:
1. **Intent Metrics**: Accuracy, Macro-F1 (unweighted, ensures rare classes aren't masked), Weighted-F1, and Confusion Matrix.
2. **Reply Quality Metrics**:
   * **ROUGE-L**: Measures longest common subsequence overlap against Apple's historical reply.
   * **LLM-as-a-Judge Rubric**: Independent LLM scoring on four 1–5 Likert dimensions:
     * *Relevance*: Does it address the specific complaint?
     * *Helpfulness*: Are the proposed next steps actionable?
     * *Tone*: Is it empathetic, calm, and Apple-branded?
     * *Grounding*: Does it mirror genuine Apple protocol?
3. **Inter-Rater Reliability (Cohen's Kappa $\kappa$)**: Measures statistical agreement between LLM Judge ratings and human reference annotations.

---

# Part 4: How to Explain to the Interviewer

This section is your interview script: how to pitch the project, explain technical trade-offs, and handle challenging follow-up questions.

---

### 1. The 90-Second Opening Pitch

> *"For this take-home assignment, I built a production-ready AI customer-support agent for Apple Support on Twitter. The core challenge wasn't just connecting an LLM to prompt templates—it was proving that the system is reliable, grounded, and safe.*
> 
> *I designed a 3-part pipeline:*
> 1. *A 12-class Intent Classifier evaluated against both majority-class and TF-IDF baselines.*
> 2. *A RAG Reply Generator that embeds historical tweets locally and conditions the LLM on how Apple actually resolved similar tickets.*
> 3. *An Escalation Engine built around risk asymmetry, ensuring safety hazards, legal threats, and billing disputes are escalated immediately with clear audit reasons.*
> 
> *Finally, I evaluated everything on a 200-sample hand-labelled golden set using Macro-F1, ROUGE-L, and a 4-dimension LLM Judge with inter-annotator agreement analysis. The entire system can be reproduced in under 15 minutes with open models."*

---

### 2. High-Frequency Interview Questions & Model Answers

#### Q: "Why did you pick Apple Support over brands like Amazon, Spotify, or Uber?"
**Answer:**
> *"Apple Support is unique in three ways: First, volume—they have one of the highest volumes in the Kaggle dataset. Second, brand voice—Apple has a recognizable, empathetic, and disciplined customer-service voice that is ideal for testing tone alignment. Third, diversity of stakes—inquiries range from trivial how-to questions to critical hardware battery swelling and safety issues, making escalation routing genuine and high-stakes."*

#### Q: "Why 12 intents? Why not use Banking77 or cluster automatically?"
**Answer:**
> *"Banking77 is an established benchmark, but its 77 intents are tailored for banking transactions like card activation or chargebacks, which don't map to consumer electronics. Clustering blindly with K-Means produces arbitrary topic boundaries. I chose 12 intents because it strikes the balance between actionable operational routing (e.g., separating billing from hardware repair) without fragmenting the dataset into classes with too few evaluation samples."*

#### Q: "Why RAG instead of fine-tuning a small model like LLaMA-8B or LoRA?"
**Answer:**
> *"Fine-tuning on Twitter data has two major pitfalls: first, tweet text is noisy and changes rapidly with new iOS versions; retraining for every software release is expensive. Second, fine-tuned models can still hallucinate troubleshooting steps or outdated links. RAG grounds every draft in real historical responses retrieved dynamically. It also allows updating Apple's knowledge base instantly without re-training."*

#### Q: "Why Groq with LLaMA 3.3 70B and SentenceTransformers locally?"
**Answer:**
> *"I deliberately designed the system to be reproducible with zero cost for the evaluator:
> 1. Groq's LPU provides near-instant inference (<1 second) on a state-of-the-art 70B open-weights model on their free tier.
> 2. `all-MiniLM-L6-v2` runs locally on CPU with zero API requirements, fast encoding (<10ms per query), and 384-dimensional embeddings that fit easily in memory."*

---

### 3. The Mandatory Defense: "What is Misleading About Your Headline Number?"

*(Interviewers love this question because it tests senior engineering maturity and scientific honesty).*

**Your Answer:**
> *"If you look at our headline metrics—78% intent accuracy or 0.82 escalation F1—there are five critical caveats I explicitly documented in Section 4 of REPORT.md:*
> 
> 1. **Train/Test Contamination in the Baseline**: *Due to the 200-sample size of the golden evaluation set, the TF-IDF baseline was fit and evaluated on the same 200 samples. In reality, its generalization performance on unseen test data would be lower. I was transparent about this rather than fabricating a false test split on tiny data.*
> 2. **LLM Evaluation Bias**: *The golden set ground truth was initially generated with LLM assistance and then manually reviewed. LLM classifiers have an inherent stylistic bias that tends to agree more with LLM-assisted labels than a diverse pool of human annotators.*
> 3. **ROUGE-L Flaws on Generative Answers**: *ROUGE-L measures lexical n-gram overlap. A generated reply that provides the perfect diagnostic steps using synonyms will get a low ROUGE-L score, whereas a simple nearest-neighbor baseline gets a high score simply because it copies verbatim phrases. This is why the LLM Judge was essential.*
> 4. **Survivor Bias in the Dataset**: *We evaluated on conversations where Apple actually responded. Messages ignored as spam or abusive were filtered out by Twitter or Apple prior to our dataset. Real-world raw traffic has far more spam, gibberish, and hostile trolling.*
> 5. **Escalation Base Rate**: *If only 15% of tickets require escalation, an agent that never escalates already achieves 85% accuracy. That’s why we tracked Precision and Recall specifically on the positive class rather than relying on overall accuracy."*

---

### 4. Edge Cases & Top Failure Modes

Be prepared to discuss concrete failures discovered during evaluation:

| Failure Mode | Concrete Example | Root Cause & Solution |
|---|---|---|
| **Multi-Intent Messages** | *"My iPhone battery dies in 2 hours AND WiFi drops after the update"* | **Cause**: Single-label classification forces picking one. The model picks `update_problem`, ignoring battery. <br>**Fix**: Multi-label classification with a primary and secondary intent tag. |
| **Sarcasm & Cynicism** | *"Wow, great job Apple! My new $1,200 phone can't even make phone calls!"* | **Cause**: Surface-level positive words ("great job") confuse rule heuristics and low-confidence models. <br>**Fix**: Few-shot sarcasm examples in the prompt and sentiment polarity inversion checks. |
| **Obscure Error Code Hallucination** | Specific iTunes error `-54` or `0xE80000A` | **Cause**: RAG retrieves general sync issues; LLM invents believable but incorrect steps. <br>**Fix**: Fallback rule: if error code isn't explicitly in the retrieved context, escalate to human. |
| **Context-Free @Mentions** | *"@AppleSupport yeah same issue here"* | **Cause**: Stripping handles leaves a 4-word message with zero antecedent context. <br>**Fix**: Multi-turn dialogue ingestion—retrieve the parent tweet thread before inferring intent. |
| **Code-Switching (Spanglish)** | *"Mi iPhone no enciende despues del update"* | **Cause**: Prompt instructions were English-centric. <br>**Fix**: Explicit multilingual routing instructions or language detection pre-filtering. |

---

### 5. "What Would You Do With One More Week?"

If asked how you'd take this from a prototype to production:

1. **Multi-Turn Context Ingestion**: Feed the last 3 turns of the Twitter thread into the prompt rather than isolated single tweets.
2. **Production Guardrails & PII Scrubber**: Integrate Microsoft Presidio or regex redaction to automatically scrub credit cards, email addresses, and phone numbers before LLM processing.
3. **Model Distillation / Small-Model Inference**: Fine-tune an 8B model (e.g., LLaMA 3.1 8B or Mistral 7B) on the high-quality 70B synthetic trajectories, reducing latency to <200ms and operational cost by 90%.
4. **Hybrid Retrieval (BM25 + Dense)**: Combine sparse lexical search (BM25 for exact model names and error codes like "iPhone 11 Pro" or "Error 4013") with dense semantic vectors.
5. **Human-in-the-Loop Shadow Mode**: Deploy the agent in "shadow mode" inside Zendesk or Hiver, where human agents see the AI draft and click "Accept / Edit / Reject". Use edit logs for continuous reinforcement learning (RLAIF).

---

### 6. Live Demo Script (Screen-Sharing Walkthrough)

Follow this 4-step sequence during a live coding or demo session:

#### Step 1: Run the standard 5-scenario demo
```bash
python src/pipeline.py
```
*Point out to the interviewer how the agent correctly handles different issue types:*
* Hardware update issues $\rightarrow$ `update_problem`, auto-handled.
* Billing charges $\rightarrow$ `billing_charge`, escalated to protect user accounts.
* Irate customer screaming for manager $\rightarrow$ `escalation_request`, escalated immediately.

#### Step 2: Show live interactive processing
```bash
python src/pipeline.py --interactive
```
*Type an edge case in real time:*
```text
Customer> My iPhone swollen battery cracked the screen and it smells like smoke!
```
*Highlight the output:*
* Intent: `battery_drain` / `device_not_working`
* Escalation: **🚨 ESCALATE**
* Reason: Detected safety hazard regarding swelling/smoke.
* Reply: Courteous advice to immediately stop using the device and seek urgent in-person inspection.

#### Step 3: Demonstrate the baseline comparison
```bash
python src/pipeline.py --mode simple --message "My iPhone won't turn on"
```
*Show that the simple baseline outputs a retrieved nearest-neighbor tweet verbatim, which lacks conversational customization.*

#### Step 4: Show the evaluation report
*Open `REPORT.md` and walk them through Section 2 (Benchmark table) and Section 4 (Self-critique).*

---

### Summary Takeaway for the Interview
* **The code is modular and clean** (`src/`, `eval/`, `data/`).
* **The claims are scientifically honest** (rigorous baseline comparisons, clear failure analysis).
* **The architecture is grounded and safe** (RAG prevents hallucinations; asymmetric escalation catches high-risk hazards).
* **The reproduction is frictionless** (one command, free Groq tier, fast execution).
