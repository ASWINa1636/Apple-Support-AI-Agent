# Golden Evaluation Set — Labelling Notes

## Sampling Strategy

The golden evaluation set contains **200 hand-labelled examples** sampled from Apple Support conversation pairs on Twitter.

### How examples were sampled:

1. **Source**: Customer→Apple response pairs extracted from the Twitter Customer Support dataset, filtered to AppleSupport brand conversations only.

2. **Stratification**: Examples were stratified by message length to ensure diversity:
   - 30% short messages (≤10 words) — tests handling of terse/ambiguous inputs
   - 50% medium messages (11-25 words) — the bulk of real traffic
   - 20% long messages (>25 words) — complex/multi-issue scenarios

3. **Random seed**: `42` for reproducibility.

4. **Filtering**: Messages shorter than 10 characters were excluded (usually just @mentions or emojis with no substantive content).

### How examples were labelled:

1. **Initial labelling**: Each example was labelled by an LLM (llama-3.1-70b-versatile via Groq) with the following fields:
   - `intent`: One of 12 categories from our taxonomy
   - `should_escalate`: Boolean
   - `escalation_reason`: Free text explanation
   - `difficulty`: easy/medium/hard
   - `notes`: Edge case annotations

2. **Manual review**: All 200 examples were reviewed by hand to:
   - Correct obvious misclassifications
   - Handle ambiguous multi-intent messages (assigned to primary intent)
   - Validate escalation decisions against real-world criteria
   - Flag edge cases and add notes

3. **Labeller field**: Each example has a `labeller` field:
   - `llm_initial` — LLM-generated, not yet reviewed
   - `human_reviewed` — Manually reviewed and corrected
   - `heuristic` — Fallback heuristic labelling (if LLM unavailable)

### Known limitations:

- **Subjectivity**: Intent boundaries are inherently fuzzy. Some messages could reasonably be classified into 2+ categories. We defaulted to the most actionable interpretation.
- **Escalation ground truth**: "Should escalate" is a policy decision, not an objective fact. Our labels reflect a reasonable default policy for Apple Support.
- **Sample bias**: The sampling is from conversation pairs only (where Apple actually responded). Messages that were ignored or unanswered are not represented.
- **Temporal bias**: The dataset spans a specific time period. Product names, iOS versions, and common issues may not generalize to other periods.

### Intent Taxonomy (12 categories):

| Intent | Description |
|--------|-------------|
| `device_not_working` | Device won't turn on, frozen, unresponsive, or crashing |
| `app_issue` | App crashing, not downloading, not updating |
| `account_locked` | Apple ID locked, password reset, 2FA issues |
| `billing_charge` | Unexpected charges, subscription issues, refunds |
| `battery_drain` | Battery draining fast, not charging |
| `connectivity` | WiFi, Bluetooth, cellular problems |
| `update_problem` | iOS/macOS update failed or causing issues |
| `data_loss` | Lost photos, contacts, backup/restore issues |
| `repair_warranty` | Repair status, warranty questions |
| `general_inquiry` | General questions, how-to, feedback |
| `escalation_request` | Explicit request for supervisor/formal complaint |
| `other` | Doesn't fit above categories |
