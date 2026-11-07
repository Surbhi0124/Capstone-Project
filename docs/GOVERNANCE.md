# System governance and limitations

## Risk classification: Medium

This educational project responds to customer-support questions and reads synthetic ticket statuses. Wrong escalation guidance or a policy hallucination may mislead a user, so it has greater operational risk than simple transcription. It does **not** issue refunds, alter real accounts, or make financial/medical/hiring decisions. Accordingly, the brief's Low/Medium/High scale places this support-ticket agent at **Medium** risk. A production version needs authentication, per-user ticket access controls, and human oversight before consequential actions.

## Four-layer treatment

1. **Application:** `build_crew()` assigns `check_support_ticket_status` only to the Lookup Agent; `assert_least_autonomy()` verifies wiring. The Retrieval Agent is assigned only RAG. The Composer combines recorded evidence.
2. **Runtime:** `BudgetPolicy` blocks oversized inputs using documented *estimated* tokens and simulated USD costs, not actual provider billing. `Phase4Chat` screens injection and masks recognized fixed-format phone numbers before messages enter memory; output checks fail closed. A bounded in-memory cache stores only reviewed, grounded policy responses, never explicit ticket-ID lookups.
3. **Model/data:** CrewAI's custom `BaseLLM` is deterministic in MOCK mode; local SentenceTransformers embeddings and ChromaDB use synthetic policies and records. No API keys or paid models are required after dependencies and model weights have been prepared.
4. **Operations and organizational oversight:** FastAPI JSONL logging records trace IDs, status, timing, and masked request text, without raw policy bodies. Reproducible evaluation provides per-case quality gates; the two-person AutoGen review validates/revises drafts using original tool evidence. A human administrator remains responsible for policy changes, auditing errors, and risk acceptance.

## Escalation score justification

`score = round(0.7 * int(escalated) + 0.3 * (1 - days_since_created / 30), 4)`.
The synthetic dataset has 50 tickets, of which 9 have `escalated=True` (18%). With the documented threshold of **0.7**, historically escalated tickets score at least 0.7; non-flagged tickets never exceed 0.3. Consequently the threshold selects exactly those 9 flags in this generated dataset. This is a **transparent demonstration rule**, not evidence that 0.7 is a calibrated predictive risk threshold. Age contributes within each group; a different scoring model is needed for prioritizing unflagged tickets by urgency. A production scoring threshold should be calibrated against outcomes rather than inferred from mock data.

## Threats and limitations

- Regex masking does not reliably remove names, addresses, payment details, or all phone-number formats. Use fabricated records, never real PII.
- Prompt-injection pattern rules are illustrative, not comprehensive.
- Reranking is a simple domain heuristic; it was added for the measured E03 confusion between ticket escalation and priority customer handling. Keep the original E03 query and regression test.
- The deterministic AutoGen review is an orchestrated replay of predefined verdicts; it validates evidence consistency, **not** human-level independent semantic reasoning.
- The four evaluation metrics are deterministic rule-based scores on 15 authored examples, not external real-world benchmark accuracy.
- A working API is not equivalent to deployed public infrastructure. Local service and memory are process-bound.
- Caches do not store ticket-ID lookups, but production applications must also consider authorization, dynamic policy versions, concurrency, and access controls.
