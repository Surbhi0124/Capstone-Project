#  Capstone — Support (Ola)

**Selected track:** Ola (Business Operations / Customer Support)  
**Exact domain to enter in Question 1:** `Support`  
**Question 2:** Submit **one public GitHub repository URL** for this complete project.

**Project:** Offline, deterministic, production-minded *educational* customer-support agent. All tickets and policies are invented; they are **not** actual Ola records or official policies.

## Final status and reproducibility

Phases 1–6 have been exercised on the student's Windows development computer, where the student reported 48/48 unit tests passing before Phase 7, successful CrewAI and FastAPI demos, and all 15 deterministic evaluation cases scoring 1.00 on the four rubric metrics after fixing E03. **Those are student-reported runs; this archive does not contain machine-generated evaluation JSON/CSV from that computer.** Re-run the commands below to create the actual grading evidence in your own repo and confirm tests after the Phase 7 changes. The new E03 reranking tests and an asyncio fix are included here.

CrewAI uses a custom deterministic `BaseLLM`, not a hosted provider. AutoGen uses a local deterministic `ReplayChatCompletionClient` with two agents, `RoundRobinGroupChat(max_turns=2)`, and structured `ReviewVerdict`. Actual API usage costs **$0** in MOCK mode. Model weights and Python dependencies must first be installed/downloaded on an internet-connected environment, and then can be used offline; they are intentionally not included in Git.

## Dataset choices — required Part 1 disclosure

| Choice | Value |
|---|---|
| Seed | `260313`, private `random.Random` instance |
| Ticket count | 50 |
| Categories and sampling weights | `Billing:26`, `Technical Issue:25`, `Account Access:19`, `Product Defect:15`, `General Inquiry:15` |
| Statuses and sampling weights | `Open:24`, `In Progress:27`, `Escalated:10`, `Resolved:24`, `Closed:15` |
| Escalated Boolean draw | independent Bernoulli probability `0.20` |
| `resolution_time_hours` | randomly chosen integer from **1 to 72** hours; illustrative support operations range |
| `days_since_created` | integer 0–30 inclusive |

Actual seeded data: 50 total; Billing 16, Technical Issue 6, Account Access 11, Product Defect 9, General Inquiry 8; Open 20, In Progress 10, Escalated 5, Resolved 7, Closed 8; `escalated=True` on 9 (18%). Records are not manually altered to force these results.

`knowledge_base/policies.json` has 12 original 2–5 sentence policies covering priority classification, SLA, escalation, refunds, channels, business-hours/holidays, repeat complaints, service credits, feedback, VIP treatment, outages and retention.

## Install and run (Windows PowerShell, from repo root)

Requires a supported Python 3.12 environment and a virtual environment. If the same model is already downloaded and ChromaDB built, retain local `models/`, `data/chroma_index/`, and the student's measured `evaluation/rag_results.json` when applying incremental changes.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-phase6.txt
# Once online, download model weights to ./models/:
python -m rag.prepare_model --download-model
python dataset.py
python -m rag.build_index
python -m evaluation.run_rag
```

**Important:** The next commands use local weights. CrewAI telemetry and model downloads are disabled in the application, and the environment variables make offline intent explicit. For an actual no-network demonstration, disconnect networking or apply a network blocker *after* installing packages/weights. Do not run the `--download-model` step offline.

```powershell
$env:CREWAI_DISABLE_TELEMETRY='true'
$env:OTEL_SDK_DISABLED='true'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python -m agents.crew_phase3b
python -m agents.session_phase4
python -m agents.review_phase6
python -m agents.governance_phase6
python -m agents.pipeline_phase6
python -m evaluation.judge_phase6
python -m unittest discover -s tests -v
python -m scripts.check_submission --strict
```

The preflight `--strict` requires locally generated `evaluation/rag_results.json`, `evaluation/phase6_results.json` and `evaluation/phase6_results.csv`. It does not verify GitHub publication or prove network isolation.

## RAG methodology and measured results

`rag/core.py` uses 300-character chunks with 70-character overlap versus groups of two sentences. Both include policy titles and parent document IDs; `rag/build_index.py` uses local `all-MiniLM-L6-v2` embeddings and Chroma cosine collections `support_fixed_overlap` and `support_sentence`. The measured fallback thresholds are determined empirically from top-1 similarity clusters on 6 in-scope and 3 deliberately unrelated queries.

| Metric | Fixed overlap | Sentence |
|---|---:|---:|
| Mean document Precision@3, 6 queries | 0.4444 | **0.4722** |
| Mean document Recall@3, 6 queries | 1.0000 | **1.0000** |
| Empirical fallback threshold | 0.3322565 | **0.3189165** |
| Minimum in-scope top-1 similarity | 0.5496 | 0.5408 |
| Maximum out-of-scope top-1 similarity | 0.1149 | 0.0970 |

Per-question precision arithmetic for fixed: `1/2, 1/2, 1/2, 1/3, 1/3, 1/2`; for sentence: `1/2, 1/2, 1/2, 1/3, 1/2, 1/2`. Recall for each question in both strategies: `1/1`. Thus sentence chunking is selected for its slightly higher average precision at equal recall on these **six examples**, not asserted as universally better. `python -m evaluation.run_rag` prints the question-level scores and at least five grounded demos plus an out-of-scope refusal.

**Post-evaluation improvement:** The original E03 question, *“Who receives an unresolved high priority support issue?”*, retrieved `POL-10` ahead of `POL-03` (raw cosine 0.6092 vs 0.5693). `rerank_hits()` applies a small title/content boost for unresolved escalation intent, while preserving original cosine scores and retrieval threshold. The reranking operates in the CrewAI policy tool as well as the direct RAG baseline. The original E03 evaluation case is unchanged and has a regression test. Confirm the fix using live `python -m evaluation.judge_phase6` on the target machine.

## Multi-agent workflow and verification

- **CrewAI:** Retrieval Agent owns RAG tool, Lookup Agent owns `check_support_ticket_status`, Composer returns a validated Pydantic `AgentAnswer`. Execution uses real `.kickoff()` and records tool invocations. Both tools are called per demo (empty ticket IDs return an explicit missing-ID result for policy queries).
- **Ticket escalation:** `score = 0.7 * int(escalated) + 0.3 * (1 - days_since_created/30)`, rounded to four decimals. Threshold **0.7** selects the 9 historically escalated records (18%) in the seeded dataset. This is an explanatory demonstration threshold, **not** statistically trained escalation prediction; see `docs/GOVERNANCE.md`.
- **Memory/guardrails:** LangChain `RunnableWithMessageHistory` uses per-session `InMemoryChatMessageHistory` and supports fresh-session isolation; PII masking, prompt-injection rejection, unsupported-output refusal, and Pydantic checks are independently tested. LangChain deprecation warnings are expected under the assignment's permitted API.
- **AutoGen independent review:** Policy Compliance Reviewer and Final Editor speak sequentially in a two-turn RoundRobin team. The editor emits structured `approved`, `final_answer` and `reason`; a deterministic evidence check validates that the verdict matches original tool evidence. Run `python -m agents.review_phase6` to demonstrate both approve and revise.
- **Governance:** Medium-risk customer support; least-autonomy tool wiring; request input/token/simulated-cost cap; reviewed-policy-only in-memory LRU cache, with cache-hit counter and no ticket response caching. Details in `docs/GOVERNANCE.md`.

## API and logging

```powershell
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. Endpoints: `POST /ask`, `POST /add-document` and WebSocket `/ws/chat` (plus `GET /health`). `/ask` takes `{"session_id":"demo-A", "question":"What is the status of TKT-0001?"}`; follow with `{"session_id":"demo-A", "question":"Does it need escalation?"}`. Another session should not inherit the ticket ID. `POST /add-document` is for **fictional** policies; it updates both Chroma collections and invalidates cached answers. Log output is `logs/requests.jsonl`, ignored by Git, with masked request text, trace ID and durations. WebSocket disconnect is handled without terminating the server.

Swagger `/docs` returned HTTP 200 in the student demo, and the real `/ask` returned ticket data. Retest `/ask` **after** enabling Phase 6 review, and manually verify the WebSocket and policy-write path in the final environment.

## Evaluation, scores and evidence

`evaluation/capstone_cases.json` contains 15 cases (12 policy topics plus 3 out-of-scope). `evaluation/judge_phase6.py` is a **deterministic rule-based MOCK_LLM judge**, checking expected source, required answer phrases, fallback and phone masking. It does not perform independent semantic LLM judging.

On the student's final reported evaluation (after the E03 reranking fix), all 15 cases earned `Accuracy=1.0`, `Grounding=1.0`, `Completeness=1.0`, `Safety=1.0` with means **1.0000** each. This is a 15-case synthetic test result, not a real-world performance claim. Execute the evaluator to generate and commit `evaluation/phase6_results.json` and `.csv`; **do not fabricate those files**. Commit the measured `evaluation/rag_results.json` as well, after checking it contains no personal information.

For reproducibility and final audit, consult `docs/RUBRIC_AUDIT.md` and `docs/GOVERNANCE.md`. Original exercise materials need not be committed separately, but capture the live offline transcript of all tasks (commands below).

## Capture grading transcripts and publish

After preparing local model and index, run in PowerShell:

```powershell
New-Item -ItemType Directory -Force docs/transcripts | Out-Null
python -m agents.crew_phase3b 2>&1 | Tee-Object docs/transcripts/crew_demo.txt
python -m agents.session_phase4 2>&1 | Tee-Object docs/transcripts/memory_demo.txt
python -m agents.review_phase6 2>&1 | Tee-Object docs/transcripts/autogen_review.txt
python -m agents.governance_phase6 2>&1 | Tee-Object docs/transcripts/governance_demo.txt
python -m agents.pipeline_phase6 2>&1 | Tee-Object docs/transcripts/cache_demo.txt
python -m evaluation.run_rag 2>&1 | Tee-Object docs/transcripts/rag_evaluation.txt
python -m evaluation.judge_phase6 2>&1 | Tee-Object docs/transcripts/final_evaluation.txt
python -m unittest discover -s tests -v 2>&1 | Tee-Object docs/transcripts/tests.txt
python -m scripts.check_submission --strict
```

Inspect transcripts before publishing: **never commit real phone numbers, personal data, API credentials, local user paths, or private logs**. The tests use only fabricated examples.

```powershell
git init
git add .
git status --short
git ls-files
# Check staged files for unintended personal/sensitive content, then:
git commit -m "Final offline support agent capstone"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

Create the empty public GitHub repository yourself and substitute your actual URL. Submit **only** that real URL in Question 2. Do not copy the placeholder URL. Enter **only `Support`** in Question 1.

## Caveats

No React, paid account, PDF ingestion or unrelated agent framework is required. Library/API versions are not tightly pinned; maintain a tested Python environment and consider recording `pip freeze` after confirming the tests, without including secrets. All generated artifacts are examples for academic evaluation. See the governance document for security limitations and human oversight needs.
