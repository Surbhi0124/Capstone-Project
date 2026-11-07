# Original Support track — grading rubric audit

This audit maps all 16 tasks from the provided Ola Support capstone brief to actual project locations. A source file's presence does **not** by itself prove a live integration; where a transcript is needed, run the specified command on the submission laptop and keep its output.

| Task | Requirement | Source / evidence | Final check |
|---|---|---|---|
| 1 | Seeded >=40 tickets, all categories/statuses, flag 10–30% | `dataset.py`, `data/support_tickets.json`; `python dataset.py` | Run and retain output |
| 2 | >=12 original policies | `knowledge_base/policies.json` | Verify original fictional contents |
| 3 | Two chunking strategies + independent Chroma collections | `rag/core.py`, `rag/build_index.py` | `python -m rag.build_index` |
| 4 | Grounded >=5 in-scope and 1 fallback, empirically calibrated threshold | `evaluation/run_rag.py`, `rag/core.py`, `evaluation/rag_results.json` | Run and retain output |
| 5 | Per-query document-level precision and recall for both methods | `evaluation/run_rag.py` and measured results | Preserve arithmetic and recommendation |
| 6 | Ticket lookup + designed bounded escalation score | `agents/lookup.py`, `agents/governance_phase6.py` | Document dataset-based threshold rationale |
| 7 | Three CrewAI agents, native tools, real `.kickoff()` | `agents/crew_phase3b.py` | `python -m agents.crew_phase3b`; capture tool events |
| 8 | Memory continuity and distinct fresh session | `agents/session_phase4.py` | `python -m agents.session_phase4` |
| 9 | Pydantic output validation | `AgentAnswer`, `Phase4Chat`, `api/app.py` | Full tests |
| 10 | Phone masking, injection and unsupported-output guards | `agents/guardrails.py`, `agents/session_phase4.py` | Tests and demo |
| 11 | Two HTTP endpoints + WebSocket disconnection | `api/app.py`, `tests/test_api_phase5.py` | Swagger + tests |
| 12 | Trace ID, timing and masked JSONL | `api/secure_logging.py`, API tests | Verify `logs/requests.jsonl` locally; do not commit logs |
| 13 | 15 cases, 4 per-case metrics and 4 averages under Mock LLM | `evaluation/judge_phase6.py`, `evaluation/capstone_cases.json` | Generate JSON/CSV on laptop |
| 14 | AutoGen two-person RoundRobin review, approve/revise | `agents/review_phase6.py`, `tests/test_phase6.py` | `python -m agents.review_phase6` |
| 15 | Least autonomy, risk, budget rejection | `agents/governance_phase6.py`, `docs/GOVERNANCE.md` | `python -m agents.governance_phase6` |
| 16 | Reviewed policy-only cache, repeated query hit | `agents/pipeline_phase6.py`, `agents/governance_phase6.py` | `python -m agents.pipeline_phase6` |

## Must verify before submission

- Run all integration and regression tests on the grader-compatible Python environment (user previously reported **48 passing tests** before Phase 7 changes).
- Check **E03** continues to select `POL-03` via the CrewAI tool. Phase 7 includes a pure reranking regression test; the actual 15-case integration test must still pass.
- Record the final AutoGen demo after fixing synchronous CrewAI use within an async event loop.
- Rerun the **actual FastAPI `/ask`** after the Phase 6 review integration, not only the Phase 5 stub tests.
- Confirm local-only offline execution with packages and model weights preinstalled. Environment flags alone are not proof of zero network traffic.
- Verify `/add-document` updates both vector collections and invalidates cache using fabricated content.
- Save the student's own measured JSON and CSV artifacts. The Phase 7 ZIP intentionally does not invent metrics files not available here.
- Check `git status`/`git ls-files` for secrets, logs, model weights, index, and `.venv`.
