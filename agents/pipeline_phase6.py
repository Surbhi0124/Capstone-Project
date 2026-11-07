"""Phase 6 public chat pipeline: Phase4 memory -> CrewAI -> AutoGen review.

Policy responses are cached only AFTER review. Ticket responses are not cached.
The regular Phase4Chat boundary continues masking and screening every request.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from agents.session_phase4 import Phase4Chat
from agents.crew_phase3b import TICKET_RE, run_crew
from agents.governance_phase6 import BudgetPolicy, PolicyAnswerCache
from agents.review_phase6 import review_draft, apply_verdict


class Phase6Chat:
    def __init__(self, runner=run_crew, reviewer=review_draft, budget=None):
        self.runner = runner
        self.reviewer = reviewer
        self.budget = budget or BudgetPolicy()
        self.cache = PolicyAnswerCache()
        self.last_review = None
        self.inner = Phase4Chat(runner=self._run_reviewed)
        self._policy_version = self._policy_mtime()

    @staticmethod
    def _policy_mtime():
        from rag.core import ROOT
        p = ROOT / 'knowledge_base' / 'policies.json'
        return p.stat().st_mtime_ns if p.exists() else None

    def _run_reviewed(self, question: str):
        self.budget.enforce(question)
        current_version = self._policy_mtime()
        if current_version != self._policy_version:
            self.cache.clear()
            self._policy_version = current_version

        # Cache holds only previously REVIEWED policy outcomes.
        def checked_runner(q):
            answer, state = self.runner(q)
            verdict, transcript = asyncio.run(self.reviewer(answer.answer, state))
            self.last_review = transcript
            return apply_verdict(answer, verdict), state
        return self.cache.execute(question, checked_runner)

    def chat(self, session_id: str, question: str):
        # Oversized input is rejected BEFORE history persistence or crew invocation.
        self.budget.enforce(question)
        return self.inner.chat(session_id, question)

    def history(self, session_id: str):
        return self.inner.history(session_id)

    def reset(self, session_id: str):
        self.inner.reset(session_id)

    def clear_cache(self):
        self.cache.clear()


def main():
    chat = Phase6Chat()
    q = 'What is the first response time for a critical ticket?'
    for query in (q, q, 'What is the status of TKT-0001?'):
        response = chat.chat('phase6-demo', query)
        print('QUESTION:', query)
        print(response.model_dump_json(indent=2))
        print('CACHE:', {'hits': chat.cache.hits, 'misses': chat.cache.misses,
                         'crew_calls': chat.cache.crew_calls})
    print('Note: Real API spend is zero in MOCK_LLM mode.')


if __name__ == '__main__':
    main()
