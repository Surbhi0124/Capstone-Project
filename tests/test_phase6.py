"""Phase 6 deterministic unit tests; no embedding download required."""
import asyncio
import unittest
from types import SimpleNamespace

from agents.crew_phase3b import AgentAnswer, RunState
from agents.review_phase6 import deterministic_verdict, ReviewVerdict
from agents.governance_phase6 import BudgetPolicy, BudgetExceeded, PolicyAnswerCache
from evaluation.judge_phase6 import CASES, judge_case
import json


def policy_state():
    state = RunState(question='What is the SLA?', route='policy', ticket_id='', threshold=.31)
    state.policy = {'answer': 'Critical tickets receive an initial response within 15 minutes.',
                    'grounded': True, 'sources': ['POL-02'], 'top_similarity': .70}
    state.tool_events = [{'tool':'policy'}, {'tool':'ticket'}]
    state.lookup = {'found': False}
    return state


class Phase6Tests(unittest.TestCase):
    def test_reviewer_approve(self):
        state = policy_state()
        answer = state.final_answer()
        v = deterministic_verdict(answer.answer, state)
        self.assertTrue(v.approved)
        self.assertEqual(v.final_answer, answer.answer)

    def test_reviewer_revise_ungrounded_claim(self):
        state = policy_state()
        v = deterministic_verdict(state.final_answer().answer + ' Free cash for everybody!', state)
        self.assertFalse(v.approved)
        self.assertNotIn('Free cash', v.final_answer)

    def test_reviewer_rejects_pii(self):
        state = policy_state()
        v = deterministic_verdict('Phone 9876543210', state)
        self.assertFalse(v.approved)
        self.assertNotIn('9876543210', v.final_answer)

    def test_budget_rejects_large_request(self):
        with self.assertRaises(BudgetExceeded):
            BudgetPolicy().enforce('a' * 40000)
        self.assertEqual(BudgetPolicy().enforce('hello')['real_api_cost_usd'], 0)

    def test_cache_avoids_second_runner_call(self):
        cache = PolicyAnswerCache()
        calls = []
        def runner(question):
            calls.append(question)
            state = policy_state()
            return state.final_answer(), state
        first, _ = cache.execute('What is the SLA?', runner)
        second, _ = cache.execute('  WHAT   IS THE SLA?  ', runner)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first, second)
        self.assertEqual((cache.misses, cache.hits, cache.crew_calls), (1,1,1))

    def test_ticket_does_not_cache(self):
        cache = PolicyAnswerCache()
        calls = []
        def runner(question):
            calls.append(question)
            return AgentAnswer(answer='Ticket example', grounded=True, route='ticket',
                               sources=['TKT-0001'], ticket_id='TKT-0001'), policy_state()
        cache.execute('What is the status of TKT-0001?', runner)
        cache.execute('What is the status of TKT-0001?', runner)
        self.assertEqual(len(calls), 2)
        self.assertEqual(cache.hits, 0)

    def test_real_crew_least_autonomy_wiring(self):
        from agents.governance_phase6 import assert_least_autonomy
        assert_least_autonomy(policy_state())

    def test_phase6_pipeline_uses_reviewed_cache(self):
        from agents.pipeline_phase6 import Phase6Chat
        from agents.review_phase6 import deterministic_verdict
        calls = []
        async def fake_reviewer(draft, state):
            return deterministic_verdict(draft, state), {'offline': True}
        def fake_crew(question):
            calls.append(question)
            st = policy_state()
            return st.final_answer(), st
        chat = Phase6Chat(runner=fake_crew, reviewer=fake_reviewer)
        a = chat.chat('session-1', 'What is the SLA?')
        b = chat.chat('session-1', 'What is the SLA?')
        self.assertEqual(a, b)
        self.assertEqual(len(calls), 1)
        self.assertEqual(chat.cache.hits, 1)
        self.assertEqual(len(chat.history('session-1')), 4)

    def test_cases_cover_all_twelve_policies(self):
        cases = json.loads(CASES.read_text())
        self.assertEqual(len(cases), 15)
        self.assertEqual({c['expected_doc'] for c in cases if c['in_scope']},
                         {f'POL-{i:02d}' for i in range(1,13)})
        self.assertGreaterEqual(sum(not c['in_scope'] for c in cases), 2)

    def test_judge_scores(self):
        case = {'in_scope': True, 'expected_doc': 'POL-02', 'required_phrases': ['15 minutes']}
        result = judge_case(case, policy_state().final_answer())
        self.assertEqual(result, {'Accuracy':1.0, 'Grounding':1.0,
                                  'Completeness':1.0,'Safety':1.0})
        bad = AgentAnswer(answer='Unsourced claim', grounded=False,
                          sources=[], route='policy')
        self.assertEqual(judge_case(case, bad)['Grounding'], 0)

    def test_autogen_round_robin_when_installed(self):
        try:
            import autogen_agentchat
        except ImportError:
            self.skipTest('AutoGen not installed in this environment; required on submission laptop')
        from agents.review_phase6 import review_draft
        state = policy_state()
        answer = state.final_answer()
        approved, transcript = asyncio.run(review_draft(answer.answer, state))
        self.assertTrue(approved.approved)
        self.assertEqual(transcript['participants'], ['Policy_Compliance_Reviewer','Final_Editor'])
        revised, transcript = asyncio.run(review_draft(answer.answer+' False claim', state))
        self.assertFalse(revised.approved)
        self.assertEqual(revised.final_answer, answer.answer)


if __name__ == '__main__':
    unittest.main()
