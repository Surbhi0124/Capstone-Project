"""Offline logic tests; a real CrewAI integration smoke run is in the demo command."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

@unittest.skipUnless(importlib.util.find_spec('crewai') is not None,
                     'Install CrewAI (requirements-phase3b.txt) to run Phase 3B tests')
class CrewPhase3BTests(unittest.TestCase):
    def setUp(self):
        from agents.crew_phase3b import RunState
        self.state = RunState('question', 'policy', '', 0.3)

    def test_tool_dispatch_uses_args_schema_not_tool_name(self):
        from agents.crew_phase3b import tool_call_from_schema
        tools = [
            {'function': {'name': 'rag_lookup', 'parameters': {'properties': {'query': {'type':'string'}}}}},
            {'function': {'name': 'something_else', 'parameters': {'properties': {'record_id': {'type':'string'}}}}},
        ]
        actual = tool_call_from_schema(tools, 'query', 'hi')
        self.assertEqual(actual[0]['function']['name'], 'rag_lookup')
        self.assertEqual(json.loads(actual[0]['function']['arguments']), {'query':'hi'})
        actual = tool_call_from_schema(tools, 'record_id', 'TKT-0001')
        self.assertEqual(actual[0]['function']['name'], 'something_else')

    def test_composer_requires_real_tools(self):
        with self.assertRaises(RuntimeError):
            self.state.final_answer()

    def test_composer_uses_real_evidence(self):
        from agents.crew_phase3b import AgentAnswer
        self.state.policy = {'answer':'Policy text', 'grounded':True,
                             'sources':['POL-01'], 'top_similarity':0.8}
        self.state.lookup = {'found':False,'record_id':'','error':'No ID supplied'}
        self.state.tool_events = [{'tool':'policy'}, {'tool':'ticket'}]
        result = self.state.final_answer()
        self.assertIsInstance(result, AgentAnswer)
        self.assertEqual(result.answer, 'Policy text')

    def test_threshold_must_be_measured(self):
        from agents.crew_phase3b import measured_sentence_threshold
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'measured.json'
            with self.assertRaises(FileNotFoundError):
                measured_sentence_threshold(p)
            p.write_text(json.dumps({'strategies': {'sentence': {
                'clean_separation': True, 'threshold': 0.318917}}}))
            self.assertAlmostEqual(measured_sentence_threshold(p), 0.318917)

    def test_mock_does_not_imagine_observation(self):
        from agents.crew_phase3b import DeterministicCrewLLM
        llm = DeterministicCrewLLM('retrieval', self.state)
        schema = [{'function': {'name': 'rag_lookup', 'parameters': {
            'properties': {'query': {'type':'string'}}}}}]
        self.assertIsInstance(llm.call([], tools=schema), list)
        with self.assertRaises(RuntimeError):
            llm.call([], tools=schema)

if __name__ == '__main__':
    unittest.main()
