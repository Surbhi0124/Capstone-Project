"""Phase 3B: real CrewAI agents and native tool calls, using a deterministic BaseLLM.

No hosted models, API keys, or telemetry. Requires pre-cached local embedding
model and the sentence Chroma collection built in Phase 2.
"""
from __future__ import annotations

# Must be set *before* importing crewai. Also disable local optional tracing.
import os
os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true'
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['CREWAI_TRACING_ENABLED'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

from pydantic import BaseModel, Field, ConfigDict
from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool

from agents.guardrails import screen_input
from agents.lookup import check_support_ticket_status
from agents.mock_engine import compose_lookup_answer, compose_policy_answer
from rag.core import FALLBACK, ROOT, load_model, retrieve, rerank_hits

TICKET_RE = re.compile(r'\bTKT-\d{4,}\b', re.I)
RESULTS_FILE = ROOT / 'evaluation' / 'rag_results.json'


def measured_sentence_threshold(path: Path = RESULTS_FILE) -> float:
    """Fail closed if student's actual Phase 2 calibration isn't present."""
    if not path.is_file():
        raise FileNotFoundError('No calibrated threshold. Run python -m evaluation.run_rag first.')
    report = json.loads(path.read_text(encoding='utf-8'))
    row = report['strategies']['sentence']
    if not row['clean_separation'] or row['threshold'] is None:
        raise ValueError('Sentence retrieval threshold was not successfully calibrated')
    return float(row['threshold'])


class PolicyQuery(BaseModel):
    query: str = Field(description='Customer support policy question to search')


class TicketQuery(BaseModel):
    record_id: str = Field(description='Synthetic ticket identifier such as TKT-0001')


class AgentAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    answer: str
    grounded: bool
    sources: list[str]
    route: str
    ticket_id: Optional[str] = None
    top_similarity: Optional[float] = None


@dataclass
class RunState:
    question: str
    route: str
    ticket_id: str
    threshold: float
    policy: dict[str, Any] | None = None
    lookup: dict[str, Any] | None = None
    tool_events: list[dict[str, Any]] = field(default_factory=list)

    def final_answer(self) -> AgentAnswer:
        """Compose ONLY from completed tool results; no synthetic evidence."""
        if not any(e['tool'] == 'policy' for e in self.tool_events):
            raise RuntimeError('CrewAI did not execute policy retrieval tool')
        if not any(e['tool'] == 'ticket' for e in self.tool_events):
            raise RuntimeError('CrewAI did not execute ticket lookup tool')
        if self.route == 'ticket':
            if not self.lookup:
                raise RuntimeError('Ticket lookup did not return evidence')
            response = compose_lookup_answer(self.lookup)
            return AgentAnswer(answer=response.answer, grounded=response.grounded,
                               sources=list(response.sources), route=self.route,
                               ticket_id=self.ticket_id or None)
        if not self.policy:
            raise RuntimeError('Policy retrieval did not return evidence')
        return AgentAnswer(answer=self.policy['answer'], grounded=self.policy['grounded'],
                           sources=self.policy['sources'], route=self.route,
                           top_similarity=self.policy['top_similarity'])


class PolicyRetrievalTool(BaseTool):
    name: str = 'rag_lookup'
    description: str = 'Retrieve grounded policy evidence from the local sentence-based Chroma collection.'
    args_schema: Type[BaseModel] = PolicyQuery
    state: Any = Field(exclude=True)
    model: Any = Field(default=None, exclude=True)

    def _run(self, query: str) -> str:
        if self.model is None:
            self.model = load_model()
        hits = retrieve(query, self.model, strategy='sentence', top_k=3)
        hits = rerank_hits(query, hits)
        result = compose_policy_answer(hits, self.state.threshold)
        record = {
            'answer': result.answer,
            'grounded': result.grounded,
            'sources': list(result.sources),
            'top_similarity': round(hits[0].similarity, 6) if hits else None,
        }
        self.state.policy = record
        self.state.tool_events.append({'tool': 'policy', 'query': query})
        return json.dumps(record, sort_keys=True)


class TicketLookupTool(BaseTool):
    name: str = 'check_support_ticket_status'
    description: str = 'Find one fabricated support ticket by ID and compute its escalation score.'
    args_schema: Type[BaseModel] = TicketQuery
    state: Any = Field(exclude=True)

    def _run(self, record_id: str) -> str:
        # Never substitute an unrelated ticket when no ID is supplied.
        result = (check_support_ticket_status(record_id) if record_id else
                  {'record_id': '', 'found': False, 'error': 'No ticket ID supplied'})
        self.state.lookup = result
        self.state.tool_events.append({'tool': 'ticket', 'record_id': record_id})
        return json.dumps(result, sort_keys=True)


def tool_call_from_schema(tools: list[dict], arg_name: str, arg_value: str) -> list[dict]:
    """Choose by declared parameters, never by a substring of the tool name."""
    for item in tools:
        func = item.get('function', {})
        params = func.get('parameters', {})
        properties = params.get('properties', {})
        if arg_name in properties:
            return [{
                'id': f'call_{arg_name}_1', 'type': 'function',
                'function': {'name': func['name'],
                             'arguments': json.dumps({arg_name: arg_value})},
            }]
    raise RuntimeError(f'Expected CrewAI tool schema argument {arg_name!r}, got {tools!r}')


class DeterministicCrewLLM(BaseLLM):
    """CrewAI-native offline LLM with the three fixed roles used in the capstone.

    The runner constructs a NEW instance per role and invocation, so repeated
    queries never leak prior tool state. No keyword search of the CrewAI built-in
    ReAct template or 'Observation:' extraction is performed.
    """
    def __init__(self, role_mode: str, state: RunState):
        super().__init__(model='mock/offline-deterministic', temperature=0)
        if role_mode not in {'retrieval', 'lookup', 'composer'}:
            raise ValueError('Unknown role')
        self.role_mode = role_mode
        self.state = state
        self._requested_tool = False

    def supports_function_calling(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192

    def call(self, messages: Any, tools: Any = None, callbacks: Any = None,
             available_functions: Any = None, **kwargs: Any) -> Any:
        # CrewAI chooses between a native tool invocation and a text response.
        # Tool selection uses the schema, not the tool name or prompt wording.
        if self.role_mode == 'composer':
            return self.state.final_answer().model_dump_json()
        if not self._requested_tool:
            if not tools:
                raise RuntimeError('CrewAI did not expose expected tools to Mock LLM')
            self._requested_tool = True
            if self.role_mode == 'retrieval':
                return tool_call_from_schema(tools, 'query', self.state.question)
            return tool_call_from_schema(tools, 'record_id', self.state.ticket_id)
        # Do not fabricate a tool observation. Require the corresponding _run event.
        kind = 'policy' if self.role_mode == 'retrieval' else 'ticket'
        if not any(e['tool'] == kind for e in self.state.tool_events):
            raise RuntimeError(f'CrewAI did not execute the {kind} tool after requesting it')
        return f'{kind} tool completed successfully. Proceed with the recorded evidence.'


def build_crew(state: RunState) -> Crew:
    retrieval = Agent(
        role='Support Policy Retrieval Agent',
        goal='Retrieve the exact support policy relevant to the request.',
        backstory='You use a local, read-only policy knowledge base.',
        tools=[PolicyRetrievalTool(state=state)],
        llm=DeterministicCrewLLM('retrieval', state),
        allow_delegation=False, verbose=False, max_iter=3,
    )
    lookup = Agent(
        role='Support Ticket Lookup Agent',
        goal='Look up a synthetic ticket record and its escalation score.',
        backstory='You only consult the generated support-ticket dataset.',
        tools=[TicketLookupTool(state=state)],
        llm=DeterministicCrewLLM('lookup', state),
        allow_delegation=False, verbose=False, max_iter=3,
    )
    composer = Agent(
        role='Evidence-Grounded Response Composer',
        goal='Return one answer justified by actual tool results.',
        backstory='You cannot invent ticket records or policy evidence.',
        llm=DeterministicCrewLLM('composer', state),
        allow_delegation=False, verbose=False, max_iter=2,
    )
    tasks = [
        Task(description='Retrieve policy evidence for: {question}',
             expected_output='Policy tool completed, evidence recorded.', agent=retrieval),
        Task(description='Lookup synthetic ticket ID {ticket_id} (empty means no ID supplied).',
             expected_output='Ticket lookup tool completed, evidence recorded.', agent=lookup),
        Task(description='Compose a structured JSON response for: {question}. Use the '
                         'recorded tool evidence only, and use the appropriate route.',
             expected_output='JSON with answer, grounded, sources, route, ticket_id, top_similarity.',
             agent=composer, context=[]),
    ]
    return Crew(agents=[retrieval, lookup, composer], tasks=tasks,
                process=Process.sequential, verbose=False, memory=False,
                cache=False, planning=False)


def run_crew(question: str) -> tuple[AgentAnswer, RunState]:
    screened = screen_input(question)
    if not screened['accepted']:
        raise ValueError(screened['error'])
    cleaned = screened['query'].strip()
    if not cleaned:
        raise ValueError('Question cannot be empty')
    m = TICKET_RE.search(cleaned)
    record_id = m.group(0).upper() if m else ''
    route = 'ticket' if record_id else 'policy'
    state = RunState(question=cleaned, route=route, ticket_id=record_id,
                     threshold=measured_sentence_threshold())
    crew = build_crew(state)
    result = crew.kickoff(inputs={'question': cleaned, 'ticket_id': record_id})
    # Pydantic independently validates the actual final CrewAI output.
    response = AgentAnswer.model_validate_json(result.raw)
    if response != state.final_answer():
        raise RuntimeError('Crew output mismatched the recorded tool evidence')
    return response, state


def main():
    for q in ('What is the first response time for a critical ticket?',
              'What is the status of TKT-0001?',
              'How many moons orbit Neptune?'):
        answer, state = run_crew(q)
        print('\nQuestion:', q)
        print('Answer:', answer.model_dump_json(indent=2))
        print('Tool invocations:', json.dumps(state.tool_events))


if __name__ == '__main__':
    main()
