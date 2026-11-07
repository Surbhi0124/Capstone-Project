"""Runtime budget, conservative cache and tool-access checks."""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

from agents.crew_phase3b import AgentAnswer, RunState, build_crew


class BudgetExceeded(ValueError):
    pass


@dataclass(frozen=True)
class BudgetPolicy:
    max_input_chars: int = 2000
    max_estimated_tokens: int = 3200
    max_simulated_cost_usd: float = 0.02
    simulated_usd_per_1000_tokens: float = 0.002
    # Include a deliberately conservative fixed reserve for multi-agent prompts.
    prompt_overhead_tokens: int = 1200

    def enforce(self, question: str) -> dict:
        # Actual model has no tokenizer. These are documented estimates, not bills.
        estimated_tokens = (len(question) + 3) // 4 + self.prompt_overhead_tokens
        estimated_cost = estimated_tokens * self.simulated_usd_per_1000_tokens / 1000
        if (len(question) > self.max_input_chars or
                estimated_tokens > self.max_estimated_tokens or
                estimated_cost > self.max_simulated_cost_usd):
            raise BudgetExceeded('Request exceeds configured input/token/cost budget')
        return {'estimated_tokens': estimated_tokens,
                'simulated_cost_usd': round(estimated_cost, 8),
                'real_api_cost_usd': 0.0}


def assert_least_autonomy(state: RunState) -> None:
    """Fail if a crew design permits the ticket tool outside the Lookup Agent."""
    crew = build_crew(state)
    for agent in crew.agents:
        allowed = agent.role == 'Support Ticket Lookup Agent'
        for tool in agent.tools or []:
            if tool.name == 'check_support_ticket_status' and not allowed:
                raise PermissionError(f'Unauthorized lookup tool on {agent.role}')
    actual = [(a.role, t.name) for a in crew.agents for t in (a.tools or [])
              if t.name == 'check_support_ticket_status']
    if actual != [('Support Ticket Lookup Agent', 'check_support_ticket_status')]:
        raise PermissionError('Ticket lookup tool must be wired only to the Lookup Agent')


def normalized_query(question: str) -> str:
    return re.sub(r'\s+', ' ', question.strip().casefold())


class PolicyAnswerCache:
    """In-process bounded cache, policy-only (no per-user ticket answers)."""
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self._entries: OrderedDict[str, tuple[AgentAnswer, RunState]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.crew_calls = 0

    def execute(self, question: str, runner: Callable) -> tuple[AgentAnswer, RunState]:
        # Never cache ticket-specific content, even with a common question text.
        from agents.crew_phase3b import TICKET_RE
        if TICKET_RE.search(question):
            self.crew_calls += 1
            return runner(question)
        key = normalized_query(question)
        if key in self._entries:
            self.hits += 1
            self._entries.move_to_end(key)
            return self._entries[key]
        self.misses += 1
        self.crew_calls += 1
        answer, state = runner(question)
        # Only valid grounded policies; not refusals, sensitive tickets, or PII.
        if answer.route == 'policy' and answer.grounded and answer.sources:
            self._entries[key] = (answer, state)
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
        return answer, state

    def clear(self) -> None:
        self._entries.clear()


def main():
    from agents.crew_phase3b import RunState
    state = RunState(question='test governance', route='policy',
                     ticket_id='', threshold=0.318916)
    assert_least_autonomy(state)
    print('Least-autonomy check: PASS — only Lookup Agent holds the ticket tool')
    budget = BudgetPolicy()
    print('Normal request budget:', budget.enforce('What is the SLA?'))
    try:
        budget.enforce('oversized ' * 500)
    except BudgetExceeded as exc:
        print('Oversized request rejected:', str(exc))
    else:
        raise AssertionError('Oversized request unexpectedly accepted')


if __name__ == '__main__':
    main()
