"""Deterministic Mock LLM response logic.

A policy answer is copied from retrieved evidence, never fabricated. The same
inputs yield byte-identical outputs. This is not yet a replacement for a full
CrewAI BaseLLM integration; that integration is a separate required milestone.
"""
from __future__ import annotations
from dataclasses import dataclass
from rag.core import FALLBACK


@dataclass(frozen=True)
class MockResponse:
    answer: str
    grounded: bool
    sources: tuple[str, ...]


def compose_policy_answer(hits: list, threshold: float) -> MockResponse:
    """Compose from the top retrieved chunk if measured similarity is enough."""
    if not hits or hits[0].similarity < threshold:
        return MockResponse(FALLBACK, False, ())
    return MockResponse(hits[0].text, True, (hits[0].doc_id,))


def compose_lookup_answer(lookup_result: dict) -> MockResponse:
    if not lookup_result['found']:
        return MockResponse('Ticket not found.', False, ())
    # Data-grounded, deterministic, and based only on permitted lookup fields.
    result = lookup_result
    answer = (f"Ticket {result['record_id']} is {result['status']}. "
              f"Recorded resolution time: {result['resolution_time_hours']} hours. "
              f"Escalation score: {result['escalation_score']:.4f}. "
              f"Escalation recommended: {'yes' if result['recommend_escalation'] else 'no'}.")
    return MockResponse(answer, True, (result['record_id'],))
