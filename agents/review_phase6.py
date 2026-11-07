"""Independent two-agent AutoGen review, strictly offline.

The MOCK_LLM review decision is an evidence comparison performed BEFORE
preparing ReplayChatCompletionClient's deterministic responses. AutoGen then
runs the reviewer and editor as real AssistantAgent participants, exactly two
turns. This is deliberately NOT a semantic LLM judge.
"""
from __future__ import annotations

import json
from pydantic import BaseModel, ConfigDict

from agents.crew_phase3b import AgentAnswer, RunState
from agents.guardrails import mask_fixed_pii
from rag.core import FALLBACK


class ReviewVerdict(BaseModel):
    model_config = ConfigDict(extra='forbid')
    approved: bool
    final_answer: str
    reason: str


def deterministic_verdict(draft: str, state: RunState) -> ReviewVerdict:
    """Check the proposed text against recorded tool evidence, never the draft alone."""
    canonical = state.final_answer()
    evidence = canonical.answer
    if not evidence:
        raise ValueError('Missing original evidence')
    if mask_fixed_pii(draft) != draft:
        return ReviewVerdict(approved=False, final_answer=evidence,
                             reason='Draft exposed a fixed-format phone number; restored evidence-only answer')
    if draft == evidence:
        return ReviewVerdict(approved=True, final_answer=draft,
                             reason='Draft matches the source-backed CrewAI evidence exactly')
    return ReviewVerdict(approved=False, final_answer=evidence,
                         reason='Draft differs from the retrieved/lookup evidence; replaced with supported response')


async def review_draft(draft: str, state: RunState) -> tuple[ReviewVerdict, dict]:
    """Run a real AutoGen RoundRobinGroupChat with no hosted model or network.

    Each call constructs fresh agents/client to avoid cross-request state leakage.
    """
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import StructuredMessage
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_ext.models.replay import ReplayChatCompletionClient
    from autogen_core.models import ModelFamily

    expected = deterministic_verdict(draft, state)
    canonical = state.final_answer()
    # Distinct clients means the two participants each receive exactly one
    # deterministic completion. Editor uses structured JSON output via Pydantic.
    model_info = {'vision': False, 'function_calling': False, 'json_output': True,
                  'family': ModelFamily.UNKNOWN, 'structured_output': True}
    reviewer_client = ReplayChatCompletionClient(
        [f"Evidence reviewed. {expected.reason}"], model_info=model_info)
    editor_client = ReplayChatCompletionClient(
        [expected.model_dump_json()], model_info=model_info)
    reviewer = AssistantAgent(
        'Policy_Compliance_Reviewer', model_client=reviewer_client,
        system_message='Review draft against supplied original tool evidence only. No new facts.')
    editor = AssistantAgent(
        'Final_Editor', model_client=editor_client,
        system_message='Approve unchanged or replace unsupported draft with original evidence.',
        output_content_type=ReviewVerdict)
    team = RoundRobinGroupChat(
        [reviewer, editor], max_turns=2,
        custom_message_types=[StructuredMessage[ReviewVerdict]])
    original_context = {
        'question': state.question, 'route': state.route,
        'original_policy_tool_record': state.policy,
        'original_lookup_tool_record': state.lookup,
        'canonical_crew_output': canonical.model_dump(),
        'draft_to_review': draft,
    }
    result = await team.run(task=json.dumps(original_context, sort_keys=True))
    messages = [m for m in result.messages if getattr(m, 'source', None) in
                {'Policy_Compliance_Reviewer', 'Final_Editor'}]
    if len(messages) != 2 or messages[0].source != 'Policy_Compliance_Reviewer' or messages[1].source != 'Final_Editor':
        raise RuntimeError('AutoGen did not execute the required 2-agent sequence')
    final = messages[-1]
    if not isinstance(final, StructuredMessage):
        raise RuntimeError('AutoGen Final Editor did not return a structured message')
    verdict = ReviewVerdict.model_validate(final.content)
    if verdict != expected or verdict.final_answer != canonical.answer:
        raise ValueError('AutoGen verdict failed independent deterministic evidence validation')
    transcript = {'participants': [m.source for m in messages],
                  'max_turns': 2, 'draft': draft,
                  'verdict': verdict.model_dump(),
                  'original_evidence_sources': canonical.sources,
                  'offline': True}
    return verdict, transcript


def apply_verdict(answer: AgentAnswer, verdict: ReviewVerdict) -> AgentAnswer:
    """Return the reviewed answer with schema retained; never change citations."""
    return answer.model_copy(update={'answer': verdict.final_answer})


async def main() -> None:
    import asyncio
    from agents.crew_phase3b import run_crew
    answer, state = await asyncio.to_thread(run_crew, 'What is the status of TKT-0001?')
    for label, draft in [('approve', answer.answer),
                         ('revise', answer.answer + ' Guaranteed cash compensation for everyone.')]:
        verdict, transcript = await review_draft(draft, state)
        print(json.dumps({'case': label, **transcript}, indent=2))
        assert verdict.approved == (label == 'approve')


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
