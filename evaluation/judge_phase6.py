"""Task 13: deterministic MOCK_LLM-as-judge, 15 specified cases, four scores.

Scores are discrete rule checks, not actual semantic LLM evaluations. Explicit
rubric and zero-API reproducibility matter more than claiming a perfect judge.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from statistics import mean

from agents.guardrails import mask_fixed_pii
from rag.core import FALLBACK, ROOT

CASES = ROOT / 'evaluation' / 'capstone_cases.json'
OUT = ROOT / 'evaluation' / 'phase6_results.json'
CSV = ROOT / 'evaluation' / 'phase6_results.csv'

JUDGE_PROMPT = '''You are an evidence-grounded support-agent judge. Evaluate ONLY
against the provided expected policy ID, required phrases, source citations and
fallback behavior. Return numeric scores: Accuracy, Grounding, Completeness,
Safety in [0,1]. This capstone's MOCK_LLM implements this rubric as deterministic
Python rather than relying on an external model. No claims about medical, legal,
or real Ola policy are made.'''


def judge_case(case: dict, answer) -> dict:
    """Four explicit observable dimensions, no semantic inference."""
    valid = bool(answer.answer.strip())
    safe = mask_fixed_pii(answer.answer) == answer.answer
    if case['in_scope']:
        correct_source = case['expected_doc'] in answer.sources
        accuracy = float(answer.grounded and correct_source)
        grounding = float(answer.grounded and correct_source and valid)
        completeness = (sum(p.casefold() in answer.answer.casefold() for p in case['required_phrases'])
                        / max(len(case['required_phrases']), 1))
        # Completeness only earns credit if expected evidence is sourced.
        completeness *= float(correct_source)
    else:
        expected_fallback = answer.answer == FALLBACK and not answer.grounded and not answer.sources
        accuracy = float(expected_fallback)
        grounding = float(expected_fallback)
        completeness = float(expected_fallback)
    return {'Accuracy': accuracy, 'Grounding': grounding,
            'Completeness': completeness, 'Safety': float(safe)}


def evaluate(runner, cases=None) -> dict:
    cases = cases if cases is not None else json.loads(CASES.read_text(encoding='utf-8'))
    assert len(cases) == 15 and len({c['expected_doc'] for c in cases if c['in_scope']}) == 12
    rows = []
    for case in cases:
        start = time.perf_counter()
        try:
            answer = runner(case['query'])
            result = judge_case(case, answer)
            response_text = answer.answer
            sources = answer.sources
            error = None
        except Exception as exc:
            # Fail closed: crashes receive zero rather than disappearing from average.
            result = {name: 0.0 for name in ('Accuracy','Grounding','Completeness','Safety')}
            response_text, sources, error = '', [], type(exc).__name__
        rows.append({'id': case['id'], 'query': case['query'],
                     'expected_doc': case['expected_doc'],
                     'answer': response_text, 'sources': sources,
                     'error': error, 'duration_ms': round((time.perf_counter()-start)*1000, 2),
                     **result})
    metrics = {name: round(mean(row[name] for row in rows), 4)
               for name in ('Accuracy','Grounding','Completeness','Safety')}
    return {'mode': 'MOCK_LLM', 'real_api_cost_usd': 0,
            'judge_type': 'deterministic_rule_based',
            'judge_prompt': JUDGE_PROMPT, 'cases': rows, 'averages': metrics}


def save(report):
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    with CSV.open('w', newline='', encoding='utf-8') as stream:
        fields = ['id','query','expected_doc','sources','Accuracy','Grounding','Completeness','Safety','duration_ms','error']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in report['cases']:
            writer.writerow({**row, 'sources': json.dumps(row['sources'])})


def main():
    # Each query uses a separate session to isolate memory. The review is live.
    from agents.pipeline_phase6 import Phase6Chat
    chat = Phase6Chat()
    cases = json.loads(CASES.read_text(encoding='utf-8'))
    ids = iter(c['id'] for c in cases)
    report = evaluate(lambda q: chat.chat('eval-'+next(ids), q), cases)
    save(report)
    for row in report['cases']:
        print(f"{row['id']} {row['expected_doc'] or 'OOS'}: "
              f"A={row['Accuracy']:.2f} G={row['Grounding']:.2f} "
              f"C={row['Completeness']:.2f} S={row['Safety']:.2f} "
              f"({row['duration_ms']:.0f} ms)")
    print('MEANS:', report['averages'])
    print('Saved:', OUT, CSV)


if __name__ == '__main__':
    main()
