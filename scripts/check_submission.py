"""Static submission preflight: does not call the network or import CrewAI.

Run at project root: python -m scripts.check_submission
Use --strict only after generating evaluated artifacts on this machine.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    'dataset.py','knowledge_base/policies.json','data/support_tickets.json',
    'rag/core.py','rag/build_index.py','agents/crew_phase3b.py',
    'agents/session_phase4.py','agents/review_phase6.py',
    'agents/governance_phase6.py','agents/pipeline_phase6.py',
    'api/app.py','evaluation/capstone_cases.json','evaluation/judge_phase6.py',
    'README.md','docs/RUBRIC_AUDIT.md'
]
EVIDENCE = ['evaluation/rag_results.json', 'evaluation/phase6_results.json',
            'evaluation/phase6_results.csv']
FORBIDDEN = ['.venv', 'models', 'data/chroma_index', 'logs', '.env']


def audit(strict: bool = False):
    failures, warnings = [], []
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            failures.append('Missing required file: ' + name)
    for name in EVIDENCE:
        if not (ROOT / name).is_file():
            (failures if strict else warnings).append('Generate local measured evidence: ' + name)
    try:
        policies = json.loads((ROOT / 'knowledge_base/policies.json').read_text(encoding='utf-8'))
        tickets = json.loads((ROOT / 'data/support_tickets.json').read_text(encoding='utf-8'))
        cases = json.loads((ROOT / 'evaluation/capstone_cases.json').read_text(encoding='utf-8'))
        if len(policies) < 12 or len({p['doc_id'] for p in policies}) != len(policies):
            failures.append('Knowledge base: fewer than 12 unique documents')
        if len(tickets) < 40:
            failures.append('Dataset: fewer than 40 tickets')
        categories = Counter(t['category'] for t in tickets)
        statuses = Counter(t['status'] for t in tickets)
        for category in ['Billing','Technical Issue','Account Access','Product Defect','General Inquiry']:
            if categories[category] < 3:
                failures.append('Insufficient category: ' + category)
        for status in ['Open','In Progress','Escalated','Resolved','Closed']:
            if statuses[status] < 1:
                failures.append('Missing status: ' + status)
        percent = 100 * sum(t['escalated'] for t in tickets) / len(tickets)
        if not 10 <= percent <= 30:
            failures.append('Escalated percentage outside 10–30%')
        if len(cases) != 15:
            failures.append('Evaluation needs exactly 15 cases')
        if len({c['expected_doc'] for c in cases if c['in_scope']}) < 12:
            failures.append('Evaluation missing a policy topic')
    except Exception as exc:
        failures.append('Cannot inspect JSON inputs: ' + str(exc))
    if (ROOT / 'evaluation/phase6_results.json').is_file():
        try:
            result = json.loads((ROOT / 'evaluation/phase6_results.json').read_text(encoding='utf-8'))
            if len(result.get('cases', [])) != 15 or len(result.get('averages', {})) != 4:
                failures.append('Evaluation report is incomplete')
        except Exception as exc:
            failures.append('Cannot inspect evaluation report: ' + str(exc))
    # These are permitted locally but must not be in Git; do not delete them.
    for name in FORBIDDEN:
        if (ROOT / name).exists():
            warnings.append(f'LOCAL ONLY, verify Git ignores: {name}')
    print('SUBMISSION PREFLIGHT')
    for item in failures: print('FAIL:', item)
    for item in warnings: print('INFO:', item)
    if not failures: print('PASS: static checks passed' + (' (strict evidence)' if strict else ''))
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.strict) else 1)
