"""Empirical similarity calibration and document-level precision/recall report."""
import json
from pathlib import Path
from statistics import mean

from rag.core import ROOT, FALLBACK, grounded_answer, load_model, retrieve

TESTS = json.loads((ROOT / 'evaluation' / 'rag_queries.json').read_text(encoding='utf-8'))
OUT = ROOT / 'evaluation' / 'rag_results.json'


def precision_recall(predicted: list[str], expected: list[str]) -> dict:
    # A retrieved document is counted once even if multiple chunks match it.
    p, e = set(predicted), set(expected)
    tp = len(p & e)
    return {"tp": tp, "retrieved": len(p), "relevant": len(e),
            "precision": tp / len(p) if p else 0.0,
            "recall": tp / len(e) if e else 0.0}


def select_threshold(in_scores: list[float], out_scores: list[float]) -> float:
    """A valid separator must be learned from measured scores, never guessed."""
    lower = max(out_scores)
    upper = min(in_scores)
    if not lower < upper:
        raise ValueError(f'No clean separation: max out-of-scope={lower:.4f}, '
                         f'min in-scope={upper:.4f}. Revise test queries or '
                         'KB, rerun retrieval, and document the overlap. '
                         'Do NOT report an uncalibrated threshold.')
    return (lower + upper) / 2


def main():
    model = load_model()
    report = {"test_set": TESTS, "strategies": {}}
    for strategy in ('fixed', 'sentence'):
        in_rows, out_rows = [], []
        for name, bucket in [('in_scope', in_rows), ('out_of_scope', out_rows)]:
            for testcase in TESTS[name]:
                hits = retrieve(testcase['query'], model, strategy=strategy, top_k=3)
                # Document-level dedup for precision and recall.
                docs = list(dict.fromkeys(h.doc_id for h in hits))
                bucket.append({"query": testcase['query'],
                               "top1_similarity": round(hits[0].similarity, 6),
                               "returned_docs": docs,
                               "expected_docs": testcase['relevant_docs'],
                               **precision_recall(docs, testcase['relevant_docs'])})
        # Calibration uses >=3 in scope and >=2 deliberately out of scope.
        in_scores = [r['top1_similarity'] for r in in_rows]
        out_scores = [r['top1_similarity'] for r in out_rows]
        try:
            threshold = select_threshold(in_scores, out_scores)
            separation = True
        except ValueError as exc:
            threshold, separation = None, False
            print(f'{strategy} WARNING: {exc}')
        report['strategies'][strategy] = {
            "threshold": round(threshold, 6) if threshold is not None else None,
            "clean_separation": separation,
            "in_scope": in_rows, "out_of_scope": out_rows,
            "mean_document_precision_at_3": round(mean(r['precision'] for r in in_rows), 4),
            "mean_document_recall_at_3": round(mean(r['recall'] for r in in_rows), 4)}
        print(f'\n{strategy.upper()} — threshold: {threshold}')
        for row in in_rows:
            print(f"  {row['query']}: similarity={row['top1_similarity']:.4f}, "
                  f"P={row['tp']}/{row['retrieved']}={row['precision']:.3f}, "
                  f"R={row['tp']}/{row['relevant']}={row['recall']:.3f}")
        for row in out_rows:
            print(f"  OOS: {row['query']}: similarity={row['top1_similarity']:.4f}")
        if threshold is not None:
            for row in in_rows[:5] + out_rows[:1]:
                response = grounded_answer(row['query'], model, threshold, strategy=strategy)
                print('  DEMO:', row['query'], '=>', response['answer'][:110])
                if row in out_rows and response['answer'] != FALLBACK:
                    raise AssertionError('Out-of-scope fallback did not trigger')
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'\nSaved measured results to {OUT}')
    if not all(r['clean_separation'] for r in report['strategies'].values()):
        raise SystemExit('Calibration needs attention. No threshold guessed; see measurements.')
    ranked = sorted(report['strategies'].items(),
                    key=lambda item: (item[1]['mean_document_recall_at_3'],
                                      item[1]['mean_document_precision_at_3']), reverse=True)
    print(f'Candidate strategy: {ranked[0][0]} (compare measured results in README before choosing)')

if __name__ == '__main__':
    main()
