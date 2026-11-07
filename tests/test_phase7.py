"""Phase 7 regression checks: ranking and raw cosine are separate."""
import unittest
from rag.core import Hit, rerank_hits


class Phase7Tests(unittest.TestCase):
    def test_e03_original_question_prefers_escalation(self):
        q = "Who receives an unresolved high priority support issue?"
        hits = [
            Hit('POL-10', 'Priority Customer Handling',
                'Priority customers receive a dedicated queue.', 0.6092),
            Hit('POL-03', 'Ticket Escalation Matrix',
                'Unresolved High-priority issues are escalated to a team lead.', 0.5693),
            Hit('POL-01', 'Ticket Priority Classification',
                'Priority is based on impact.', 0.5313),
        ]
        reranked = rerank_hits(q, hits)
        self.assertEqual(reranked[0].doc_id, 'POL-03')
        self.assertEqual(reranked[0].similarity, 0.5693)
        self.assertEqual(hits[0].doc_id, 'POL-10')

    def test_priority_customer_question_keeps_semantic_order(self):
        hits = [Hit('POL-10', 'Priority Customer Handling', 'Dedicated queue', .67),
                Hit('POL-03', 'Ticket Escalation Matrix', 'Escalation', .51)]
        self.assertEqual(rerank_hits('What benefits do priority customers receive?', hits), hits)

    def test_rerank_has_no_effect_on_unrelated_query(self):
        hits = [Hit('POL-02', 'Response and Resolution SLA', '15 minutes', .8),
                Hit('POL-03', 'Ticket Escalation Matrix', 'escalated to lead', .3)]
        self.assertEqual(rerank_hits('What is the critical ticket SLA?', hits), hits)


if __name__ == '__main__':
    unittest.main()
