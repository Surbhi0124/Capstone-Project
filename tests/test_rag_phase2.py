import unittest

from evaluation.run_rag import precision_recall, select_threshold
from rag.core import fixed_chunks, load_policies, make_chunks, sentence_chunks


class Phase2UnitTests(unittest.TestCase):
    def test_two_methods_maintain_parent_ids(self):
        policies = load_policies()
        for method in ('fixed', 'sentence'):
            chunks = make_chunks(policies, method)
            self.assertGreaterEqual(len(chunks), 12)
            self.assertEqual({c['doc_id'] for c in chunks}, {p['doc_id'] for p in policies})
            self.assertEqual(len({c['id'] for c in chunks}), len(chunks))

    def test_character_overlap(self):
        chunks = fixed_chunks('0123456789', size=6, overlap=2)
        self.assertEqual(chunks, ['012345', '456789'])

    def test_sentence_boundaries(self):
        self.assertEqual(sentence_chunks('First. Second! Third?', 2),
                         ['First. Second!', 'Third?'])

    def test_document_level_precision_recall(self):
        # Duplicate chunks from same parent must not count twice.
        self.assertEqual(precision_recall(['A', 'A', 'B'], ['A'])['precision'], 0.5)
        self.assertEqual(precision_recall(['A', 'A', 'B'], ['A'])['recall'], 1.0)

    def test_threshold_empirical_not_preset(self):
        self.assertAlmostEqual(select_threshold([0.80, 0.78, 0.75], [0.22, 0.30]), 0.525)
        with self.assertRaises(ValueError):
            select_threshold([0.3, 0.4, 0.5], [0.6, 0.1])


if __name__ == '__main__':
    unittest.main()
