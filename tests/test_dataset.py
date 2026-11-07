"""Run with: python -m unittest discover -s tests -v"""
import unittest
from dataset import SUPPORT_TICKETS, generate_tickets, validate_tickets


class DatasetTests(unittest.TestCase):
    def test_acceptance_criteria(self):
        self.assertEqual(validate_tickets(SUPPORT_TICKETS)["total"], 50)

    def test_deterministic_generation(self):
        self.assertEqual(generate_tickets(), generate_tickets())

    def test_different_seed_changes_records(self):
        self.assertNotEqual(generate_tickets(seed=1), generate_tickets(seed=2))

    def test_unique_ids(self):
        self.assertEqual(len({t["record_id"] for t in SUPPORT_TICKETS}), len(SUPPORT_TICKETS))


if __name__ == "__main__":
    unittest.main()
