import json
import re
from pathlib import Path
import unittest

REQUIRED_TOPICS = {
    "ticket-priority classification rules", "SLA-by-severity policy", "escalation matrix",
    "refund/compensation policy", "customer-communication-channel policy",
    "business-hours/holiday-support policy", "repeat-complaint-handling policy",
    "service-credit policy", "feedback-collection process", "VIP-customer handling policy",
    "outage-communication protocol", "data-retention policy for tickets",
}


class PolicyTests(unittest.TestCase):
    def test_required_policies(self):
        path = Path(__file__).resolve().parents[1] / "knowledge_base" / "policies.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(data), 12)
        self.assertTrue(REQUIRED_TOPICS.issubset({d["topic"] for d in data}))
        self.assertEqual(len({d['doc_id'] for d in data}), len(data))
        for item in data:
            self.assertTrue(item['content'].strip())
            sentence_count = len([s for s in re.split(r'(?<=[.!?])\s+', item['content'].strip()) if s])
            self.assertGreaterEqual(sentence_count, 2)
            self.assertLessEqual(sentence_count, 5)


if __name__ == "__main__":
    unittest.main()
