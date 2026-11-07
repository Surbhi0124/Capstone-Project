"""Deterministic, synthetic support-ticket dataset for the IITR capstone.

Run: python dataset.py
All data are fictional; no genuine Ola customer data is used.
"""
from collections import Counter
from pathlib import Path
import json
import random

SEED = 260313
NUM_TICKETS = 50
CATEGORIES = (
    "Billing", "Technical Issue", "Account Access", "Product Defect", "General Inquiry"
)
STATUSES = ("Open", "In Progress", "Escalated", "Resolved", "Closed")
# The distribution intentionally includes all required values while keeping most
# routine tickets in the more common operational categories.
CATEGORY_WEIGHTS = (26, 25, 19, 15, 15)
STATUS_WEIGHTS = (24, 27, 10, 24, 15)
ESCALATION_PROBABILITY = 0.20
RESOLUTION_HOURS_RANGE = (1, 72)


def generate_tickets(seed: int = SEED, count: int = NUM_TICKETS) -> list[dict]:
    """Create deterministic records from the given PRNG seed, without manual edits."""
    if count < 40:
        raise ValueError("Capstone requires at least 40 tickets")
    rng = random.Random(seed)
    tickets = []
    for i in range(count):
        tickets.append({
            "record_id": f"TKT-{i + 1:04d}",
            "category": rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0],
            "status": rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0],
            "resolution_time_hours": rng.randint(*RESOLUTION_HOURS_RANGE),
            "days_since_created": rng.randint(0, 30),
            "escalated": rng.random() < ESCALATION_PROBABILITY,
        })
    return tickets


def validate_tickets(tickets: list[dict]) -> dict:
    """Fail loudly if any grading requirements are not satisfied."""
    assert len(tickets) >= 40, "Not enough tickets"
    assert len({r['record_id'] for r in tickets}) == len(tickets), "Duplicate IDs"
    categories = Counter(r["category"] for r in tickets)
    statuses = Counter(r["status"] for r in tickets)
    assert all(categories[c] >= 3 for c in CATEGORIES), f"Category coverage: {categories}"
    assert all(statuses[s] >= 1 for s in STATUSES), f"Status coverage: {statuses}"
    for ticket in tickets:
        assert set(ticket) == {"record_id", "category", "status", "resolution_time_hours", "days_since_created", "escalated"}
        assert isinstance(ticket["resolution_time_hours"], int)
        assert 1 <= ticket["resolution_time_hours"] <= 72
        assert type(ticket["days_since_created"]) is int and 0 <= ticket["days_since_created"] <= 30
        assert type(ticket["escalated"]) is bool
    escalated_count = sum(r["escalated"] for r in tickets)
    percentage = 100 * escalated_count / len(tickets)
    assert 10 <= percentage <= 30, f"Escalation rate out of range: {percentage:.1f}%"
    return {
        "seed": SEED,
        "total": len(tickets),
        "category_counts": {c: categories[c] for c in CATEGORIES},
        "status_counts": {s: statuses[s] for s in STATUSES},
        "escalated_count": escalated_count,
        "escalated_percent": round(percentage, 2),
    }


SUPPORT_TICKETS = generate_tickets()


if __name__ == "__main__":
    report = validate_tickets(SUPPORT_TICKETS)
    output = Path(__file__).resolve().parent / "data" / "support_tickets.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(SUPPORT_TICKETS, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Exported {len(SUPPORT_TICKETS)} synthetic tickets to {output}")
