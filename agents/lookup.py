"""Synthetic ticket lookup and a transparent, bounded escalation score."""
from __future__ import annotations
from dataset import SUPPORT_TICKETS

_BY_ID = {ticket['record_id']: ticket for ticket in SUPPORT_TICKETS}


def escalation_score(ticket: dict) -> float:
    """Historical escalation flag contributes 70%; recency urgency contributes 30%.

    Recency is 1 for newly created tickets and 0 for 30-day-old tickets.
    This is a pedagogical risk score, not an Ola production severity model.
    """
    return round(0.7 * int(ticket['escalated']) +
                 0.3 * (1.0 - ticket['days_since_created'] / 30.0), 4)


def check_support_ticket_status(record_id: str) -> dict:
    """Return only the fields required by the capstone; no customer PII."""
    record = _BY_ID.get(record_id.strip().upper())
    if record is None:
        return {'record_id': record_id, 'found': False, 'error': 'Ticket not found'}
    return {
        'record_id': record['record_id'], 'found': True,
        'status': record['status'],
        'resolution_time_hours': record['resolution_time_hours'],
        'escalation_score': escalation_score(record),
        'recommend_escalation': escalation_score(record) >= 0.7,
    }
