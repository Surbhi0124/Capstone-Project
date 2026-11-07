"""Minimal fixed-format input checks for the offline mock path.

    This cannot identify all free-text PII, and must not be treated as a full
    security system. Only fabricated information belongs in this coursework.
"""
import re

# Indian phone format: stand-alone 10 digits beginning 6-9, optional +91/91 prefix.
_PHONE = re.compile(r'(?<!\w)(?:(?:\+91|91)[\s-]?)?[6-9]\d{9}(?!\w)')
_INJECTION = re.compile(r'\b(?:ignore (?:all )?(?:previous|prior) instructions|'
                        r'reveal (?:your |the )?system prompt|bypass (?:the )?guardrails)\b', re.I)


def mask_fixed_pii(text: str) -> str:
    return _PHONE.sub('[PHONE_MASKED]', text)


def screen_input(query: str) -> dict:
    sanitized = mask_fixed_pii(query)
    if _INJECTION.search(query):
        return {'accepted': False, 'query': sanitized,
                'error': 'Potential prompt injection detected'}
    return {'accepted': True, 'query': sanitized, 'error': None}
