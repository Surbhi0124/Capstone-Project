"""Allowlisted JSONL request audit records. Never log exception strings or raw payloads."""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from uuid import uuid4
from agents.guardrails import mask_fixed_pii

LOG_DIR = Path(__file__).resolve().parents[1] / 'logs'


def new_trace_id() -> str:
    return uuid4().hex


class AuditLogger:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else LOG_DIR / 'requests.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f'ola.audit.{id(self)}')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = logging.FileHandler(self.path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self._handler = handler

    def record(self, *, trace_id: str, endpoint: str, status: str,
               started: float, request_text: str = '', session_id: str = '',
               error_code: str = '') -> None:
        # Use only fixed names/codes. Don't log arbitrary error messages, headers,
        # ticket lookup contents, user paths, or free-form exception objects.
        entry = {
            'trace_id': trace_id,
            'endpoint': endpoint,
            'status': status,
            'duration_ms': round((time.perf_counter() - started) * 1000, 3),
            'request_masked': mask_fixed_pii(request_text)[:2000],
            'session_id': mask_fixed_pii(session_id)[:80],
            'error_code': error_code,
        }
        self.logger.info(json.dumps(entry, ensure_ascii=False))

    def close(self):
        self._handler.close()
        self.logger.removeHandler(self._handler)
