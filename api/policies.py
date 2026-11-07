"""Append a fabricated policy and synchronize BOTH Chroma collections.

This endpoint is intended for local coursework, not an unauthenticated hosted service.
"""
from __future__ import annotations
import json
import re
import threading
from pathlib import Path
from typing import Callable

from rag.core import POLICIES, COLLECTIONS, chroma_client, load_model, make_chunks
from agents.guardrails import mask_fixed_pii

_ID = re.compile(r'^POL-[A-Z0-9-]{2,24}$')
_LOCK = threading.Lock()


def index_one_policy(document: dict) -> None:
    """Embed and upsert chunks; compensate partial indexes on failure."""
    model = load_model()
    client = chroma_client()
    prepared = {}
    for strategy, collection_name in COLLECTIONS.items():
        chunks = make_chunks([document], strategy)
        prepared[collection_name] = (chunks, model.encode(
            [chunk['text'] for chunk in chunks], normalize_embeddings=True
        ).tolist())
    written: list[tuple[str, list[str]]] = []
    try:
        for collection_name, (chunks, vectors) in prepared.items():
            collection = client.get_collection(collection_name)  # Never silently build a new index
            ids = [chunk['id'] for chunk in chunks]
            collection.upsert(ids=ids, embeddings=vectors,
                              documents=[chunk['text'] for chunk in chunks],
                              metadatas=[{'doc_id': c['doc_id'], 'title': c['title']}
                                         for c in chunks])
            written.append((collection_name, ids))
    except Exception:
        for collection_name, ids in written:
            try:
                client.get_collection(collection_name).delete(ids=ids)
            except Exception:
                pass  # Requires index repair if cleanup itself fails
        raise


def add_policy(document: dict, *, path: Path = POLICIES,
               indexer: Callable[[dict], None] = index_one_policy) -> dict:
    doc_id = document['doc_id'].strip().upper()
    title = document['title'].strip()
    content = mask_fixed_pii(document['content'].strip())
    if not _ID.fullmatch(doc_id):
        raise ValueError('Policy ID must match POL- followed by 2-24 uppercase letters, numbers or hyphens')
    if len(title) < 5 or len(title) > 120:
        raise ValueError('Policy title must contain 5-120 characters')
    if len(content) < 40 or len(content) > 2000:
        raise ValueError('Policy content must contain 40-2000 characters')
    if len(re.split(r'(?<=[.!?])\s+', content)) not in {2, 3, 4, 5}:
        raise ValueError('Policy content must have 2-5 sentences')
    if re.search(r'\b(ignore (previous|all) instructions|reveal (the )?system prompt)\b',
                 content, re.I):
        raise ValueError('Policy contains disallowed instructions')
    clean = {'doc_id': doc_id, 'title': title, 'content': content}
    # A process-local lock is sufficient for the course's single-process server.
    with _LOCK:
        previous = json.loads(path.read_text(encoding='utf-8'))
        if any(p['doc_id'] == doc_id for p in previous):
            raise FileExistsError('Policy ID already exists')
        # Update index first; do not save a policy that cannot be searched.
        indexer(clean)
        try:
            tmp = path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(previous + [clean], indent=2, ensure_ascii=False) + '\n',
                           encoding='utf-8')
            tmp.replace(path)
        except Exception:
            # If the file cannot be written, repair indexes via rebuilding from KB.
            # The server fails closed and operator is informed through internal logs.
            raise RuntimeError('Could not save policy after indexing; rebuild indexes before retrying')
    return {'doc_id': doc_id, 'indexed_strategies': list(COLLECTIONS)}
