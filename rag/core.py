"""Phase 2: reproducible local retrieval for the fictional Ola support policies.

The index is built by a separate command; regular queries never modify the KB.
No paid LLM and no generative API is used in this phase.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "knowledge_base" / "policies.json"
INDEX_DIR = ROOT / "data" / "chroma_index"
MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTIONS = {"fixed": "support_fixed_overlap", "sentence": "support_sentence"}
FALLBACK = "I don't know based on the available support policies."


def load_policies(path: Path = POLICIES) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) >= 12
    assert len({d["doc_id"] for d in data}) == len(data)
    return data


def fixed_chunks(text: str, size: int = 300, overlap: int = 70) -> list[str]:
    """Character chunks, with overlap; positions measured on original text."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")
    if not text:
        return []
    chunks = []
    for start in range(0, len(text), size - overlap):
        part = text[start:start + size].strip()
        if part:
            chunks.append(part)
        if start + size >= len(text):
            break
    return chunks


def sentence_chunks(text: str, sentences_per_chunk: int = 2) -> list[str]:
    """Non-overlapping sentence groups; the policies use plain English sentences."""
    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be positive")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return [" ".join(sentences[i:i + sentences_per_chunk])
            for i in range(0, len(sentences), sentences_per_chunk)]


def make_chunks(policies: Iterable[dict], strategy: str) -> list[dict]:
    if strategy not in COLLECTIONS:
        raise ValueError("Unknown chunk strategy")
    chunks = []
    for doc in policies:
        # Add the document title to each chunk to aid retrieval, preserve parent ID.
        parts = fixed_chunks(doc["content"]) if strategy == "fixed" else sentence_chunks(doc["content"])
        for idx, part in enumerate(parts):
            chunks.append({"id": f'{doc["doc_id"]}:{strategy}:{idx:03d}',
                           "doc_id": doc["doc_id"], "title": doc["title"],
                           "text": f'{doc["title"]}. {part}'})
    return chunks


def load_model(*, download: bool = False):
    """Use a committed local model dir; explicit opt-in downloads for preparation."""
    from sentence_transformers import SentenceTransformer
    if MODEL_DIR.is_dir():
        return SentenceTransformer(str(MODEL_DIR), local_files_only=True)
    if not download:
        raise RuntimeError("Model not cached locally. Run: python -m rag.prepare_model --download-model "
                           "once with internet, then rerun offline.")
    model = SentenceTransformer(MODEL_NAME)
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR))
    return model


def chroma_client():
    import chromadb
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(INDEX_DIR))


def index_all(model, client=None) -> dict[str, int]:
    """Only called by explicit build command. Both collections are separate."""
    client = client or chroma_client()
    policies = load_policies()
    out = {}
    for strategy, name in COLLECTIONS.items():
        chunks = make_chunks(policies, strategy)
        coll = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True).tolist()
        coll.upsert(ids=[c["id"] for c in chunks], embeddings=embeddings,
                    documents=[c["text"] for c in chunks],
                    metadatas=[{"doc_id": c["doc_id"], "title": c["title"]} for c in chunks])
        # Delete stale chunks from older generation so a rebuild is deterministic.
        actual_ids = set(coll.get()["ids"])
        stale = sorted(actual_ids - {c["id"] for c in chunks})
        if stale:
            coll.delete(ids=stale)
        out[strategy] = len(chunks)
    return out


@dataclass(frozen=True)
class Hit:
    doc_id: str
    title: str
    text: str
    similarity: float



def rerank_hits(query: str, hits: list[Hit]) -> list[Hit]:
    """Transparent intent-based ordering over the ORIGINAL retrieved hits.

    Keeps cosine similarity unchanged: the measured fallback threshold must
    continue to use raw cosine scores, not a boosted ranking score.
    """
    q = query.casefold()
    escalation_intent = (
        any(term in q for term in ("unresolved", "escalat", "incident"))
        and any(term in q for term in ("issue", "ticket", "support", "incident"))
    )

    def score(hit: Hit) -> float:
        bonus = 0.0
        if escalation_intent:
            if "escalation" in hit.title.casefold():
                bonus += 0.08
            if "escalated to" in hit.text.casefold():
                bonus += 0.03
        return hit.similarity + bonus

    # Python sort is stable for ties; raw values remain available for reporting.
    return sorted(hits, key=score, reverse=True)


def retrieve(query: str, model, *, strategy: str = "sentence", top_k: int = 3,
             client=None) -> list[Hit]:
    """Cosine similarity = 1 - cosine distance from explicitly cosine collection."""
    if strategy not in COLLECTIONS or top_k < 1:
        raise ValueError("Invalid strategy or top_k")
    client = client or chroma_client()
    coll = client.get_collection(COLLECTIONS[strategy])
    if coll.count() == 0:
        raise RuntimeError("Index is empty; run python -m rag.build_index")
    vector = model.encode([query], normalize_embeddings=True).tolist()
    result = coll.query(query_embeddings=vector, n_results=min(top_k, coll.count()),
                        include=["documents", "metadatas", "distances"])
    return [Hit(doc_id=m["doc_id"], title=m["title"], text=doc,
                similarity=max(-1.0, min(1.0, 1.0 - dist)))
            for m, doc, dist in zip(result["metadatas"][0], result["documents"][0],
                                     result["distances"][0])]


def grounded_answer(query: str, model, threshold: float, *, strategy: str = "sentence",
                    top_k: int = 3, client=None) -> dict:
    """Phase-2 extractive stand-in: policy text, never hallucinated generative claims.

    Phase 3 replaces the composer with the mandated CrewAI-compatible MOCK_LLM.
    """
    hits = rerank_hits(query, retrieve(query, model, strategy=strategy, top_k=top_k, client=client))
    if not hits or hits[0].similarity < threshold:
        return {"answer": FALLBACK, "grounded": False, "sources": [],
                "top_similarity": hits[0].similarity if hits else None}
    # Demonstrably grounded: returned text is exactly the top retrieved chunk.
    return {"answer": hits[0].text, "grounded": True,
            "sources": [hits[0].doc_id], "top_similarity": hits[0].similarity}
