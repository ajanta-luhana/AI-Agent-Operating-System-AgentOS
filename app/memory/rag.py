"""pgvector store: chunking, ingestion, hybrid (vector + full-text) search with RRF."""
from app.llm import client as llm
from app.memory.db import get_pool


def chunk_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return [c for c in out if len(c) > 80]


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def index_chunks(run_id: str | None, source: str, chunks: list[str]):
    if not chunks:
        return
    embs = []
    for i in range(0, len(chunks), 32):
        embs += llm.embed(chunks[i:i + 32])
    with get_pool().connection() as c, c.cursor() as cur:
        cur.executemany(
            "INSERT INTO documents (run_id, source, content, embedding) VALUES (%s,%s,%s,%s::vector)",
            [(run_id, source, ch, _vec(e)) for ch, e in zip(chunks, embs)])


def hybrid_search(query: str, k: int = 6, run_id: str | None = None,
                  source_prefix: str | None = None) -> list[dict]:
    where, params = ["TRUE"], []
    if run_id:
        where.append("run_id = %s"); params.append(run_id)
    if source_prefix:
        where.append("source LIKE %s"); params.append(source_prefix + "%")
    w = " AND ".join(where)
    qv = _vec(llm.embed([query])[0])
    with get_pool().connection() as c:
        vec = c.execute(f"SELECT id, source, content FROM documents WHERE {w} "
                        f"ORDER BY embedding <=> %s::vector LIMIT %s", [*params, qv, k * 2]).fetchall()
        fts = c.execute(f"SELECT id, source, content FROM documents WHERE {w} "
                        f"AND tsv @@ plainto_tsquery('english', %s) "
                        f"ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC LIMIT %s",
                        [*params, query, query, k * 2]).fetchall()
    scores, rows = {}, {}
    for lst in (vec, fts):                       # Reciprocal Rank Fusion
        for rank, r in enumerate(lst):
            scores[r["id"]] = scores.get(r["id"], 0) + 1 / (60 + rank)
            rows[r["id"]] = r
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return [rows[i] for i in top]


def indexed_sources(run_id: str) -> set[str]:
    with get_pool().connection() as c:
        return {r["source"] for r in c.execute(
            "SELECT DISTINCT source FROM documents WHERE run_id = %s", [run_id]).fetchall()}
