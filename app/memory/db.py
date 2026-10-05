import json
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from app.config import settings

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(settings.database_url, min_size=1, max_size=10, open=True,
                               kwargs={"autocommit": True, "row_factory": dict_row})
    return _pool


def schema_sql() -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, goal TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
  plan JSONB, report TEXT, error TEXT, has_dataset BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT now(), updated_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE IF NOT EXISTS agent_events (
  id BIGSERIAL PRIMARY KEY, run_id TEXT, ts TIMESTAMPTZ DEFAULT now(),
  node TEXT, kind TEXT, latency_ms INT, payload JSONB);
CREATE INDEX IF NOT EXISTS ev_run ON agent_events(run_id, id);
CREATE TABLE IF NOT EXISTS documents (
  id BIGSERIAL PRIMARY KEY, run_id TEXT, source TEXT NOT NULL, content TEXT NOT NULL,
  embedding vector({settings.embed_dim}),
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  created_at TIMESTAMPTZ DEFAULT now());
CREATE INDEX IF NOT EXISTS doc_run ON documents(run_id);
CREATE INDEX IF NOT EXISTS doc_tsv ON documents USING gin(tsv);
CREATE INDEX IF NOT EXISTS doc_vec ON documents USING hnsw (embedding vector_cosine_ops);
"""


def init_schema():
    with get_pool().connection() as c:
        c.execute(schema_sql())


def update_run(run_id: str, **fields):
    sets, vals = [], []
    for k, v in fields.items():
        sets.append(f"{k} = %s")
        vals.append(json.dumps(v) if k == "plan" and v is not None else v)
    sets.append("updated_at = now()")
    with get_pool().connection() as c:
        c.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id = %s", [*vals, run_id])
