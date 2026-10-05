"""Audit log (Postgres agent_events) + live pub/sub (Redis) for SSE."""
import json, logging
from app.config import settings

log = logging.getLogger("agentos")
_redis = None


def _r():
    global _redis
    if _redis is None:
        import redis
        _redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def emit(run_id, node, kind, payload=None, latency_ms=None):
    log.info("run=%s node=%s kind=%s %s", run_id, node, kind, payload or "")
    if settings.no_db or not run_id:
        return
    try:
        from app.memory.db import get_pool
        with get_pool().connection() as c:
            row = c.execute(
                "INSERT INTO agent_events (run_id,node,kind,latency_ms,payload) VALUES (%s,%s,%s,%s,%s) RETURNING id, ts",
                [run_id, node, kind, latency_ms, json.dumps(payload or {}, default=str)]).fetchone()
        _r().publish(f"run:{run_id}", json.dumps(
            {"id": row["id"], "ts": str(row["ts"]), "node": node, "kind": kind,
             "latency_ms": latency_ms, "payload": payload or {}}, default=str))
    except Exception as e:                       # observability must never break the run
        log.warning("emit failed: %s", e)
