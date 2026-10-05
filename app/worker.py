"""Worker: consumes jobs from Redis, runs the LangGraph, handles pause/resume (HITL)."""
import json, logging, traceback
import redis
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from app.config import settings
from app.graph.builder import build_graph
from app.memory.db import get_pool, init_schema, update_run
from app.observability.events import emit

logging.basicConfig(level=logging.INFO)
QUEUE = "agentos:jobs"


def process(graph, job: dict):
    run_id = job["run_id"]
    cfg = {"configurable": {"thread_id": run_id}, "recursion_limit": 50}
    try:
        update_run(run_id, status="running")
        emit(run_id, "run", "status", {"status": "running"})
        if job["type"] == "start":
            inp = {"run_id": run_id, "goal": job["goal"], "has_dataset": job["has_dataset"],
                   "plan": [], "results": [], "critique": None, "retries": 0,
                   "verified": False, "approved": False, "report": None}
            result = graph.invoke(inp, cfg)
        else:  # resume after human decision
            result = graph.invoke(Command(resume=job["decision"]), cfg)

        if result.get("__interrupt__"):
            payload = result["__interrupt__"][0].value
            update_run(run_id, status="awaiting_approval", plan=payload.get("plan"))
            emit(run_id, "run", "status", {"status": "awaiting_approval", "plan": payload.get("plan")})
        else:
            update_run(run_id, status="completed", report=result.get("report"),
                       plan=result.get("plan"))
            emit(run_id, "run", "status", {"status": "completed", "verified": result.get("verified")})
    except Exception as e:
        traceback.print_exc()
        update_run(run_id, status="failed", error=repr(e))
        emit(run_id, "run", "status", {"status": "failed", "error": repr(e)})


def main():
    init_schema()
    saver = PostgresSaver(get_pool())
    saver.setup()
    graph = build_graph(saver)
    r = redis.Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=30)
    logging.info("worker ready")
    while True:
        item = r.blpop(QUEUE, timeout=5)
        if item:
            process(graph, json.loads(item[1]))


if __name__ == "__main__":
    main()
