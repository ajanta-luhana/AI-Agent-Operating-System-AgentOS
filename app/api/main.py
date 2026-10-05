import json, uuid
from contextlib import asynccontextmanager
from pathlib import Path
import redis.asyncio as aredis
import redis as sredis
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from app.config import settings
from app.memory.db import get_pool, init_schema
from app.tools.sandbox import run_dir

QUEUE = "agentos:jobs"
STATIC = Path(__file__).resolve().parent.parent / "static"
sync_redis = sredis.Redis.from_url(settings.redis_url, decode_responses=True)


@asynccontextmanager
async def lifespan(app):
    await run_in_threadpool(init_schema)
    yield

app = FastAPI(title="AgentOS", lifespan=lifespan)


class Decision(BaseModel):
    approved: bool = True
    feedback: str | None = None


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/goals")
async def create_goal(goal: str = Form(...), file: UploadFile | None = File(None)):
    run_id = uuid.uuid4().hex[:12]
    has_ds = False
    if file is not None:
        if not (file.filename or "").lower().endswith(".csv"):
            raise HTTPException(400, "only .csv supported")
        data = await file.read()
        if len(data) > 50_000_000:
            raise HTTPException(413, "file too large (50MB max)")
        (run_dir(run_id) / "data" / "input.csv").write_bytes(data)
        has_ds = True
    def _insert():
        with get_pool().connection() as c:
            c.execute("INSERT INTO runs (id, goal, has_dataset) VALUES (%s,%s,%s)", [run_id, goal, has_ds])
    await run_in_threadpool(_insert)
    sync_redis.rpush(QUEUE, json.dumps({"type": "start", "run_id": run_id, "goal": goal, "has_dataset": has_ds}))
    return {"run_id": run_id, "status": "queued"}


def _get_run(run_id):
    with get_pool().connection() as c:
        return c.execute("SELECT * FROM runs WHERE id=%s", [run_id]).fetchone()


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    r = _get_run(run_id)
    if not r:
        raise HTTPException(404)
    return r


@app.post("/runs/{run_id}/approve")
def approve(run_id: str, d: Decision):
    r = _get_run(run_id)
    if not r:
        raise HTTPException(404)
    if r["status"] != "awaiting_approval":
        raise HTTPException(409, f"run is {r['status']}, not awaiting approval")
    sync_redis.rpush(QUEUE, json.dumps({"type": "resume", "run_id": run_id, "decision": d.model_dump()}))
    return {"status": "resuming"}


@app.get("/runs/{run_id}/events")
def events(run_id: str):
    with get_pool().connection() as c:
        return c.execute("SELECT id, ts, node, kind, latency_ms, payload FROM agent_events "
                         "WHERE run_id=%s ORDER BY id", [run_id]).fetchall()


@app.get("/runs/{run_id}/stream")
async def stream(run_id: str):
    ar = aredis.Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=60)

    async def gen():
        ps = ar.pubsub()
        await ps.subscribe(f"run:{run_id}")           # subscribe first, then replay -> no gap
        seen = set()
        done = False
        for e in await run_in_threadpool(events, run_id):
            seen.add(e["id"])
            yield f"data: {json.dumps(e, default=str)}\n\n"
            done = e["node"] == "run" and e["payload"].get("status") in ("completed", "failed")
        while not done:
            msg = await ps.get_message(ignore_subscribe_messages=True, timeout=15)
            if msg is None:
                yield ": keepalive\n\n"
                continue
            e = json.loads(msg["data"])
            if e["id"] in seen:
                continue
            seen.add(e["id"])
            yield f"data: {json.dumps(e)}\n\n"
            done = e["node"] == "run" and e["payload"].get("status") in ("completed", "failed")
        await ps.close(); await ar.close()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/runs/{run_id}/report", response_class=PlainTextResponse)
def report(run_id: str):
    r = _get_run(run_id)
    if not r or not r["report"]:
        raise HTTPException(404, "report not ready")
    return r["report"]


@app.get("/runs/{run_id}/files/{name}")
def files(run_id: str, name: str):
    p = (run_dir(run_id) / "out" / name).resolve()
    if p.parent != (run_dir(run_id) / "out").resolve() or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)
