"""Wiring test with fake LLM/web/RAG: plan -> approve (HITL) -> parallel agents -> verify -> report."""
import os
os.environ["NO_DB"] = "true"
os.environ["SANDBOX_MODE"] = "local"
import json
import pandas as pd, numpy as np
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from app.config import settings
from app.graph.builder import build_graph
from app.llm import client as llm
from app.memory import rag, longterm
from app.tools import web, sandbox
from app.agents import planner as P, researcher as R, verifier as V

ANALYST_CODE = '''
import os, json, pandas as pd
df = pd.read_csv(os.environ.get("DATA_DIR","/data")+"/input.csv")
q1,q3 = df.revenue.quantile([.25,.75]); iqr=q3-q1
bad = df[(df.revenue>q3+1.5*iqr)|(df.revenue<q1-1.5*iqr)]
print(json.dumps({"summary":"found","anomalies":[{"row":int(i),"column":"revenue","value":float(v),"reason":"iqr"} for i,v in bad.revenue.items()],"stats":{}}))
'''


@pytest.fixture(autouse=True)
def fakes(monkeypatch, tmp_path):
    settings.runs_dir = str(tmp_path); settings.sandbox_mode = "local"; settings.no_db = True
    settings.require_plan_approval = True
    def structured(schema, system, user, retries=3):
        n = schema.__name__
        if n == "Plan":
            return P.Plan(tasks=[P.Task(id="t1", description="find anomalies", agent="analyst"),
                                 P.Task(id="t2", description="research competitors", agent="researcher")])
        if n == "Queries": return R.Queries(queries=["acme competitors"])
        if n == "ResearchAnswer":
            return R.ResearchAnswer(summary="ok", findings=[R.Finding(claim="Acme leads", source_url="http://x.test/a")])
        if n == "Support": return V.Support(supported=[True])
    monkeypatch.setattr(llm, "structured", structured)
    monkeypatch.setattr(llm, "code", lambda s, u: ANALYST_CODE)
    monkeypatch.setattr(llm, "text", lambda s, u, temperature=0.2: "# Report\nAll good.")
    monkeypatch.setattr(web, "search", lambda q, n=5: [{"title": "x", "url": "http://x.test/a"}])
    monkeypatch.setattr(web, "fetch_text", lambda u, limit=0: "Acme is a market leader. " * 20)
    monkeypatch.setattr(rag, "index_chunks", lambda *a, **k: None)
    monkeypatch.setattr(rag, "hybrid_search", lambda *a, **k: [{"source": "http://x.test/a", "content": "Acme leads"}])
    monkeypatch.setattr(rag, "indexed_sources", lambda run_id: {"http://x.test/a"})
    monkeypatch.setattr(longterm, "recall", lambda g, k=3: [])
    monkeypatch.setattr(longterm, "remember", lambda *a, **k: None)


def _init(run_id):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"revenue": rng.normal(100, 5, 200)}); df.loc[10, "revenue"] = 900
    df.to_csv(sandbox.run_dir(run_id) / "data" / "input.csv", index=False)
    return {"run_id": run_id, "goal": "g", "has_dataset": True, "plan": [], "results": [], "critique": None,
            "retries": 0, "verified": False, "approved": False, "report": None}


def test_full_flow_with_hitl():
    g = build_graph(MemorySaver())
    cfg = {"configurable": {"thread_id": "r1"}}
    out = g.invoke(_init("r1"), cfg)
    assert out["__interrupt__"][0].value["type"] == "plan_approval"      # paused for human
    out = g.invoke(Command(resume={"approved": True}), cfg)
    assert out["verified"] is True and out["retries"] == 0
    assert {r["type"] for r in out["results"]} == {"analysis", "research"}
    assert 10 in [a["row"] for r in out["results"] if r["type"] == "analysis" for a in r["anomalies"]]
    assert out["report"].startswith("# Report")


def test_reject_plan_loops_back_to_planner():
    g = build_graph(MemorySaver())
    cfg = {"configurable": {"thread_id": "r2"}}
    g.invoke(_init("r2"), cfg)
    out = g.invoke(Command(resume={"approved": False, "feedback": "add pricing"}), cfg)
    assert out["__interrupt__"]                                          # asks again with new plan


def test_verifier_catches_hallucinated_anomaly(monkeypatch):
    bad = ANALYST_CODE.replace('"row":int(i)', '"row":int(i)+1')        # wrong row index
    monkeypatch.setattr(llm, "code", lambda s, u: bad)
    settings.require_plan_approval = False; settings.max_retries = 1
    g = build_graph(MemorySaver())
    out = g.invoke(_init("r3"), {"configurable": {"thread_id": "r3"}, "recursion_limit": 60})
    assert out["verified"] is False and out["retries"] == 2              # retried, then capped
    assert "UNVERIFIED" in out["report"]
