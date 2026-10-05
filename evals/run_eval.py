"""Offline eval: run full graph (in-memory checkpointer, auto-approve) on planted-anomaly data.
Metrics: anomaly precision/recall + LLM-as-judge (faithfulness, completeness). Needs Ollama + Postgres(pgvector)."""
import json, shutil, sys, uuid
from pathlib import Path
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
from app.config import settings
settings.require_plan_approval = False
from app.graph.builder import build_graph
from app.llm import client as llm
from app.memory.db import init_schema
from app.tools.sandbox import run_dir
from evals.make_dataset import make

GOAL = "Analyze my company sales data, find anomalies, research competitors and prepare a report."


class Judge(BaseModel):
    faithfulness: int   # 1-5: report claims supported by results
    completeness: int   # 1-5: covers anomalies, research, recommendations
    notes: str


def main(min_recall=0.7, min_precision=0.5):
    init_schema()
    planted = set(make())
    run_id = "eval-" + uuid.uuid4().hex[:6]
    shutil.copy("evals/sales.csv", run_dir(run_id) / "data" / "input.csv")
    g = build_graph(MemorySaver())
    out = g.invoke({"run_id": run_id, "goal": GOAL, "has_dataset": True, "plan": [], "results": [],
                    "critique": None, "retries": 0, "verified": False, "approved": False, "report": None},
                   {"configurable": {"thread_id": run_id}, "recursion_limit": 50})
    flagged = {a["row"] for r in out["results"] if r["type"] == "analysis" for a in r.get("anomalies", [])}
    tp = len(flagged & planted)
    recall, precision = tp / len(planted), (tp / len(flagged) if flagged else 0)
    j = llm.structured(Judge, "Score the report 1-5 on faithfulness and completeness vs the results.",
                       f"RESULTS:\n{json.dumps(out['results'], default=str)[:8000]}\n\nREPORT:\n{out['report'][:6000]}")
    print(json.dumps({"recall": round(recall, 2), "precision": round(precision, 2), "verified": out["verified"],
                      "retries": out["retries"], "judge": j.model_dump()}, indent=2))
    ok = recall >= min_recall and precision >= min_precision and j.faithfulness >= 3
    sys.exit(0 if ok else 1)          # use as CI gate on every prompt/model change


if __name__ == "__main__":
    main()
