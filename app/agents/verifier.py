"""Verifier: deterministic recomputation for numbers + source-grounding for claims."""
import json
from pydantic import BaseModel
from app.config import settings
from app.llm import client as llm
from app.memory import rag
from app.tools import sandbox
from app.agents.analyst import last_json_line
from app.observability.tracing import traced

RECOMPUTE = """
import os, json, pandas as pd, numpy as np
flagged = json.loads({payload})
df = pd.read_csv(os.environ.get('DATA_DIR','/data') + '/input.csv')
checked = match = confirmed = 0
for a in flagged:
    r, c = a.get('row'), a.get('column')
    if c not in df.columns or not isinstance(r, int) or not (0 <= r < len(df)):
        checked += 1; continue
    checked += 1
    s = pd.to_numeric(df[c], errors='coerce'); v = s.iloc[r]
    if pd.isna(v) or not np.isclose(v, a.get('value', np.nan), rtol=1e-3, atol=1e-6): continue
    match += 1
    q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    med = s.median(); mad = (s - med).abs().median() or 1e-9
    if v > q3 + 1.5*iqr or v < q1 - 1.5*iqr or abs(0.6745*(v-med)/mad) > 3.5: confirmed += 1
print(json.dumps({{'checked': checked, 'value_match': match, 'confirmed': confirmed}}))
"""


class Support(BaseModel):
    supported: list[bool]


class Verdict(BaseModel):
    ok: bool
    reason: str = ""


def _check_analysis(run_id, r) -> list[str]:
    if r.get("error"):
        return [f"[{r['task_id']}] analysis error: {r['error']}"]
    items = r.get("anomalies", [])
    if not items:
        return []
    code = RECOMPUTE.format(payload=json.dumps(json.dumps(items)))
    res = sandbox.run_python(code, run_id, f"verify_{r['task_id']}")
    if not res.ok:
        return [f"[{r['task_id']}] verification code failed: {res.stderr[-300:]}"]
    v = json.loads(last_json_line(res.stdout))
    probs = []
    if v["value_match"] < v["checked"]:
        probs.append(f"[{r['task_id']}] {v['checked'] - v['value_match']}/{v['checked']} flagged rows don't match the data "
                     f"(wrong row index, column or value).")
    elif v["confirmed"] / max(v["checked"], 1) < 0.5:
        probs.append(f"[{r['task_id']}] only {v['confirmed']}/{v['checked']} flagged rows are statistical outliers (IQR/MAD).")
    return probs


def _check_research(run_id, r) -> list[str]:
    tid, probs = r["task_id"], []
    if not r.get("findings"):
        return [f"[{tid}] research returned no findings; broaden queries."]
    known = rag.indexed_sources(run_id)
    bad = [f["source_url"] for f in r["findings"] if f["source_url"] not in known]
    if bad:
        probs.append(f"[{tid}] findings cite sources that were never retrieved: {bad[:3]}")
    pairs = []
    for f in r["findings"][:8]:
        ev = rag.hybrid_search(f["claim"], k=2, run_id=run_id)
        pairs.append({"claim": f["claim"], "evidence": " ".join(e["content"] for e in ev)[:1500]})
    s = llm.structured(Support, "For each claim decide if the evidence supports it. Return one boolean per claim, in order.",
                       json.dumps(pairs))
    unsupported = [p["claim"] for p, ok in zip(pairs, s.supported) if not ok]
    if len(unsupported) > len(pairs) / 3:
        probs.append(f"[{tid}] unsupported claims: {unsupported[:3]}")
    return probs


@traced("verifier")
def verifier(state):
    run_id, problems = state["run_id"], []
    for r in state.get("results", []):
        try:
            problems += _check_analysis(run_id, r) if r["type"] == "analysis" else _check_research(run_id, r)
        except Exception as e:
            problems.append(f"[{r.get('task_id')}] verifier crashed: {e!r}")
    if not state.get("results"):
        problems.append("no results produced")
    ok = not problems
    return {"verified": ok, "critique": None if ok else "\n".join(problems),
            "retries": state.get("retries", 0) + (0 if ok else 1)}
