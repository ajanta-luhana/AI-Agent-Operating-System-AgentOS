import json
from pydantic import BaseModel
from app.llm import client as llm
from app.tools import sandbox
from app.observability.tracing import traced


class Anomaly(BaseModel):
    row: int
    column: str
    value: float
    reason: str = ""


class AnalysisOutput(BaseModel):
    summary: str
    anomalies: list[Anomaly] = []
    stats: dict = {}


PROFILE_CODE = """
import os, pandas as pd
df = pd.read_csv(os.environ.get('DATA_DIR','/data') + '/input.csv')
print('shape:', df.shape); print(df.dtypes.to_string()); print(df.head(5).to_string())
print(df.describe(include='all').T.head(20).to_string())
"""

SYSTEM = """You write Python 3 data-analysis code. Rules:
- Read CSV: pd.read_csv(os.environ.get('DATA_DIR','/data') + '/input.csv'). No network.
- Save charts as PNG into os.environ.get('OUT_DIR','/out') (matplotlib, Agg backend).
- Libraries: pandas, numpy, scipy, sklearn, matplotlib.
- For anomalies combine robust methods (IQR / modified z-score, optionally IsolationForest).
- Your LAST printed line must be ONE JSON object:
  {"summary": str, "anomalies": [{"row": int (0-based row position in CSV), "column": str, "value": number, "reason": str}], "stats": {}}
- Output ONLY code."""


def last_json_line(stdout: str) -> str:
    for line in reversed(stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            return line.strip()
    raise ValueError("no JSON line in stdout")


@traced("analyst")
def analyst(inp):
    run_id, task = inp["run_id"], inp["task"]
    base = {"task_id": task["id"], "type": "analysis"}
    if not sandbox.has_dataset(run_id):
        return {"results": [{**base, "error": "no dataset uploaded"}]}
    prof = sandbox.run_python(PROFILE_CODE, run_id, "profile")
    user = f"Task: {task['description']}\n\nDataset profile:\n{prof.stdout}"
    last_err, out = "", None
    for attempt in range(3):
        code = llm.code(SYSTEM, user + (f"\n\nPrevious attempt failed, fix it:\n{last_err}" if last_err else ""))
        res = sandbox.run_python(code, run_id, f"{task['id']}_{attempt}")
        if not res.ok:
            last_err = res.stderr[-1500:]
            continue
        try:
            out = AnalysisOutput.model_validate_json(last_json_line(res.stdout))
            break
        except Exception as e:
            last_err = f"Final line was not valid JSON ({e}). stdout tail: {res.stdout[-400:]}"
    if out is None:
        return {"results": [{**base, "error": f"analysis failed after 3 attempts: {last_err[:500]}"}]}
    return {"results": [{**base, "summary": out.summary,
                         "anomalies": [a.model_dump() for a in out.anomalies[:200]],
                         "n_anomalies": len(out.anomalies), "stats": out.stats,
                         "charts": sandbox.list_outputs(run_id)}]}
