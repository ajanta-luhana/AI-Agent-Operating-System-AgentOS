import json
from pathlib import Path
from app.llm import client as llm
from app.memory import longterm
from app.tools import sandbox
from app.observability.tracing import traced

SYS = """You write a professional business report in Markdown from VERIFIED results only. Sections:
# Title, ## Executive Summary, ## Data Analysis & Anomalies (table of top anomalies), ## Competitor / Market Research,
## Recommendations, ## Limitations. Do not invent numbers or sources. Be concise."""


@traced("reporter")
def reporter(state):
    results = state.get("results", [])
    compact = json.dumps([{k: v for k, v in r.items() if k != "anomalies"} | (
        {"top_anomalies": r["anomalies"][:15]} if r.get("anomalies") else {}) for r in results], default=str)[:12000]
    body = llm.text(SYS, f"Goal: {state['goal']}\n\nResults:\n{compact}")
    banner = "" if state.get("verified") else (
        "> ⚠️ **UNVERIFIED**: automated verification still failed after max retries. "
        f"Open issues:\n> {str(state.get('critique'))[:600]}\n\n")
    charts = [c for r in results for c in r.get("charts", [])]
    chart_md = "\n## Charts\n" + "\n".join(f"![{c}](files/{c})" for c in sorted(set(charts))) if charts else ""
    sources = sorted({s for r in results for s in r.get("sources", [])})
    src_md = "\n## Sources\n" + "\n".join(f"- {s}" for s in sources) if sources else ""
    report = banner + body + chart_md + src_md
    (sandbox.run_dir(state["run_id"]) / "report.md").write_text(report)
    try:
        longterm.remember(state["run_id"], state["goal"], body[:600])
    except Exception:
        pass
    return {"report": report}
