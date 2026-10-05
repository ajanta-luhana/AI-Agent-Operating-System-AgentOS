from app.llm import client as llm
from app.observability.tracing import traced

SYS = """You are a reflection agent. Given a failed verification, write a short, concrete critique
(max 6 lines) telling the planner which tasks to redo and how (e.g. different method, more specific search queries)."""


@traced("reflector")
def reflector(state):
    crit = llm.text(SYS, f"Goal: {state['goal']}\nPlan: {state['plan']}\nVerification problems:\n{state['critique']}")
    return {"critique": f"{state['critique']}\nGuidance: {crit}"}
