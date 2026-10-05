from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from app.config import settings
from app.agents.planner import planner
from app.agents.analyst import analyst
from app.agents.researcher import researcher
from app.agents.verifier import verifier
from app.agents.reflector import reflector
from app.agents.reporter import reporter
from app.graph.state import State
from app.observability.tracing import traced


@traced("approve_plan")
def approve_plan(state):
    # Human-in-the-loop gate #1: approve the plan (first pass only; retries auto-continue).
    if not settings.require_plan_approval or state.get("retries", 0) > 0:
        return {"approved": True}
    decision = interrupt({"type": "plan_approval", "plan": state["plan"]})
    if decision.get("approved", False):
        return {"approved": True}
    return {"approved": False, "critique": decision.get("feedback") or "Human rejected the plan; revise it."}


def route_after_approval(state):
    if not state.get("approved"):
        return "planner"
    if not state["plan"]:
        return "reporter"
    return [Send(t["agent"], {"run_id": state["run_id"], "goal": state["goal"], "task": t})
            for t in state["plan"]]


def route_after_verify(state):
    if not state.get("verified") and state.get("retries", 0) <= settings.max_retries:
        return "reflector"
    return "reporter"


def build_graph(checkpointer=None):
    g = StateGraph(State)
    for name, fn in [("planner", planner), ("approve_plan", approve_plan), ("analyst", analyst),
                     ("researcher", researcher), ("verifier", verifier), ("reflector", reflector),
                     ("reporter", reporter)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_edge("planner", "approve_plan")
    g.add_conditional_edges("approve_plan", route_after_approval,
                            ["planner", "analyst", "researcher", "reporter"])
    g.add_edge("analyst", "verifier")
    g.add_edge("researcher", "verifier")
    g.add_conditional_edges("verifier", route_after_verify, ["reflector", "reporter"])
    g.add_edge("reflector", "planner")
    g.add_edge("reporter", END)
    return g.compile(checkpointer=checkpointer)
