from typing import Literal
from pydantic import BaseModel
from app.llm import client as llm
from app.memory import longterm
from app.observability.tracing import traced


class Task(BaseModel):
    id: str
    description: str
    agent: Literal["analyst", "researcher"]


class Plan(BaseModel):
    tasks: list[Task]


SYSTEM = """You are the Planner of a multi-agent system. Break the user's goal into 1-5 independent subtasks.
Agents:
- analyst: runs Python on the user's uploaded CSV (profiling, anomaly detection, statistics, charts).
- researcher: web research + document retrieval (competitors, market info) with cited sources.
Each description must be specific and self-contained. Task ids: t1, t2, ...
{dataset_note}"""


@traced("planner")
def planner(state):
    note = ("A CSV dataset IS uploaded." if state.get("has_dataset")
            else "NO dataset uploaded: do not create analyst tasks.")
    parts = [f"Goal: {state['goal']}"]
    past = longterm.recall(state["goal"]) if not state.get("critique") else []
    if past:
        parts.append("Relevant past runs:\n" + "\n---\n".join(past))
    if state.get("plan") and state.get("critique"):
        parts.append(f"Previous plan: {state['plan']}\nCritique to address: {state['critique']}")
    plan = llm.structured(Plan, SYSTEM.format(dataset_note=note), "\n\n".join(parts))
    tasks = [t.model_dump() for t in plan.tasks][:5]
    if not state.get("has_dataset"):
        tasks = [t for t in tasks if t["agent"] != "analyst"]
    return {"plan": tasks, "results": [{"_reset": True}]}
