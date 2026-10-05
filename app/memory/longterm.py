"""Long-term memory: summaries of past runs, retrieved by the planner."""
from app.memory import rag


def remember(run_id: str, goal: str, summary: str):
    rag.index_chunks(None, f"memory:{run_id}", [f"Goal: {goal}\nOutcome: {summary}"[:1500].ljust(90)])


def recall(goal: str, k: int = 3) -> list[str]:
    try:
        return [h["content"] for h in rag.hybrid_search(goal, k=k, source_prefix="memory:")]
    except Exception:
        return []
